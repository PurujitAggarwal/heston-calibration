"""Daily paper-trading orchestrator: fetch → price → signal → execute → report.

Chains the paper-trading modules into one entry point
(``python -m src.paper_trading.daily``) that runs a single trading day:

    1. pull today's SPX option chain from IBKR (:func:`data_live.run`);
    2. imply the zero curve and dividend yield from the chain by put-call
       parity (:func:`data_live.imply_rates_and_dividends`);
    3. price every OTM quote with the Heston parameters governing today and
       flag short-vol signals (:func:`build_flagged_panel`, reusing
       :mod:`src.heston.calibration` and :mod:`src.heston.signals`);
    4. classify the volatility regime from trailing SPX closes
       (:func:`classify_regime`, reusing :func:`backtest.high_vol_regime`);
    5. execute the day against the persistent paper book
       (:func:`execution.execute_daily`);
    6. record the daily equity snapshot and save the book;
    7. email the daily report (:func:`reporter.send_daily_report`).

Recalibration: today's quotes are priced with the parameters from the existing
rolling calibration (``heston_params.parquet``) via
:func:`surface.params_for_date` — the latest converged quarter at or before
today, which is always calibrated strictly before its quarter, so there is no
lookahead. A freshly started live book has no year of stored live chains to
recalibrate from, so it reuses the research pipeline's quarterly parameters
(matching the backtest's quarterly-recalibration rule) until enough live
history accumulates.

Every collaborator is dependency-injected (``chain``, ``panel``, ``params``,
``portfolio``, ``is_high_vol``, ``ib``, ``send``), so the execute → record →
report path is fully testable offline with ``ib=None`` and ``send=False``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.heston.backtest import REGIME_LOOKBACK, high_vol_regime
from src.heston.calibration import (
    PARAMS_PARQUET,
    build_calibration_sample,
    model_iv_for_sample,
)
from src.heston.fft import HestonParams
from src.heston.signals import PANEL_EXTRA_COLUMNS, add_signal_flags
from src.heston.surface import params_for_date
from src.paper_trading import reporter
from src.paper_trading.data_live import (
    LIVE_CHAINS_DIR,
    imply_rates_and_dividends,
    load_stored_chains,
    trading_date_now,
)
from src.paper_trading.data_live import run as fetch_chain
from src.paper_trading.execution import execute_daily
from src.paper_trading.portfolio import Portfolio

logger = logging.getLogger("paper_trading.daily")

# --- Config -----------------------------------------------------------------------
EVAL_WINDOW_DAYS: int = 1  # today's evaluation window is [today, today + 1 day)
REGIME_HISTORY_DAYS: int = 500  # calendar days of stored chains for the regime series


def build_flagged_panel(
    chain: pd.DataFrame,
    params: HestonParams,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    today: pd.Timestamp,
) -> pd.DataFrame:
    """Model-price and signal-flag today's OTM quotes for execution.

    Builds the single-day evaluation sample (every OTM quote, every expiry),
    attaches Heston model IVs and deviations, then the entry/exit/liquidity
    flags — the flagged, model-priced panel :func:`execute_daily` consumes.

    Args:
        chain: Today's live chain.
        params: Heston parameters governing today.
        zero_curve: Zero curve implied from the chain.
        div_yield: Dividend yield implied from the chain.
        today: Trading date.

    Returns:
        Flagged panel (empty frame if today has no usable OTM quotes).
    """
    sample = build_calibration_sample(
        chain,
        zero_curve,
        div_yield,
        today,
        today + pd.Timedelta(days=EVAL_WINDOW_DAYS),
        month_ends_only=False,
        nearest_tenors_only=False,
        extra_columns=PANEL_EXTRA_COLUMNS,
    )
    if sample.empty:
        return sample
    sample = sample.copy()
    sample["model_iv"] = model_iv_for_sample(params, sample)
    sample["deviation"] = sample["market_iv"] - sample["model_iv"]
    return add_signal_flags(sample)


def trailing_spx_closes(
    today: pd.Timestamp,
    directory=LIVE_CHAINS_DIR,
    history_days: int = REGIME_HISTORY_DAYS,
) -> pd.DataFrame:
    """One SPX close per stored trading date within the trailing window.

    Args:
        today: Trading date (inclusive window end).
        directory: Directory of stored daily chains.
        history_days: Calendar days of history to load.

    Returns:
        Frame with ``date`` and ``close`` columns (empty if no stored chains).
    """
    start = today - pd.Timedelta(days=history_days)
    chains = load_stored_chains(start, today, directory)
    if chains.empty:
        return pd.DataFrame(columns=["date", "close"])
    return (
        chains.groupby("date")["spot"]
        .first()
        .reset_index()
        .rename(columns={"spot": "close"})
    )


def classify_regime(
    today: pd.Timestamp, chain: pd.DataFrame, directory=LIVE_CHAINS_DIR
) -> bool:
    """High-vol regime flag for today from trailing plus today's SPX close.

    Args:
        today: Trading date.
        chain: Today's chain (its spot supplies today's close).
        directory: Directory of stored daily chains.

    Returns:
        True if today is classified high-vol; calm (False) until enough
        trailing history exists (:func:`backtest.high_vol_regime`).
    """
    closes = trailing_spx_closes(today, directory)
    today_close = float(chain["spot"].iloc[0]) if not chain.empty else float("nan")
    today_row = pd.DataFrame([{"date": pd.Timestamp(today), "close": today_close}])
    if closes.empty:
        closes = today_row
    else:
        closes = pd.concat([closes, today_row], ignore_index=True).drop_duplicates(
            "date", keep="last"
        )
    flags = high_vol_regime(closes)
    return bool(flags.reindex([pd.Timestamp(today)]).fillna(False).iloc[0])


def run_day(
    today: pd.Timestamp | None = None,
    ib: Any | None = None,
    *,
    chain: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    params: HestonParams | None = None,
    params_table: pd.DataFrame | None = None,
    portfolio: Portfolio | None = None,
    is_high_vol: bool | None = None,
    send: bool = True,
    password: str | None = None,
) -> dict[str, Any]:
    """Run one paper-trading day end to end.

    Any collaborator may be injected; otherwise it is fetched/derived. Passing
    ``ib=None`` books fills at the quoted mid (paper simulation); ``send=False``
    skips the email.

    Args:
        today: Trading date; defaults to the current US Eastern date.
        ib: Connected IBKR instance for live routing, or None to book at mid.
        chain: Today's chain; fetched via :func:`data_live.run` when None.
        panel: Pre-built flagged panel; built from ``chain`` when None.
        params: Heston parameters for today; from ``params_table`` when None.
        params_table: Rolling calibration; read from disk when None and needed.
        portfolio: The book; loaded/created when None.
        is_high_vol: Regime flag; classified from trailing closes when None.
        send: Whether to email the daily report.
        password: Gmail app password; read from the secrets file when None.

    Returns:
        Summary dict with the date, regime flag, opened/closed records and the
        marked equity after the day.

    Raises:
        ValueError: If no converged Heston parameters govern ``today``.
    """
    today = trading_date_now() if today is None else pd.Timestamp(today).normalize()

    if chain is None:
        chain, _ = fetch_chain(today)

    if params is None:
        if params_table is None:
            params_table = pd.read_parquet(PARAMS_PARQUET)
        params = params_for_date(params_table, today)
        if params is None:
            raise ValueError(f"no converged Heston parameters govern {today:%Y-%m-%d}")

    if panel is None:
        zero_curve, div_yield = imply_rates_and_dividends(chain)
        panel = build_flagged_panel(chain, params, zero_curve, div_yield, today)

    if is_high_vol is None:
        is_high_vol = classify_regime(today, chain)

    if portfolio is None:
        portfolio = Portfolio.load_or_new()

    summary = execute_daily(portfolio, panel, params, is_high_vol, today, ib=ib)

    spx_close = float(chain["spot"].iloc[0]) if not chain.empty else None
    portfolio.record_equity(today, spx_close)
    portfolio.save()

    if send:
        reporter.send_daily_report(portfolio, as_of=today, password=password)

    logger.info(
        "%s: high_vol=%s opened=%d closed=%d equity=%.2f",
        today.date(),
        is_high_vol,
        summary["n_opened"],
        summary["n_closed"],
        portfolio.equity,
    )
    return {
        "date": today,
        "is_high_vol": is_high_vol,
        "equity": portfolio.equity,
        **summary,
    }


def main() -> None:
    """CLI entry point: ``python -m src.paper_trading.daily``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run_day()
    print(
        f"{result['date']:%Y-%m-%d}: opened {result['n_opened']}, "
        f"closed {result['n_closed']}, equity ${result['equity']:,.2f}"
    )


if __name__ == "__main__":
    main()
