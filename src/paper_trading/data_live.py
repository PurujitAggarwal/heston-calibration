"""Live SPX option-chain adapter — Interactive Brokers replacement for WRDS.

Pulls the SPX option chain from IBKR (via ``ib_insync``) once per trading day
and writes it, in the exact schema produced by :mod:`src.heston.data`, to
``data/processed/live_chains/{YYYY-MM-DD}.parquet`` so the existing calibration,
signal and pricing code consumes it unchanged.

Differences from the OptionMetrics pull, and how they are bridged:
    - Market implied vol / delta / vega come from IBKR's own option-computation
      ticks (``Ticker.modelGreeks``) rather than an OptionMetrics column.
    - IBKR does not publish a zero-coupon curve or an implied dividend yield.
      Both are backed out of the chain itself by put-call parity
      (:func:`imply_rates_and_dividends`): across near-ATM strikes at one
      maturity, ``C - K/P = S e^{-qT} - K e^{-rT}`` is linear in the strike, so
      a least-squares line gives ``e^{-rT}`` (slope) and ``S e^{-qT}``
      (intercept), hence ``r`` and ``q``.

Only the AM-settled monthly SPX class is pulled (trading class ``SPX``),
matching the historical ``am_settlement = 1`` / no-weeklies filter.

``ib_insync`` is imported lazily inside the connection helpers, exactly as
:mod:`src.heston.data` imports ``wrds``, so the offline test suite needs no
IBKR install and no running TWS.

Run (requires TWS running and logged into the paper account):
    python -m src.paper_trading.data_live
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.heston.calibration import DAYS_PER_YEAR, PERCENT
from src.heston.data import (
    DAYS_PER_MONTH,
    MONEYNESS_MAX,
    MONEYNESS_MIN,
    PROJECT_ROOT,
    add_derived_columns,
    drop_crossed_quotes,
    validate_chain,
)

logger = logging.getLogger("paper_trading.data_live")

# --- IBKR connection --------------------------------------------------------------
IBKR_HOST: str = "127.0.0.1"
IBKR_PORT: int = 7497  # TWS paper-trading socket
IBKR_ACCOUNT: str = "DUR195917"  # paper account
DATA_CLIENT_ID: int = 11  # client id reserved for the data adapter
CONNECT_TIMEOUT_SECONDS: float = 15.0

# --- Underlying (SPX cash index) --------------------------------------------------
UNDERLYING_SYMBOL: str = "SPX"
UNDERLYING_SECTYPE: str = "IND"
UNDERLYING_EXCHANGE: str = "CBOE"
UNDERLYING_CURRENCY: str = "USD"

# --- Option universe --------------------------------------------------------------
OPTION_EXCHANGE: str = "CBOE"
OPTION_TRADING_CLASS: str = "SPX"  # AM-settled monthlies (excludes SPXW weeklies)
OPTION_MULTIPLIER: str = "100"
OPTION_RIGHTS: tuple[str, ...] = ("C", "P")
MIN_MATURITY_MONTHS: int = 1
MAX_MATURITY_MONTHS: int = 12
# Nearest listed monthly to each bound can sit ~3 weeks off the exact month mark.
MATURITY_TOLERANCE_DAYS: float = 20.0

# --- Market-data snapshot pacing --------------------------------------------------
SNAPSHOT_BATCH_SIZE: int = 50  # contracts per reqTickers call (IBKR line limits)
SNAPSHOT_PACING_SECONDS: float = 1.0

# --- Rate / dividend implication (put-call parity) --------------------------------
PARITY_MONEYNESS_MIN: float = 0.95  # near-ATM strikes give the cleanest parity fit
PARITY_MONEYNESS_MAX: float = 1.05
PARITY_MIN_STRIKES: int = 3  # strikes carrying both a call and a put per maturity
MAX_DISCOUNT_FACTOR: float = 1.5  # reject degenerate parity fits (e^{-rT} sanity cap)

# --- Time / output ----------------------------------------------------------------
EASTERN_TZ: str = "America/New_York"
LIVE_CHAINS_DIR: Path = PROJECT_ROOT / "data" / "processed" / "live_chains"
CHAIN_DATE_FORMAT: str = "%Y-%m-%d"
IB_EXPIRY_FORMAT: str = "%Y%m%d"

# Base columns emitted before :func:`add_derived_columns` appends the rest; the
# resulting frame matches ``data/raw/spx_options.parquet`` exactly.
LIVE_CHAIN_BASE_COLUMNS: tuple[str, ...] = (
    "date",
    "exdate",
    "cp_flag",
    "strike",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "vega",
    "optionid",
    "spot",
)


def trading_date_now() -> pd.Timestamp:
    """Current trading date in US Eastern time (midnight-normalised).

    Returns:
        Today's date in ``America/New_York`` as a ``pd.Timestamp`` with no
        time component, matching the historical daily ``date`` convention.
    """
    now_eastern = dt.datetime.now(ZoneInfo(EASTERN_TZ))
    return pd.Timestamp(now_eastern.date())


# --- IBKR connectivity (thin wrappers over ib_insync) -----------------------------


def connect(
    host: str = IBKR_HOST,
    port: int = IBKR_PORT,
    client_id: int = DATA_CLIENT_ID,
    timeout: float = CONNECT_TIMEOUT_SECONDS,
) -> Any:
    """Open a read-only IBKR connection for market-data snapshots.

    Args:
        host: TWS/Gateway host.
        port: TWS/Gateway socket (paper trading is 7497).
        client_id: Client id unique to this process.
        timeout: Connection timeout in seconds.

    Returns:
        A connected ``ib_insync.IB`` instance (caller must ``disconnect``).
    """
    from ib_insync import IB  # lazy: offline tests never import ib_insync

    ib = IB()
    ib.connect(host=host, port=port, clientId=client_id, timeout=timeout, readonly=True)
    logger.info("connected to IBKR %s:%d (clientId=%d)", host, port, client_id)
    return ib


def qualify_underlying(ib: Any) -> Any:
    """Qualify the SPX index contract.

    Args:
        ib: Connected IBKR instance.

    Returns:
        The fully qualified SPX ``Index`` contract (carries its ``conId``).

    Raises:
        ValueError: If the contract cannot be qualified.
    """
    from ib_insync import Index

    qualified = ib.qualifyContracts(
        Index(UNDERLYING_SYMBOL, UNDERLYING_EXCHANGE, UNDERLYING_CURRENCY)
    )
    if not qualified:
        raise ValueError(f"could not qualify {UNDERLYING_SYMBOL} index contract")
    return qualified[0]


def underlying_spot(ib: Any, underlying: Any) -> float:
    """Snapshot the SPX index level.

    Args:
        ib: Connected IBKR instance.
        underlying: Qualified SPX index contract.

    Returns:
        The current index level, falling back to the prior close if the live
        market price is unavailable.

    Raises:
        ValueError: If neither a live price nor a close is available.
    """
    (ticker,) = ib.reqTickers(underlying)
    price = ticker.marketPrice()
    if price is None or not np.isfinite(price):
        price = ticker.close
    if price is None or not np.isfinite(price):
        raise ValueError("no usable SPX index price from IBKR")
    return float(price)


def select_option_chain(chains: Sequence[Any]) -> Any:
    """Pick the AM-settled monthly SPX chain from option-parameter results.

    Args:
        chains: Result of ``ib.reqSecDefOptParams`` (one entry per
            exchange/trading-class).

    Returns:
        The ``OptionChain`` for :data:`OPTION_TRADING_CLASS` on
        :data:`OPTION_EXCHANGE`.

    Raises:
        ValueError: If no matching chain is present.
    """
    for chain in chains:
        if (
            chain.tradingClass == OPTION_TRADING_CLASS
            and chain.exchange == OPTION_EXCHANGE
        ):
            return chain
    raise ValueError(
        f"no {OPTION_TRADING_CLASS} chain on {OPTION_EXCHANGE} in option parameters"
    )


def filter_expirations(
    expirations: Iterable[str], today: pd.Timestamp
) -> list[str]:
    """Keep expirations whose maturity falls in the 1-12 month band.

    Args:
        expirations: IBKR expiration strings (``YYYYMMDD``).
        today: Trading date the maturities are measured from.

    Returns:
        Sorted expiration strings whose days-to-expiry lie within
        ``[1 month - tol, 12 months + tol]``.
    """
    lower = MIN_MATURITY_MONTHS * DAYS_PER_MONTH - MATURITY_TOLERANCE_DAYS
    upper = MAX_MATURITY_MONTHS * DAYS_PER_MONTH + MATURITY_TOLERANCE_DAYS
    kept: list[str] = []
    for expiry in expirations:
        days = (pd.to_datetime(expiry, format=IB_EXPIRY_FORMAT) - today).days
        if lower <= days <= upper:
            kept.append(expiry)
    return sorted(kept)


def filter_strikes(strikes: Iterable[float], spot: float) -> list[float]:
    """Keep strikes inside the 80-120% moneyness band.

    Args:
        strikes: Listed strikes for the chain.
        spot: Current index level.

    Returns:
        Sorted strikes with ``MONEYNESS_MIN * spot <= K <= MONEYNESS_MAX * spot``.
    """
    low, high = MONEYNESS_MIN * spot, MONEYNESS_MAX * spot
    return sorted(float(k) for k in strikes if low <= k <= high)


def build_option_contracts(
    expirations: Sequence[str], strikes: Sequence[float]
) -> list[Any]:
    """Build unqualified SPX option contracts for every strike/expiry/right.

    Args:
        expirations: Selected expiration strings.
        strikes: Selected strikes.

    Returns:
        One ``Option`` per (expiry, strike, right) combination.
    """
    from ib_insync import Option

    contracts: list[Any] = []
    for expiry in expirations:
        for strike in strikes:
            for right in OPTION_RIGHTS:
                contracts.append(
                    Option(
                        symbol=UNDERLYING_SYMBOL,
                        lastTradeDateOrContractMonth=expiry,
                        strike=strike,
                        right=right,
                        exchange=OPTION_EXCHANGE,
                        multiplier=OPTION_MULTIPLIER,
                        currency=UNDERLYING_CURRENCY,
                        tradingClass=OPTION_TRADING_CLASS,
                    )
                )
    return contracts


def select_contracts(
    ib: Any, underlying: Any, spot: float, today: pd.Timestamp
) -> list[Any]:
    """Resolve the day's tradable SPX option universe from IBKR.

    Args:
        ib: Connected IBKR instance.
        underlying: Qualified SPX index contract.
        spot: Current index level (for the moneyness band).
        today: Trading date (for the maturity band).

    Returns:
        Qualified ``Option`` contracts inside the moneyness/maturity bands.
    """
    params = ib.reqSecDefOptParams(
        UNDERLYING_SYMBOL, "", UNDERLYING_SECTYPE, underlying.conId
    )
    chain = select_option_chain(params)
    expirations = filter_expirations(chain.expirations, today)
    strikes = filter_strikes(chain.strikes, spot)
    qualified = ib.qualifyContracts(*build_option_contracts(expirations, strikes))
    logger.info(
        "selected %d qualified contracts (%d expiries x %d strikes x %d rights)",
        len(qualified),
        len(expirations),
        len(strikes),
        len(OPTION_RIGHTS),
    )
    return qualified


def fetch_quotes(ib: Any, contracts: Sequence[Any]) -> list[Any]:
    """Snapshot bid/ask and option greeks for every contract, in batches.

    Args:
        ib: Connected IBKR instance.
        contracts: Qualified option contracts.

    Returns:
        One ``Ticker`` per contract, carrying quotes and ``modelGreeks``.
    """
    tickers: list[Any] = []
    for start in range(0, len(contracts), SNAPSHOT_BATCH_SIZE):
        batch = contracts[start : start + SNAPSHOT_BATCH_SIZE]
        tickers.extend(ib.reqTickers(*batch))
        ib.sleep(SNAPSHOT_PACING_SECONDS)
    logger.info("fetched %d option tickers", len(tickers))
    return tickers


# --- Pure frame construction (no ib_insync, fully testable offline) ----------------


def _usable_price(value: Any) -> float:
    """Coerce an IBKR price/size to a float, mapping IBKR's -1/None to NaN.

    Args:
        value: Raw ``bid``/``ask``/``volume`` from a ticker.

    Returns:
        The value as a non-negative float, or NaN if missing/sentinel.
    """
    if value is None:
        return float("nan")
    number = float(value)
    return number if np.isfinite(number) and number >= 0.0 else float("nan")


def _greek(value: Any) -> float:
    """Coerce an option-computation greek to a float (NaN when absent)."""
    return float("nan") if value is None else float(value)


def tickers_to_frame(
    tickers: Sequence[Any], spot: float, today: pd.Timestamp
) -> pd.DataFrame:
    """Convert IBKR tickers into the historical option-chain schema.

    Keeps only contracts with a model implied vol and a usable two-sided
    quote; market IV, delta and vega are taken from ``modelGreeks``. The
    result carries the same columns as ``data/raw/spx_options.parquet``.

    Args:
        tickers: Snapshot tickers from :func:`fetch_quotes`.
        spot: Index level to stamp on every row.
        today: Trading date to stamp on every row.

    Returns:
        Chain frame with :data:`LIVE_CHAIN_BASE_COLUMNS` plus the columns
        appended by :func:`src.heston.data.add_derived_columns`.
    """
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        contract = ticker.contract
        greeks = ticker.modelGreeks
        if greeks is None:
            continue
        iv = greeks.impliedVol
        if iv is None or not np.isfinite(iv) or iv <= 0.0:
            continue
        bid, ask = _usable_price(ticker.bid), _usable_price(ticker.ask)
        if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0.0:
            continue
        rows.append(
            {
                "date": today,
                "exdate": pd.to_datetime(
                    contract.lastTradeDateOrContractMonth, format=IB_EXPIRY_FORMAT
                ),
                "cp_flag": str(contract.right)[0].upper(),
                "strike": float(contract.strike),
                "best_bid": bid,
                "best_offer": ask,
                "volume": _usable_price(ticker.volume),
                "open_interest": float("nan"),
                "impl_volatility": float(iv),
                "delta": _greek(greeks.delta),
                "vega": _greek(greeks.vega),
                "optionid": int(contract.conId),
                "spot": float(spot),
            }
        )
    frame = pd.DataFrame(rows, columns=list(LIVE_CHAIN_BASE_COLUMNS))
    return add_derived_columns(frame)


def imply_rates_and_dividends(
    chain: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back out the zero curve and dividend yield from the chain by parity.

    For each (date, maturity) the near-ATM call-minus-put differences are
    linear in the strike: ``C - P = S e^{-qT} - K e^{-rT}``. A least-squares
    line gives ``e^{-rT}`` (negated slope) and ``S e^{-qT}`` (intercept), so
    ``r = -ln(e^{-rT}) / T`` and ``q = -ln(S e^{-qT} / S) / T``. Maturities
    with fewer than :data:`PARITY_MIN_STRIKES` paired strikes, or a
    degenerate fit, are skipped.

    Args:
        chain: A live chain (or several days concatenated) with ``mid``,
            ``moneyness``, ``days_to_expiry``, ``cp_flag``, ``strike``,
            ``spot`` and ``date`` columns.

    Returns:
        Tuple ``(zero_curve, div_yield)`` in the schemas consumed by
        :func:`src.heston.calibration.build_calibration_sample`: the zero
        curve has ``date``/``days``/``rate`` (rate in percent, one row per
        (date, maturity)); the dividend frame has ``date``/``rate`` (percent,
        one row per date, median across maturities).
    """
    zero_rows: list[dict[str, Any]] = []
    div_rows: list[dict[str, Any]] = []
    for date, day in chain.groupby("date"):
        spot = float(day["spot"].iloc[0])
        near_atm = day[
            (day["moneyness"] >= PARITY_MONEYNESS_MIN)
            & (day["moneyness"] <= PARITY_MONEYNESS_MAX)
        ]
        q_values: list[float] = []
        for days_to_expiry, maturity_group in near_atm.groupby("days_to_expiry"):
            maturity_years = float(days_to_expiry) / DAYS_PER_YEAR
            if maturity_years <= 0.0:
                continue
            calls = maturity_group.loc[
                maturity_group["cp_flag"] == "C"
            ].set_index("strike")["mid"]
            puts = maturity_group.loc[
                maturity_group["cp_flag"] == "P"
            ].set_index("strike")["mid"]
            common = calls.index.intersection(puts.index)
            if len(common) < PARITY_MIN_STRIKES:
                continue
            strikes = common.to_numpy(dtype=float)
            diff = (
                calls.loc[common].to_numpy(dtype=float)
                - puts.loc[common].to_numpy(dtype=float)
            )
            slope, intercept = np.polyfit(strikes, diff, 1)
            discount = -float(slope)  # e^{-rT}
            forward_pv = float(intercept)  # S e^{-qT}
            if not 0.0 < discount <= MAX_DISCOUNT_FACTOR or forward_pv <= 0.0:
                continue
            rate = -np.log(discount) / maturity_years
            div = -np.log(forward_pv / spot) / maturity_years
            zero_rows.append(
                {"date": date, "days": float(days_to_expiry), "rate": rate * PERCENT}
            )
            q_values.append(div)
        if q_values:
            div_rows.append(
                {"date": date, "rate": float(np.median(q_values)) * PERCENT}
            )
    zero_curve = pd.DataFrame(zero_rows, columns=["date", "days", "rate"])
    div_yield = pd.DataFrame(div_rows, columns=["date", "rate"])
    return zero_curve, div_yield


# --- Storage ----------------------------------------------------------------------


def store_chain(
    chain: pd.DataFrame, today: pd.Timestamp, directory: Path = LIVE_CHAINS_DIR
) -> Path:
    """Write one day's validated chain to a date-stamped parquet file.

    Args:
        chain: Validated live chain.
        today: Trading date (names the file).
        directory: Destination directory.

    Returns:
        Path to the written parquet file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{today.strftime(CHAIN_DATE_FORMAT)}.parquet"
    chain.to_parquet(path, index=False)
    logger.info("wrote %d quotes to %s", len(chain), path)
    return path


def load_stored_chains(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    directory: Path = LIVE_CHAINS_DIR,
) -> pd.DataFrame:
    """Load and concatenate stored daily chains within a date window.

    Args:
        start_date: Inclusive first trading date to load.
        end_date: Inclusive last trading date to load.
        directory: Directory of date-stamped chain parquet files.

    Returns:
        The concatenated chain sorted by date, or an empty frame if no
        stored file falls in the window.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.parquet")):
        file_date = pd.Timestamp(path.stem)
        if start_date <= file_date <= end_date:
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(
        drop=True
    )


# --- Orchestration ----------------------------------------------------------------


def run(today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, Path]:
    """Pull, clean, validate and store today's SPX option chain.

    Args:
        today: Trading date; defaults to the current US Eastern date.

    Returns:
        Tuple ``(chain, path)`` of the stored chain and its file path.
    """
    if today is None:
        today = trading_date_now()
    ib = connect()
    try:
        underlying = qualify_underlying(ib)
        spot = underlying_spot(ib, underlying)
        contracts = select_contracts(ib, underlying, spot, today)
        tickers = fetch_quotes(ib, contracts)
    finally:
        ib.disconnect()

    chain = drop_crossed_quotes(tickers_to_frame(tickers, spot, today))
    validate_chain(chain)
    path = store_chain(chain, today)
    return chain, path


def main() -> None:
    """CLI entry point: ``python -m src.paper_trading.data_live``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    chain, path = run()
    print(f"wrote {len(chain):,} quotes to {path}")


if __name__ == "__main__":
    main()
