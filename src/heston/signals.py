"""Mispriced-option signal generation from model-vs-market IV deviations.

Builds a daily panel of every OTM SPX quote with its Heston model IV (priced
with the parameters calibrated on that quarter's trailing window — no
lookahead) and flags:

    - entry_short: market IV > model IV + 2 vol pts (vol rich -> sell option)
    - entry_long:  market IV < model IV - 2 vol pts — PERMANENTLY DISABLED
      (LONG_VOL_ENABLED = False): the long-vol leg lost -$110.8k at a 15%
      win rate over 2011-2025, fighting vol clustering and the implied-vol
      risk premium. The production strategy is short-vol only.
    - exit_signal: |market IV - model IV| < 0.5 vol pts (deviation reverted)
    - liquid:      bid-ask spread below 0.5 vol pts (entries require this)

Run:
    python -m src.heston.signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.heston.calibration import (
    PARAMS_PARQUET,
    PERCENT,
    build_calibration_sample,
    model_iv_for_sample,
)
from src.heston.data import (
    DIV_YIELD_PARQUET,
    OPTIONS_PARQUET,
    PROJECT_ROOT,
    ZERO_CURVE_PARQUET,
)
from src.heston.surface import params_for_date

# --- Signal thresholds (vol units) ------------------------------------------------
ENTRY_THRESHOLD: float = 0.02  # deviation beyond 2 vol pts opens a position
EXIT_THRESHOLD: float = 0.005  # deviation within 0.5 vol pts closes it
MAX_SPREAD_IV: float = 0.005  # liquidity filter: spread < 0.5 vol pts
LONG_VOL_ENABLED: bool = False  # long-vol leg permanently disabled (see module docstring)

# --- Panel construction -----------------------------------------------------------
PANEL_EXTRA_COLUMNS: tuple[str, ...] = (
    "optionid",
    "cp_flag",
    "exdate",
    "best_bid",
    "best_offer",
    "mid",
    "days_to_expiry",
)

# --- Output -----------------------------------------------------------------------
PANEL_PARQUET = PROJECT_ROOT / "data" / "processed" / "signal_panel.parquet"


def build_signal_panel(
    chain: pd.DataFrame,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    params_table: pd.DataFrame,
) -> pd.DataFrame:
    """Daily OTM quote panel with model IVs and deviations, quarter by quarter.

    Every quarter's quotes are priced with :func:`params_for_date` at the
    quarter start, i.e. parameters calibrated strictly on data before the
    quarter. Dates before the first covered quarter are excluded.

    Args:
        chain: Full option chain.
        zero_curve: Zero curve frame.
        div_yield: Dividend-yield frame.
        params_table: Rolling calibration output.

    Returns:
        Panel frame with quote details, model_iv and deviation
        (market_iv - model_iv), sorted by date.
    """
    quarters = pd.DatetimeIndex(
        pd.to_datetime(params_table["quarter_start"]).sort_values().unique()
    )
    frames: list[pd.DataFrame] = []
    for q_start in quarters:
        q_end = q_start + pd.offsets.QuarterBegin(startingMonth=1)
        params = params_for_date(params_table, q_start)
        if params is None:
            print(f"{q_start:%Y-%m-%d}: no usable parameters, skipping quarter")
            continue
        sample = build_calibration_sample(
            chain,
            zero_curve,
            div_yield,
            q_start,
            q_end,
            month_ends_only=False,
            nearest_tenors_only=False,
            extra_columns=PANEL_EXTRA_COLUMNS,
        )
        if sample.empty:
            print(f"{q_start:%Y-%m-%d}: no quotes, skipping quarter")
            continue
        sample["model_iv"] = model_iv_for_sample(params, sample)
        sample["deviation"] = sample["market_iv"] - sample["model_iv"]
        frames.append(sample)
        print(
            f"{q_start:%Y-%m-%d}: {len(sample):,} quotes, "
            f"median |dev| {sample['deviation'].abs().median() * PERCENT:.2f} vol pts",
            flush=True,
        )
    panel = pd.concat(frames, ignore_index=True).sort_values("date")
    return panel.reset_index(drop=True)


def add_signal_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach liquidity and entry/exit signal flags to a deviation panel.

    Args:
        panel: Frame with ``deviation`` and ``spread_iv`` columns.

    Returns:
        Copy with boolean columns liquid, entry_short, entry_long,
        exit_signal. Quotes with a failed model IV get no flags.
        ``entry_long`` is always False while LONG_VOL_ENABLED is False; the
        column is kept so the backtest schema is unchanged.
    """
    out = panel.copy()
    dev = out["deviation"].to_numpy(float)
    valid = ~np.isnan(dev)
    liquid = (out["spread_iv"].to_numpy(float) < MAX_SPREAD_IV) & valid
    out["liquid"] = liquid
    out["entry_short"] = liquid & (dev > ENTRY_THRESHOLD)
    out["entry_long"] = liquid & (dev < -ENTRY_THRESHOLD) & LONG_VOL_ENABLED
    out["exit_signal"] = valid & (np.abs(dev) < EXIT_THRESHOLD)
    return out


def entry_events(panel: pd.DataFrame) -> pd.DataFrame:
    """All entry signals as one event table.

    Args:
        panel: Flagged panel from :func:`add_signal_flags`.

    Returns:
        Frame of entry rows with a ``direction`` column (+1 long option /
        long vol, -1 short option / short vol), sorted by date then by
        absolute deviation descending (strongest signals first).
    """
    entries = panel.loc[panel["entry_long"] | panel["entry_short"]].copy()
    entries["direction"] = np.where(entries["entry_long"], 1, -1)
    entries["abs_deviation"] = entries["deviation"].abs()
    entries = entries.sort_values(
        ["date", "abs_deviation"], ascending=[True, False]
    )
    return entries.drop(columns=["abs_deviation"]).reset_index(drop=True)


def summarize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-year signal summary used for reporting.

    Args:
        panel: Flagged panel.

    Returns:
        Frame indexed by year with quote/liquidity/entry counts.
    """
    year = panel["date"].dt.year
    return pd.DataFrame(
        {
            "quotes": panel.groupby(year).size(),
            "liquid_share": panel.groupby(year)["liquid"].mean().round(3),
            "entries_short": panel.groupby(year)["entry_short"].sum(),
            "entries_long": panel.groupby(year)["entry_long"].sum(),
        }
    )


def main() -> None:
    """CLI entry point: ``python -m src.heston.signals``."""
    chain = pd.read_parquet(OPTIONS_PARQUET)
    zero_curve = pd.read_parquet(ZERO_CURVE_PARQUET)
    div_yield = pd.read_parquet(DIV_YIELD_PARQUET)
    params_table = pd.read_parquet(PARAMS_PARQUET)

    panel = add_signal_flags(
        build_signal_panel(chain, zero_curve, div_yield, params_table)
    )
    PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PARQUET, index=False)

    print(f"\nwrote {len(panel):,} panel rows to {PANEL_PARQUET}")
    print(summarize_panel(panel).to_string())
    n_entries = int((panel["entry_long"] | panel["entry_short"]).sum())
    print(
        f"\ntotal entry signals: {n_entries:,} "
        f"(long {int(panel['entry_long'].sum()):,} / "
        f"short {int(panel['entry_short'].sum()):,})"
    )


if __name__ == "__main__":
    main()
