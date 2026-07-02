"""SPX option chain ingestion from WRDS OptionMetrics.

Pulls daily SPX (secid 108105) option chains from ``optionm_all.opprcd{yyyy}``,
restricted to standard AM-settled monthly contracts inside the target
moneyness band, together with the reference data needed for Carr-Madan
pricing downstream: forward prices, the zero-coupon curve, implied dividend
yields and the index level.

Notes on OptionMetrics conventions verified against the live database:
    - ``strike_price`` is stored multiplied by 1000.
    - Weekly contracts carry ``expiry_indicator = 'w'``; standard monthly SPX
      contracts are AM-settled (``am_settlement = 1``) with a null indicator.
    - ``zerocd.rate`` and ``idxdvd.rate`` are annualised percentages.

Run:
    python -m src.heston.data [--start-year YYYY] [--end-year YYYY]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- WRDS source ------------------------------------------------------------
SPX_SECID: int = 108105
WRDS_LIBRARY: str = "optionm_all"
WRDS_USERNAME: str = "purujitaggarwal"

# --- Sample definition --------------------------------------------------------
START_YEAR: int = 2010
END_YEAR: int = 2025
MONEYNESS_MIN: float = 0.80
MONEYNESS_MAX: float = 1.20
TARGET_TENOR_MONTHS: tuple[int, ...] = (1, 2, 3, 6, 9, 12)
DAYS_PER_MONTH: float = 30.4375
MIN_DAYS_TO_EXPIRY: int = 7
MAX_DAYS_TO_EXPIRY: int = 400
# SPX monthlies run serial for ~4 months then quarterly, so the listed expiry
# nearest a 9M target can sit ~1.5 months away.
TENOR_TOLERANCE_DAYS: float = 46.0
STRIKE_SCALE: float = 1000.0

# --- Output locations ---------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
OPTIONS_PARQUET: Path = RAW_DIR / "spx_options.parquet"
FORWARDS_PARQUET: Path = RAW_DIR / "spx_forwards.parquet"
ZERO_CURVE_PARQUET: Path = RAW_DIR / "zero_curve.parquet"
DIV_YIELD_PARQUET: Path = RAW_DIR / "spx_div_yield.parquet"
SPOT_PARQUET: Path = RAW_DIR / "spx_spot.parquet"

CHAIN_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "exdate",
    "cp_flag",
    "strike",
    "best_bid",
    "best_offer",
    "impl_volatility",
    "volume",
    "open_interest",
    "spot",
)


def connect() -> Any:
    """Open a WRDS connection using the pgpass credentials.

    Returns:
        An authenticated ``wrds.Connection``.
    """
    import wrds  # imported lazily so offline tests never require it

    return wrds.Connection(wrds_username=WRDS_USERNAME)


def build_option_query(year: int) -> str:
    """Build the SQL pulling one year of filtered SPX monthly option quotes.

    Filters applied server-side: SPX secid, standard AM-settled monthly
    contracts only (no weeklies/EOM), days-to-expiry inside
    [MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY], and strike inside the
    moneyness band around that day's index close.

    Args:
        year: Calendar year of the ``opprcd{year}`` table.

    Returns:
        SQL string ready for ``wrds.Connection.raw_sql``.
    """
    return f"""
        select o.date, o.exdate, o.cp_flag,
               o.strike_price / {STRIKE_SCALE} as strike,
               o.best_bid, o.best_offer, o.volume, o.open_interest,
               o.impl_volatility, o.delta, o.vega, o.optionid,
               s.close as spot
        from {WRDS_LIBRARY}.opprcd{year} o
        join {WRDS_LIBRARY}.secprd{year} s
          on s.secid = o.secid and s.date = o.date
        where o.secid = {SPX_SECID}
          and o.expiry_indicator is null
          and o.am_settlement = 1
          and (o.exdate - o.date) between {MIN_DAYS_TO_EXPIRY} and {MAX_DAYS_TO_EXPIRY}
          and o.strike_price >= {MONEYNESS_MIN} * s.close * {STRIKE_SCALE}
          and o.strike_price <= {MONEYNESS_MAX} * s.close * {STRIKE_SCALE}
    """


def fetch_option_year(db: Any, year: int) -> pd.DataFrame:
    """Fetch one year of filtered SPX option quotes.

    Args:
        db: Open WRDS connection.
        year: Calendar year to pull.

    Returns:
        Raw quote DataFrame with derived columns added.
    """
    frame = db.raw_sql(build_option_query(year), date_cols=["date", "exdate"])
    return add_derived_columns(frame)


def fetch_forwards_year(db: Any, year: int) -> pd.DataFrame:
    """Fetch SPX forward prices for every expiration quoted in a year."""
    query = f"""
        select date, expiration, amsettlement, forwardprice
        from {WRDS_LIBRARY}.fwdprd{year}
        where secid = {SPX_SECID}
    """
    return db.raw_sql(query, date_cols=["date", "expiration"])


def fetch_spot_year(db: Any, year: int) -> pd.DataFrame:
    """Fetch the daily SPX index close for a year."""
    query = f"""
        select date, close
        from {WRDS_LIBRARY}.secprd{year}
        where secid = {SPX_SECID}
    """
    return db.raw_sql(query, date_cols=["date"])


def fetch_zero_curve(db: Any, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch the zero-coupon yield curve (annualised percent) for the sample."""
    query = f"""
        select date, days, rate
        from {WRDS_LIBRARY}.zerocd
        where date between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(query, date_cols=["date"])


def fetch_div_yield(db: Any, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch the SPX implied dividend yield (annualised percent)."""
    query = f"""
        select date, rate
        from {WRDS_LIBRARY}.idxdvd
        where secid = {SPX_SECID}
          and date between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(query, date_cols=["date"])


def add_derived_columns(chain: pd.DataFrame) -> pd.DataFrame:
    """Add mid price, spreads, days-to-expiry and moneyness to a quote frame.

    Args:
        chain: Quote frame with at least date, exdate, best_bid, best_offer,
            strike and spot columns.

    Returns:
        Copy of ``chain`` with ``mid``, ``spread``, ``rel_spread``,
        ``days_to_expiry`` and ``moneyness`` columns appended.
    """
    out = chain.copy()
    out["mid"] = (out["best_bid"] + out["best_offer"]) / 2.0
    out["spread"] = out["best_offer"] - out["best_bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_spread"] = np.where(
            out["mid"] > 0.0, out["spread"] / out["mid"], np.nan
        )
    out["days_to_expiry"] = (out["exdate"] - out["date"]).dt.days
    out["moneyness"] = out["strike"] / out["spot"]
    return out


def target_tenor_days(months: Sequence[int] = TARGET_TENOR_MONTHS) -> np.ndarray:
    """Convert target tenors in months to calendar days.

    Args:
        months: Target tenors, e.g. ``(1, 2, 3, 6, 9, 12)``.

    Returns:
        Array of target maturities in days.
    """
    return np.asarray([m * DAYS_PER_MONTH for m in months])


def select_nearest_expiries(
    days_to_expiry: Sequence[int],
    targets_days: Sequence[float] | None = None,
    tolerance_days: float = TENOR_TOLERANCE_DAYS,
) -> dict[float, int]:
    """Map each target tenor to the nearest listed days-to-expiry.

    Args:
        days_to_expiry: Distinct listed days-to-expiry available on one date.
        targets_days: Target maturities in days; defaults to the standard
            1/2/3/6/9/12-month grid.
        tolerance_days: Targets with no listed expiry within this distance
            are dropped.

    Returns:
        Mapping of target-days -> selected listed days-to-expiry. A listed
        expiry may serve at most one target (nearest target wins).
    """
    if targets_days is None:
        targets_days = target_tenor_days()
    listed = np.unique(np.asarray(list(days_to_expiry), dtype=float))
    selection: dict[float, int] = {}
    if listed.size == 0:
        return selection
    for target in targets_days:
        gaps = np.abs(listed - target)
        best = int(np.argmin(gaps))
        if gaps[best] > tolerance_days:
            continue
        candidate = int(listed[best])
        if candidate in selection.values():
            continue
        selection[float(target)] = candidate
    return selection


def zero_rate(curve: pd.DataFrame, days: float) -> float:
    """Interpolate an annualised zero rate (percent) for one maturity.

    Linear interpolation in days with flat extrapolation beyond the ends.

    Args:
        curve: Single-date slice of the zero curve with ``days`` and ``rate``.
        days: Maturity in calendar days.

    Returns:
        Interpolated annualised rate in percent.

    Raises:
        ValueError: If ``curve`` is empty.
    """
    if curve.empty:
        raise ValueError("zero curve slice is empty")
    nodes = curve.sort_values("days")
    return float(
        np.interp(days, nodes["days"].to_numpy(float), nodes["rate"].to_numpy(float))
    )


def drop_crossed_quotes(chain: pd.DataFrame) -> pd.DataFrame:
    """Remove quotes with bid above offer (rare exchange data artifacts).

    Args:
        chain: Quote frame with ``best_bid`` and ``best_offer``.

    Returns:
        Copy of ``chain`` without crossed quotes, index reset.
    """
    crossed = chain["best_bid"] > chain["best_offer"]
    if crossed.any():
        print(f"dropping {int(crossed.sum())} crossed quotes")
    return chain.loc[~crossed].reset_index(drop=True)


def validate_chain(chain: pd.DataFrame) -> None:
    """Validate the ingested option chain before it is written to parquet.

    Args:
        chain: Full ingested quote frame.

    Raises:
        ValueError: If required columns are missing, strikes/spots are not
            strictly positive, or a quote has bid above offer.
    """
    missing = [c for c in CHAIN_REQUIRED_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")
    if (chain["strike"] <= 0.0).any():
        raise ValueError("chain contains non-positive strikes")
    if (chain["spot"] <= 0.0).any():
        raise ValueError("chain contains non-positive spot values")
    crossed = chain["best_bid"] > chain["best_offer"]
    if crossed.any():
        raise ValueError(f"chain contains {int(crossed.sum())} crossed quotes")


def ingest(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """Run the full ingestion and write all raw parquet files.

    Args:
        start_year: First calendar year to pull.
        end_year: Last calendar year to pull (inclusive).

    Returns:
        The full option chain that was written to ``OPTIONS_PARQUET``.
    """
    db = connect()
    years = range(start_year, end_year + 1)

    chains: list[pd.DataFrame] = []
    forwards: list[pd.DataFrame] = []
    spots: list[pd.DataFrame] = []
    for year in years:
        chain_year = fetch_option_year(db, year)
        chains.append(chain_year)
        forwards.append(fetch_forwards_year(db, year))
        spots.append(fetch_spot_year(db, year))
        print(f"{year}: {len(chain_year):,} option quotes")

    chain = drop_crossed_quotes(pd.concat(chains, ignore_index=True))
    validate_chain(chain)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(OPTIONS_PARQUET, index=False)
    pd.concat(forwards, ignore_index=True).to_parquet(FORWARDS_PARQUET, index=False)
    pd.concat(spots, ignore_index=True).to_parquet(SPOT_PARQUET, index=False)
    fetch_zero_curve(db, start_year, end_year).to_parquet(
        ZERO_CURVE_PARQUET, index=False
    )
    fetch_div_yield(db, start_year, end_year).to_parquet(
        DIV_YIELD_PARQUET, index=False
    )

    print(
        f"wrote {len(chain):,} quotes "
        f"({chain['date'].min():%Y-%m-%d} to {chain['date'].max():%Y-%m-%d}) "
        f"to {OPTIONS_PARQUET}"
    )
    return chain


def main() -> None:
    """CLI entry point: ``python -m src.heston.data``."""
    parser = argparse.ArgumentParser(description="Ingest SPX chains from WRDS.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    ingest(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
