"""Rolling Levenberg-Marquardt calibration of Heston parameters to SPX IVs.

For every quarter start in the sample, calibrates (kappa, theta, sigma, rho,
v0) to the pooled month-end OTM implied-vol surfaces of the trailing one-year
window, strictly excluding the quarter start date itself (no lookahead).

Objective: weighted least squares on implied-vol residuals
    r_i = w_i * (model_iv_i - market_iv_i)
with weights inversely proportional to each quote's bid-ask spread expressed
in vol points (spread / vega), so tighter markets get more weight.

Optimiser: scipy.optimize.least_squares(method="lm") — Levenberg-Marquardt —
run in a transformed space (log for the positive parameters, tanh for rho) so
the unbounded LM iterates always map to valid Heston parameters. Multiple
random restarts guard against local optima.

Run:
    python -m src.heston.calibration
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.heston.data import (
    DIV_YIELD_PARQUET,
    OPTIONS_PARQUET,
    PROJECT_ROOT,
    ZERO_CURVE_PARQUET,
    select_nearest_expiries,
    zero_rate,
)
from src.heston.fft import (
    HestonParams,
    bs_vega_array,
    heston_call_prices,
    implied_vol_newton,
)

# --- Rolling schedule -----------------------------------------------------------
FIRST_QUARTER_START: str = "2011-01-01"  # first quarter with a full 1y window
LAST_QUARTER_START: str = "2025-07-01"
TRAILING_WINDOW_DAYS: int = 365
DAYS_PER_YEAR: float = 365.0

# --- Sample construction ----------------------------------------------------------
PERCENT: float = 100.0  # zerocd / idxdvd rates are stored in percent
MIN_BID: float = 0.0  # quotes must have a strictly positive bid
SPREAD_IV_FLOOR: float = 1e-3  # floor on spread in vol points (avoids inf weights)

# --- Optimiser -------------------------------------------------------------------
N_RESTARTS: int = 5
RANDOM_SEED: int = 42
MAX_NFEV: int = 400
NAN_RESIDUAL: float = 0.5  # penalty (in weighted vol units) for failed model IVs
# Random-restart sampling ranges (uniform), chosen to span realistic SPX regimes.
KAPPA_RANGE: tuple[float, float] = (0.5, 5.0)
THETA_RANGE: tuple[float, float] = (0.01, 0.09)
SIGMA_RANGE: tuple[float, float] = (0.2, 1.0)
RHO_RANGE: tuple[float, float] = (-0.9, -0.3)
V0_RANGE: tuple[float, float] = (0.01, 0.09)

# --- Output ----------------------------------------------------------------------
PARAMS_PARQUET = PROJECT_ROOT / "data" / "processed" / "heston_params.parquet"

SAMPLE_COLUMNS: tuple[str, ...] = (
    "date",
    "maturity_years",
    "strike",
    "spot",
    "rate",
    "div_yield",
    "market_iv",
    "spread_iv",
    "weight",
)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of calibrating one trailing window.

    Attributes:
        params: Best-fit Heston parameters (may be meaningless if not
            converged).
        rmse: Unweighted RMSE of model vs market IV at the optimum.
        weighted_rmse: Weighted RMSE at the optimum (the objective).
        converged: True if at least one restart terminated successfully.
        n_restarts_converged: Number of restarts reporting success.
        n_quotes: Number of quotes in the calibration sample.
        n_dates: Number of distinct surface dates pooled.
    """

    params: HestonParams
    rmse: float
    weighted_rmse: float
    converged: bool
    n_restarts_converged: int
    n_quotes: int
    n_dates: int


def quarter_starts(first: str = FIRST_QUARTER_START, last: str = LAST_QUARTER_START) -> pd.DatetimeIndex:
    """Quarterly calibration dates from ``first`` to ``last`` inclusive.

    Args:
        first: First quarter-start date (must fall on a quarter boundary).
        last: Last quarter-start date.

    Returns:
        DatetimeIndex of quarter starts.
    """
    return pd.date_range(start=first, end=last, freq="QS")


def month_end_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """Last available trading date of each calendar month.

    Args:
        dates: Series of trading dates (repeats allowed).

    Returns:
        Sorted unique month-end trading dates.
    """
    unique = pd.Series(pd.DatetimeIndex(dates.unique()).sort_values())
    return pd.DatetimeIndex(unique.groupby(unique.dt.to_period("M")).max().values)


def transform_to_params(x: np.ndarray) -> HestonParams:
    """Map unbounded optimiser variables to valid Heston parameters.

    kappa, theta, sigma, v0 use exp; rho uses tanh.

    Args:
        x: Unbounded vector of length 5.

    Returns:
        Valid Heston parameters.
    """
    return HestonParams(
        kappa=float(np.exp(x[0])),
        theta=float(np.exp(x[1])),
        sigma=float(np.exp(x[2])),
        rho=float(np.tanh(x[3])),
        v0=float(np.exp(x[4])),
    )


def transform_from_params(params: HestonParams) -> np.ndarray:
    """Inverse of :func:`transform_to_params`.

    Args:
        params: Valid Heston parameters.

    Returns:
        Unbounded vector of length 5.
    """
    return np.array(
        [
            np.log(params.kappa),
            np.log(params.theta),
            np.log(params.sigma),
            np.arctanh(params.rho),
            np.log(params.v0),
        ]
    )


def build_calibration_sample(
    chain: pd.DataFrame,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    month_ends_only: bool = True,
    nearest_tenors_only: bool = True,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Assemble the pooled OTM quote sample for one date window.

    Steps: restrict to [window_start, window_end); optionally keep only the
    last trading date of each month; keep OTM quotes (calls above spot, puts
    below) with a positive bid and an OptionMetrics implied vol; optionally
    keep only the listed expiry nearest each target tenor; attach
    interpolated zero rates, dividend yields, vol-point spreads and
    inverse-spread weights.

    Args:
        chain: Full option chain from ``data.ingest``.
        zero_curve: Zero curve frame (date, days, rate-in-percent).
        div_yield: Dividend-yield frame (date, rate-in-percent).
        window_start: Inclusive window start.
        window_end: Exclusive window end — never include this date or later.
        month_ends_only: If True (calibration), pool only month-end surfaces;
            if False (evaluation), keep every trading date in the window.
        nearest_tenors_only: If True, keep only the expiries nearest the
            target tenor grid; if False (signal panel), keep every expiry.
        extra_columns: Additional chain columns carried through to the
            output (e.g. optionid, cp_flag for position tracking).

    Returns:
        Sample frame with SAMPLE_COLUMNS plus ``extra_columns``.
    """
    out_columns = list(SAMPLE_COLUMNS) + list(extra_columns)
    window = chain.loc[
        (chain["date"] >= window_start) & (chain["date"] < window_end)
    ].copy()
    if window.empty:
        return pd.DataFrame(columns=out_columns)
    if month_ends_only:
        window = window.loc[window["date"].isin(month_end_dates(window["date"]))]

    otm = ((window["cp_flag"] == "C") & (window["moneyness"] >= 1.0)) | (
        (window["cp_flag"] == "P") & (window["moneyness"] < 1.0)
    )
    window = window.loc[
        otm & (window["best_bid"] > MIN_BID) & window["impl_volatility"].notna()
    ]
    if window.empty:
        return pd.DataFrame(columns=out_columns)

    if nearest_tenors_only:
        kept: list[pd.DataFrame] = []
        for date, day in window.groupby("date"):
            selected = select_nearest_expiries(day["days_to_expiry"].tolist())
            day = day.loc[day["days_to_expiry"].isin(selected.values())]
            if not day.empty:
                kept.append(day)
        if not kept:
            return pd.DataFrame(columns=out_columns)
        sample = pd.concat(kept, ignore_index=True)
    else:
        sample = window.reset_index(drop=True)

    sample["maturity_years"] = sample["days_to_expiry"] / DAYS_PER_YEAR

    curve_dates = np.sort(zero_curve["date"].unique())
    rates = np.empty(len(sample))
    for (date, dte), idx in sample.groupby(["date", "days_to_expiry"]).groups.items():
        pos = np.searchsorted(curve_dates, np.datetime64(date), side="right") - 1
        curve_slice = zero_curve.loc[zero_curve["date"] == curve_dates[pos]]
        rates[np.asarray(idx)] = zero_rate(curve_slice, float(dte)) / PERCENT
    sample["rate"] = rates

    dividends = div_yield.sort_values("date")[["date", "rate"]].rename(
        columns={"rate": "div_pct"}
    )
    sample = pd.merge_asof(
        sample.sort_values("date"), dividends, on="date", direction="backward"
    ).reset_index(drop=True)
    sample["div_yield"] = sample["div_pct"] / PERCENT

    sample["market_iv"] = sample["impl_volatility"].astype(float)
    vega = bs_vega_array(
        sample["spot"].to_numpy(float),
        sample["strike"].to_numpy(float),
        sample["maturity_years"].to_numpy(float),
        sample["rate"].to_numpy(float),
        sample["div_yield"].to_numpy(float),
        sample["market_iv"].to_numpy(float),
    )
    spread_iv = np.maximum(
        sample["spread"].to_numpy(float) / np.maximum(vega, 1e-12), SPREAD_IV_FLOOR
    )
    sample["spread_iv"] = spread_iv
    weight = 1.0 / spread_iv
    sample["weight"] = weight / weight.mean()

    return sample[out_columns].reset_index(drop=True)


def model_iv_for_sample(params: HestonParams, sample: pd.DataFrame) -> np.ndarray:
    """Heston model implied vols for every quote in a calibration sample.

    One Carr-Madan FFT per (date, maturity) group; call prices are then
    inverted to implied vols with a vectorised Newton solver seeded at the
    market IVs.

    Args:
        params: Heston parameters.
        sample: Frame with SAMPLE_COLUMNS.

    Returns:
        Model implied vols aligned with ``sample`` rows (nan on failure).
    """
    call_prices = np.empty(len(sample))
    for (_, _), idx in sample.groupby(["date", "maturity_years"]).groups.items():
        rows = sample.loc[idx]
        call_prices[np.asarray(idx)] = heston_call_prices(
            params,
            float(rows["spot"].iloc[0]),
            rows["strike"].to_numpy(float),
            float(rows["maturity_years"].iloc[0]),
            float(rows["rate"].iloc[0]),
            float(rows["div_yield"].iloc[0]),
        )
    return implied_vol_newton(
        call_prices,
        sample["spot"].to_numpy(float),
        sample["strike"].to_numpy(float),
        sample["maturity_years"].to_numpy(float),
        sample["rate"].to_numpy(float),
        sample["div_yield"].to_numpy(float),
        sample["market_iv"].to_numpy(float),
    )


def weighted_residuals(x: np.ndarray, sample: pd.DataFrame) -> np.ndarray:
    """Weighted IV residual vector for the LM optimiser.

    Args:
        x: Unbounded parameter vector (see :func:`transform_to_params`).
        sample: Calibration sample.

    Returns:
        w_i * (model_iv_i - market_iv_i), with failed model IVs replaced by
        the NAN_RESIDUAL penalty.
    """
    params = transform_to_params(x)
    model_iv = model_iv_for_sample(params, sample)
    residual = model_iv - sample["market_iv"].to_numpy(float)
    residual = np.where(np.isnan(residual), NAN_RESIDUAL, residual)
    return sample["weight"].to_numpy(float) * residual


def random_start(rng: np.random.Generator) -> HestonParams:
    """Draw one random restart point from the configured parameter ranges.

    Args:
        rng: Seeded random generator.

    Returns:
        Heston parameters inside the restart ranges.
    """
    return HestonParams(
        kappa=rng.uniform(*KAPPA_RANGE),
        theta=rng.uniform(*THETA_RANGE),
        sigma=rng.uniform(*SIGMA_RANGE),
        rho=rng.uniform(*RHO_RANGE),
        v0=rng.uniform(*V0_RANGE),
    )


def calibrate_window(
    sample: pd.DataFrame,
    rng: np.random.Generator,
    n_restarts: int = N_RESTARTS,
) -> CalibrationResult:
    """Calibrate Heston parameters to one pooled window sample.

    Args:
        sample: Calibration sample (SAMPLE_COLUMNS).
        rng: Seeded random generator driving the restarts.
        n_restarts: Number of Levenberg-Marquardt restarts.

    Returns:
        CalibrationResult with the best restart's parameters and fit stats.
    """
    if sample.empty:
        return CalibrationResult(
            params=HestonParams(*(float("nan"),) * 5),
            rmse=float("nan"),
            weighted_rmse=float("nan"),
            converged=False,
            n_restarts_converged=0,
            n_quotes=0,
            n_dates=0,
        )

    best_cost = np.inf
    best_x: np.ndarray | None = None
    n_ok = 0
    for _ in range(n_restarts):
        x0 = transform_from_params(random_start(rng))
        try:
            fit = least_squares(
                weighted_residuals,
                x0,
                args=(sample,),
                method="lm",
                max_nfev=MAX_NFEV,
            )
        except Exception as exc:  # LM can fail on pathological windows
            print(f"  restart failed: {exc}")
            continue
        if fit.success:
            n_ok += 1
        if fit.success and fit.cost < best_cost:
            best_cost = fit.cost
            best_x = fit.x

    if best_x is None:
        return CalibrationResult(
            params=HestonParams(*(float("nan"),) * 5),
            rmse=float("nan"),
            weighted_rmse=float("nan"),
            converged=False,
            n_restarts_converged=0,
            n_quotes=len(sample),
            n_dates=sample["date"].nunique(),
        )

    params = transform_to_params(best_x)
    model_iv = model_iv_for_sample(params, sample)
    diff = model_iv - sample["market_iv"].to_numpy(float)
    weight = sample["weight"].to_numpy(float)
    ok = ~np.isnan(diff)
    rmse = float(np.sqrt(np.mean(diff[ok] ** 2)))
    weighted_rmse = float(
        np.sqrt(np.sum((weight[ok] * diff[ok]) ** 2) / np.sum(weight[ok] ** 2))
    )
    return CalibrationResult(
        params=params,
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        converged=n_ok > 0,
        n_restarts_converged=n_ok,
        n_quotes=len(sample),
        n_dates=int(sample["date"].nunique()),
    )


def run_rolling_calibration(
    chain: pd.DataFrame,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    quarters: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Calibrate every quarter on its trailing one-year window.

    No lookahead: the window for quarter start Q is [Q - 365d, Q), strictly
    before Q.

    Args:
        chain: Full option chain.
        zero_curve: Zero curve frame.
        div_yield: Dividend-yield frame.
        quarters: Quarter starts to calibrate; defaults to the full schedule.

    Returns:
        One row per quarter with parameters, fit stats and convergence flag.
    """
    if quarters is None:
        quarters = quarter_starts()
    rows: list[dict[str, object]] = []
    for i, q_start in enumerate(quarters):
        window_start = q_start - pd.Timedelta(days=TRAILING_WINDOW_DAYS)
        sample = build_calibration_sample(
            chain, zero_curve, div_yield, window_start, q_start
        )
        rng = np.random.default_rng(RANDOM_SEED + i)
        result = calibrate_window(sample, rng)
        flag = "" if result.converged else "  ** DID NOT CONVERGE **"
        print(
            f"{q_start:%Y-%m-%d}: rmse={result.rmse:.4f} "
            f"kappa={result.params.kappa:.3f} theta={result.params.theta:.4f} "
            f"sigma={result.params.sigma:.3f} rho={result.params.rho:.3f} "
            f"v0={result.params.v0:.4f} "
            f"({result.n_quotes} quotes, {result.n_restarts_converged}/{N_RESTARTS} restarts){flag}",
            flush=True,
        )
        rows.append(
            {
                "quarter_start": q_start,
                "window_start": window_start,
                "window_end": q_start,
                "kappa": result.params.kappa,
                "theta": result.params.theta,
                "sigma": result.params.sigma,
                "rho": result.params.rho,
                "v0": result.params.v0,
                "rmse": result.rmse,
                "weighted_rmse": result.weighted_rmse,
                "converged": result.converged,
                "n_restarts_converged": result.n_restarts_converged,
                "n_quotes": result.n_quotes,
                "n_dates": result.n_dates,
                "feller_satisfied": result.params.feller_satisfied()
                if result.converged
                else False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point: ``python -m src.heston.calibration``."""
    chain = pd.read_parquet(OPTIONS_PARQUET)
    zero_curve = pd.read_parquet(ZERO_CURVE_PARQUET)
    div_yield = pd.read_parquet(DIV_YIELD_PARQUET)
    results = run_rolling_calibration(chain, zero_curve, div_yield)
    PARAMS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(PARAMS_PARQUET, index=False)
    n_failed = int((~results["converged"]).sum())
    print(f"\nwrote {len(results)} quarters to {PARAMS_PARQUET}")
    if n_failed:
        failed = results.loc[~results["converged"], "quarter_start"]
        print(f"WARNING: {n_failed} quarters failed to converge:")
        for q in failed:
            print(f"  {q:%Y-%m-%d}")
    else:
        print("all quarters converged")


if __name__ == "__main__":
    main()
