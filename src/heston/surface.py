"""Implied volatility surface reconstruction from calibrated Heston parameters.

Reconstructs the model IV surface on a fixed moneyness/maturity grid, scores
the model against the market surface quarter by quarter (out of sample: each
quarter is priced with parameters calibrated on its trailing window), and
plots model vs market smiles for a sample date.

Run:
    python -m src.heston.surface [--plot-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.heston.calibration import (
    DAYS_PER_YEAR,
    PARAMS_PARQUET,
    PERCENT,
    build_calibration_sample,
    model_iv_for_sample,
)
from src.heston.data import (
    DAYS_PER_MONTH,
    DIV_YIELD_PARQUET,
    OPTIONS_PARQUET,
    PROJECT_ROOT,
    TARGET_TENOR_MONTHS,
    ZERO_CURVE_PARQUET,
    zero_rate,
)
from src.heston.fft import HestonParams, heston_call_prices, implied_vol_newton

# --- Model surface grid -----------------------------------------------------------
MONEYNESS_POINTS: int = 25
MONEYNESS_GRID: np.ndarray = np.linspace(0.80, 1.20, MONEYNESS_POINTS)
MATURITY_MONTHS_GRID: tuple[int, ...] = TARGET_TENOR_MONTHS  # 1..12 months, 6 points

# --- Plot -------------------------------------------------------------------------
DEFAULT_PLOT_DATE: str = "2023-06-30"
REPORTS_DIR = PROJECT_ROOT / "reports"
IV_SURFACE_PNG = REPORTS_DIR / "iv_surface.png"
MODEL_COLOR: str = "#2a78d6"
MARKET_COLOR: str = "#e34948"
SURFACE_COLOR: str = "#fcfcfb"
TEXT_COLOR: str = "#1a1a19"
MUTED_TEXT_COLOR: str = "#5f5e56"
GRID_COLOR: str = "#e4e3dd"

# --- Output -----------------------------------------------------------------------
SURFACE_RMSE_PARQUET = PROJECT_ROOT / "data" / "processed" / "surface_rmse.parquet"


def model_iv_grid(
    params: HestonParams,
    spot: float,
    rates: np.ndarray,
    div_yield: float,
) -> pd.DataFrame:
    """Model IV surface on the fixed moneyness x maturity grid.

    Args:
        params: Calibrated Heston parameters.
        spot: Index level.
        rates: Continuously compounded zero rates, one per grid maturity.
        div_yield: Continuously compounded dividend yield.

    Returns:
        Tidy frame with moneyness, maturity_months, maturity_years, strike
        and model_iv (25 x 6 = 150 rows).
    """
    frames: list[pd.DataFrame] = []
    strikes = MONEYNESS_GRID * spot
    for months, rate in zip(MATURITY_MONTHS_GRID, rates):
        maturity = months * DAYS_PER_MONTH / DAYS_PER_YEAR
        calls = heston_call_prices(params, spot, strikes, maturity, rate, div_yield)
        n = len(strikes)
        ivs = implied_vol_newton(
            calls,
            np.full(n, spot),
            strikes,
            np.full(n, maturity),
            np.full(n, rate),
            np.full(n, div_yield),
            np.full(n, np.nan),
        )
        frames.append(
            pd.DataFrame(
                {
                    "moneyness": MONEYNESS_GRID,
                    "maturity_months": months,
                    "maturity_years": maturity,
                    "strike": strikes,
                    "model_iv": ivs,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def iv_rmse(params: HestonParams, sample: pd.DataFrame) -> float:
    """Unweighted RMSE between model and market IV over a quote sample.

    Args:
        params: Heston parameters.
        sample: Quote sample with SAMPLE_COLUMNS.

    Returns:
        RMSE in vol units (nan if the sample is empty or nothing inverts).
    """
    if sample.empty:
        return float("nan")
    model_iv = model_iv_for_sample(params, sample)
    diff = model_iv - sample["market_iv"].to_numpy(float)
    ok = ~np.isnan(diff)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean(diff[ok] ** 2)))


def quarterly_surface_rmse(
    chain: pd.DataFrame,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    params_table: pd.DataFrame,
) -> pd.DataFrame:
    """Out-of-sample model-vs-market RMSE for every calibrated quarter.

    Each quarter's quotes (all trading days) are priced with the parameters
    calibrated on that quarter's trailing window.

    Args:
        chain: Full option chain.
        zero_curve: Zero curve frame.
        div_yield: Dividend-yield frame.
        params_table: Output of the rolling calibration.

    Returns:
        One row per quarter: quarter_start, rmse, n_quotes, n_dates,
        converged.
    """
    rows: list[dict[str, object]] = []
    table = params_table.sort_values("quarter_start").reset_index(drop=True)
    for _, row in table.iterrows():
        q_start = pd.Timestamp(row["quarter_start"])
        q_end = q_start + pd.offsets.QuarterBegin(startingMonth=1)
        if not row["converged"]:
            rows.append(
                {
                    "quarter_start": q_start,
                    "rmse": float("nan"),
                    "n_quotes": 0,
                    "n_dates": 0,
                    "converged": False,
                }
            )
            continue
        sample = build_calibration_sample(
            chain, zero_curve, div_yield, q_start, q_end, month_ends_only=False
        )
        params = HestonParams(
            kappa=float(row["kappa"]),
            theta=float(row["theta"]),
            sigma=float(row["sigma"]),
            rho=float(row["rho"]),
            v0=float(row["v0"]),
        )
        rows.append(
            {
                "quarter_start": q_start,
                "rmse": iv_rmse(params, sample),
                "n_quotes": len(sample),
                "n_dates": int(sample["date"].nunique()) if not sample.empty else 0,
                "converged": True,
            }
        )
    return pd.DataFrame(rows)


def nearest_trading_date(
    available: pd.DatetimeIndex, requested: pd.Timestamp
) -> pd.Timestamp:
    """Closest available trading date to the requested date.

    Args:
        available: Trading dates present in the chain.
        requested: Desired date.

    Returns:
        The nearest available date (ties resolve to the earlier one).

    Raises:
        ValueError: If ``available`` is empty.
    """
    if len(available) == 0:
        raise ValueError("no trading dates available")
    gaps = np.abs(available - requested)
    return pd.Timestamp(available[np.argmin(gaps)])


def params_for_date(
    params_table: pd.DataFrame, date: pd.Timestamp
) -> HestonParams | None:
    """Calibrated parameters governing a given trading date.

    The applicable row is the latest converged quarter_start at or before
    ``date`` (parameters are always calibrated strictly before their
    quarter, so this never looks ahead).

    Args:
        params_table: Output of the rolling calibration.
        date: Trading date.

    Returns:
        HestonParams, or None if no converged quarter covers the date.
    """
    table = params_table.loc[
        params_table["converged"]
        & (pd.to_datetime(params_table["quarter_start"]) <= date)
    ].sort_values("quarter_start")
    if table.empty:
        return None
    row = table.iloc[-1]
    return HestonParams(
        kappa=float(row["kappa"]),
        theta=float(row["theta"]),
        sigma=float(row["sigma"]),
        rho=float(row["rho"]),
        v0=float(row["v0"]),
    )


def plot_model_vs_market(
    chain: pd.DataFrame,
    zero_curve: pd.DataFrame,
    div_yield: pd.DataFrame,
    params_table: pd.DataFrame,
    plot_date: pd.Timestamp,
    output_path: Path = IV_SURFACE_PNG,
) -> pd.Timestamp:
    """Plot model vs market IV smiles per maturity for one trading date.

    Args:
        chain: Full option chain.
        zero_curve: Zero curve frame.
        div_yield: Dividend-yield frame.
        params_table: Output of the rolling calibration.
        plot_date: Requested date (snapped to the nearest trading date).
        output_path: Destination PNG path.

    Returns:
        The trading date actually plotted.

    Raises:
        ValueError: If no converged parameters cover the plot date or the
            date has no usable quotes.
    """
    available = pd.DatetimeIndex(np.sort(chain["date"].unique()))
    date = nearest_trading_date(available, plot_date)
    params = params_for_date(params_table, date)
    if params is None:
        raise ValueError(f"no converged calibration covers {date:%Y-%m-%d}")
    sample = build_calibration_sample(
        chain,
        zero_curve,
        div_yield,
        date,
        date + pd.Timedelta(days=1),
        month_ends_only=False,
    )
    if sample.empty:
        raise ValueError(f"no usable quotes on {date:%Y-%m-%d}")

    spot = float(sample["spot"].iloc[0])
    div = float(sample["div_yield"].iloc[0])
    maturities = np.sort(sample["maturity_years"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharey=True)
    fig.patch.set_facecolor(SURFACE_COLOR)
    for ax, maturity in zip(axes.flat, maturities):
        slice_ = sample.loc[sample["maturity_years"] == maturity]
        rate = float(slice_["rate"].iloc[0])
        strikes = MONEYNESS_GRID * spot
        calls = heston_call_prices(params, spot, strikes, float(maturity), rate, div)
        n = len(strikes)
        model_iv = implied_vol_newton(
            calls,
            np.full(n, spot),
            strikes,
            np.full(n, float(maturity)),
            np.full(n, rate),
            np.full(n, div),
            np.full(n, np.nan),
        )
        ax.set_facecolor(SURFACE_COLOR)
        ax.plot(
            MONEYNESS_GRID,
            model_iv * PERCENT,
            color=MODEL_COLOR,
            linewidth=2.0,
            label="Heston model",
            zorder=3,
        )
        ax.scatter(
            slice_["strike"] / spot,
            slice_["market_iv"] * PERCENT,
            s=22,
            color=MARKET_COLOR,
            label="Market",
            zorder=4,
            edgecolors=SURFACE_COLOR,
            linewidths=0.8,
        )
        months = maturity * DAYS_PER_YEAR / DAYS_PER_MONTH
        ax.set_title(f"{months:.1f} months", color=TEXT_COLOR, fontsize=11)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7)
        ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
    for ax in axes.flat[len(maturities):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Moneyness K/S", color=MUTED_TEXT_COLOR, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied vol (%)", color=MUTED_TEXT_COLOR, fontsize=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=10,
        labelcolor=TEXT_COLOR,
    )
    fig.suptitle(
        f"SPX implied volatility: Heston model vs market — {date:%Y-%m-%d}\n"
        f"$\\kappa$={params.kappa:.2f}  $\\theta$={params.theta:.4f}  "
        f"$\\sigma$={params.sigma:.2f}  $\\rho$={params.rho:.2f}  "
        f"$v_0$={params.v0:.4f}",
        color=TEXT_COLOR,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE_COLOR)
    plt.close(fig)
    return date


def main() -> None:
    """CLI entry point: ``python -m src.heston.surface``."""
    parser = argparse.ArgumentParser(description="Reconstruct model IV surface.")
    parser.add_argument("--plot-date", type=str, default=DEFAULT_PLOT_DATE)
    args = parser.parse_args()

    chain = pd.read_parquet(OPTIONS_PARQUET)
    zero_curve = pd.read_parquet(ZERO_CURVE_PARQUET)
    div_yield = pd.read_parquet(DIV_YIELD_PARQUET)
    params_table = pd.read_parquet(PARAMS_PARQUET)

    rmse_table = quarterly_surface_rmse(chain, zero_curve, div_yield, params_table)
    SURFACE_RMSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rmse_table.to_parquet(SURFACE_RMSE_PARQUET, index=False)
    with pd.option_context("display.max_rows", None):
        display = rmse_table.assign(
            rmse_volpts=(rmse_table["rmse"] * PERCENT).round(2)
        )[["quarter_start", "rmse_volpts", "n_quotes", "n_dates", "converged"]]
        print(display.to_string(index=False))
    print(
        f"\nmean quarterly RMSE: {np.nanmean(rmse_table['rmse']) * PERCENT:.2f} "
        f"vol pts; wrote {SURFACE_RMSE_PARQUET}"
    )

    plotted = plot_model_vs_market(
        chain, zero_curve, div_yield, params_table, pd.Timestamp(args.plot_date)
    )
    print(f"wrote {IV_SURFACE_PNG} for {plotted:%Y-%m-%d}")


if __name__ == "__main__":
    main()
