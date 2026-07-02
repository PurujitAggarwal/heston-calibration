"""Tests for src.heston.surface — model IV surface reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.fft import HestonParams
from src.heston.surface import (
    MATURITY_MONTHS_GRID,
    MONEYNESS_GRID,
    iv_rmse,
    model_iv_grid,
    nearest_trading_date,
    params_for_date,
    plot_model_vs_market,
    quarterly_surface_rmse,
)
from test_calibration import (
    TRUE_PARAMS,
    make_chain,
    make_div_yield,
    make_sample,
    make_zero_curve,
)

RATES = np.full(len(MATURITY_MONTHS_GRID), 0.02)


def make_params_table() -> pd.DataFrame:
    """One converged quarter covering H2-2020 plus one failed quarter."""
    return pd.DataFrame(
        [
            {
                "quarter_start": pd.Timestamp("2020-07-01"),
                "kappa": TRUE_PARAMS.kappa,
                "theta": TRUE_PARAMS.theta,
                "sigma": TRUE_PARAMS.sigma,
                "rho": TRUE_PARAMS.rho,
                "v0": TRUE_PARAMS.v0,
                "converged": True,
            },
            {
                "quarter_start": pd.Timestamp("2020-10-01"),
                "kappa": np.nan,
                "theta": np.nan,
                "sigma": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "converged": False,
            },
        ]
    )


def test_model_iv_grid_shape_and_values() -> None:
    grid = model_iv_grid(TRUE_PARAMS, 4000.0, RATES, 0.018)
    assert len(grid) == len(MONEYNESS_GRID) * len(MATURITY_MONTHS_GRID)
    assert not grid["model_iv"].isna().any()
    assert ((grid["model_iv"] > 0.05) & (grid["model_iv"] < 1.0)).all()
    assert set(grid["maturity_months"]) == set(MATURITY_MONTHS_GRID)


def test_model_iv_grid_negative_rho_skew() -> None:
    grid = model_iv_grid(TRUE_PARAMS, 4000.0, RATES, 0.018)
    short = grid.loc[grid["maturity_months"] == 1]
    low_strike = short.loc[short["moneyness"].idxmin(), "model_iv"]
    high_strike = short.loc[short["moneyness"].idxmax(), "model_iv"]
    assert low_strike > high_strike


def test_iv_rmse_zero_when_market_is_model() -> None:
    sample = make_sample()
    assert iv_rmse(TRUE_PARAMS, sample) < 1e-7
    other = HestonParams(kappa=2.0, theta=0.09, sigma=0.4, rho=-0.7, v0=0.09)
    assert iv_rmse(other, sample) > 0.01


def test_iv_rmse_empty_sample_is_nan() -> None:
    assert np.isnan(iv_rmse(TRUE_PARAMS, make_sample().iloc[0:0]))


def test_nearest_trading_date() -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-06-01", "2020-06-03", "2020-06-10"]))
    assert nearest_trading_date(dates, pd.Timestamp("2020-06-04")) == pd.Timestamp(
        "2020-06-03"
    )
    assert nearest_trading_date(dates, pd.Timestamp("2020-06-10")) == pd.Timestamp(
        "2020-06-10"
    )
    with pytest.raises(ValueError):
        nearest_trading_date(pd.DatetimeIndex([]), pd.Timestamp("2020-06-04"))


def test_params_for_date_selection_and_fallback() -> None:
    table = make_params_table()
    # date inside the converged quarter
    params = params_for_date(table, pd.Timestamp("2020-08-15"))
    assert params is not None and params.kappa == pytest.approx(TRUE_PARAMS.kappa)
    # date inside the failed quarter falls back to the last converged one
    params = params_for_date(table, pd.Timestamp("2020-11-15"))
    assert params is not None and params.rho == pytest.approx(TRUE_PARAMS.rho)
    # date before any calibration
    assert params_for_date(table, pd.Timestamp("2020-01-15")) is None


def test_quarterly_surface_rmse_scores_and_flags() -> None:
    table = quarterly_surface_rmse(
        make_chain(), make_zero_curve(), make_div_yield(), make_params_table()
    )
    assert len(table) == 2
    good = table.loc[table["converged"]].iloc[0]
    assert np.isfinite(good["rmse"]) and good["n_quotes"] > 0
    bad = table.loc[~table["converged"]].iloc[0]
    assert np.isnan(bad["rmse"]) and bad["n_quotes"] == 0


def test_plot_writes_png(tmp_path) -> None:
    out = tmp_path / "iv_surface.png"
    plotted = plot_model_vs_market(
        make_chain(),
        make_zero_curve(),
        make_div_yield(),
        make_params_table(),
        pd.Timestamp("2020-09-30"),
        output_path=out,
    )
    assert out.exists() and out.stat().st_size > 20_000
    assert plotted in pd.DatetimeIndex(make_chain()["date"].unique())
