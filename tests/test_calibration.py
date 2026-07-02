"""Tests for src.heston.calibration — rolling LM calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.calibration import (
    SAMPLE_COLUMNS,
    build_calibration_sample,
    calibrate_window,
    model_iv_for_sample,
    month_end_dates,
    quarter_starts,
    transform_from_params,
    transform_to_params,
    weighted_residuals,
)
from src.heston.fft import HestonParams

TRUE_PARAMS = HestonParams(
    kappa=2.0, theta=0.045, sigma=0.4, rho=-0.7, v0=0.03
)


def make_sample() -> pd.DataFrame:
    """Synthetic pooled sample: 2 dates x 3 maturities x 9 strikes."""
    rows = []
    for date, spot in (("2020-11-30", 3620.0), ("2020-12-31", 3756.0)):
        for maturity in (0.1, 0.25, 0.5):
            for strike_ratio in np.linspace(0.85, 1.15, 9):
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "maturity_years": maturity,
                        "strike": spot * strike_ratio,
                        "spot": spot,
                        "rate": 0.01,
                        "div_yield": 0.018,
                        "market_iv": np.nan,
                        "weight": 1.0,
                    }
                )
    sample = pd.DataFrame(rows)
    sample["market_iv"] = model_iv_for_sample(TRUE_PARAMS, sample)
    assert not sample["market_iv"].isna().any()
    return sample


def make_chain() -> pd.DataFrame:
    """Synthetic chain frame shaped like data.ingest output."""
    rows = []
    for date in pd.bdate_range("2020-06-01", "2020-12-31"):
        spot = 3000.0
        for dte in (30, 91, 182):
            for strike in (2700.0, 2900.0, 3100.0, 3300.0):
                cp = "P" if strike < spot else "C"
                rows.append(
                    {
                        "date": date,
                        "exdate": date + pd.Timedelta(days=dte),
                        "cp_flag": cp,
                        "strike": strike,
                        "best_bid": 10.0,
                        "best_offer": 11.0,
                        "impl_volatility": 0.2,
                        "mid": 10.5,
                        "spread": 1.0,
                        "spot": spot,
                        "days_to_expiry": dte,
                        "moneyness": strike / spot,
                        "volume": 1.0,
                        "open_interest": 1.0,
                        "optionid": hash((strike, dte, cp)) % 10_000_000,
                    }
                )
    return pd.DataFrame(rows)


def make_zero_curve() -> pd.DataFrame:
    dates = pd.bdate_range("2020-05-01", "2021-01-15")
    rows = [
        {"date": d, "days": days, "rate": 1.0 + days / 3650.0}
        for d in dates
        for days in (7, 30, 91, 182, 365)
    ]
    return pd.DataFrame(rows)


def make_div_yield() -> pd.DataFrame:
    dates = pd.bdate_range("2020-05-01", "2021-01-15")
    return pd.DataFrame({"date": dates, "rate": 1.8})


def test_transform_round_trip() -> None:
    x = transform_from_params(TRUE_PARAMS)
    back = transform_to_params(x)
    assert back.kappa == pytest.approx(TRUE_PARAMS.kappa)
    assert back.theta == pytest.approx(TRUE_PARAMS.theta)
    assert back.sigma == pytest.approx(TRUE_PARAMS.sigma)
    assert back.rho == pytest.approx(TRUE_PARAMS.rho)
    assert back.v0 == pytest.approx(TRUE_PARAMS.v0)


def test_transform_always_valid() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = transform_to_params(rng.normal(scale=3.0, size=5))
        assert p.kappa > 0 and p.theta > 0 and p.sigma > 0 and p.v0 > 0
        assert -1.0 < p.rho < 1.0


def test_quarter_starts_schedule() -> None:
    quarters = quarter_starts("2011-01-01", "2025-07-01")
    assert quarters[0] == pd.Timestamp("2011-01-01")
    assert quarters[-1] == pd.Timestamp("2025-07-01")
    assert len(quarters) == 59
    assert all(q.month in (1, 4, 7, 10) and q.day == 1 for q in quarters)


def test_month_end_dates() -> None:
    dates = pd.Series(pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"]))
    ends = month_end_dates(dates)
    assert list(ends) == list(pd.to_datetime(["2020-01-31", "2020-02-28"]))


def test_build_sample_no_lookahead_and_otm_only() -> None:
    window_end = pd.Timestamp("2021-01-01")
    sample = build_calibration_sample(
        make_chain(), make_zero_curve(), make_div_yield(),
        pd.Timestamp("2020-01-01"), window_end,
    )
    assert not sample.empty
    assert (sample["date"] < window_end).all()
    # month-end pooling only
    assert sample["date"].nunique() <= 7
    # OTM: puts below spot, calls above -> every strike/spot != 1 kept once
    assert set(SAMPLE_COLUMNS) == set(sample.columns)
    assert (sample["weight"] > 0).all()
    assert sample["weight"].mean() == pytest.approx(1.0)
    # rates merged and converted from percent
    assert ((sample["rate"] > 0.005) & (sample["rate"] < 0.02)).all()
    assert np.allclose(sample["div_yield"], 0.018)


def test_build_sample_empty_window_returns_empty() -> None:
    sample = build_calibration_sample(
        make_chain(), make_zero_curve(), make_div_yield(),
        pd.Timestamp("2019-01-01"), pd.Timestamp("2019-06-30"),
    )
    assert sample.empty
    assert list(sample.columns) == list(SAMPLE_COLUMNS)


def test_residuals_zero_at_true_params() -> None:
    sample = make_sample()
    residual = weighted_residuals(transform_from_params(TRUE_PARAMS), sample)
    assert np.max(np.abs(residual)) < 1e-6


def test_calibrate_window_recovers_true_params() -> None:
    sample = make_sample()
    rng = np.random.default_rng(7)
    result = calibrate_window(sample, rng, n_restarts=3)
    assert result.converged
    assert result.rmse < 5e-4
    assert result.params.rho == pytest.approx(TRUE_PARAMS.rho, abs=0.05)
    assert result.params.v0 == pytest.approx(TRUE_PARAMS.v0, rel=0.10)
    assert result.params.theta == pytest.approx(TRUE_PARAMS.theta, rel=0.15)


def test_calibrate_window_empty_sample_flags_not_converged() -> None:
    result = calibrate_window(
        pd.DataFrame(columns=list(SAMPLE_COLUMNS)), np.random.default_rng(0)
    )
    assert not result.converged
    assert result.n_quotes == 0
    assert np.isnan(result.params.kappa)
