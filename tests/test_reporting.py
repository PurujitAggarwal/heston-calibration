"""Tests for src.heston.reporting — performance metrics and plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.reporting import (
    TRADING_DAYS_PER_YEAR,
    annualised_sharpe,
    annualised_sortino,
    compute_metrics,
    daily_returns,
    format_report,
    max_drawdown,
    write_equity_plot,
)


def make_curve() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {"date": dates, "equity": [100_000.0, 102_000.0, 99_000.0, 101_000.0, 103_000.0]}
    )


def make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 200.0, -25.0],
            "holding_days": [2, 4, 6, 8],
        }
    )


def test_daily_returns() -> None:
    returns = daily_returns(make_curve()["equity"])
    assert len(returns) == 4
    assert returns.iloc[0] == pytest.approx(0.02)


def test_annualised_sharpe_hand_computed() -> None:
    # mean 0.01, std 0.011547 -> sharpe = sqrt(252) * 0.866
    returns = pd.Series([0.02, 0.0, 0.02, 0.0])
    expected = np.sqrt(TRADING_DAYS_PER_YEAR) * 0.01 / returns.std(ddof=1)
    assert annualised_sharpe(returns) == pytest.approx(expected)
    # zero-variance series has no defined Sharpe
    assert np.isnan(annualised_sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_annualised_sortino_downside_only() -> None:
    returns = pd.Series([0.02, -0.01, 0.02, -0.01])
    downside = np.sqrt(np.mean([0.0, 0.01**2, 0.0, 0.01**2]))
    expected = np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / downside
    assert annualised_sortino(returns) == pytest.approx(expected)
    # no negative returns -> undefined
    assert np.isnan(annualised_sortino(pd.Series([0.01, 0.02])))


def test_max_drawdown_known_path() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(equity) == pytest.approx(90.0 / 120.0 - 1.0)
    # monotone series never draws down
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_compute_metrics_keys_and_values() -> None:
    metrics = compute_metrics(make_curve(), make_trades())
    assert metrics["final_equity"] == pytest.approx(103_000.0)
    assert metrics["total_return"] == pytest.approx(0.03)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["avg_holding_days"] == pytest.approx(5.0)
    assert metrics["n_trades"] == 4
    for key in ("sharpe", "sortino", "max_drawdown"):
        assert np.isfinite(metrics[key])


def test_format_report_contains_required_lines() -> None:
    report = format_report(compute_metrics(make_curve(), make_trades()))
    for label in (
        "Sharpe", "Sortino", "drawdown", "Win rate", "holding", "trades",
        "Final equity",
    ):
        assert label in report


def test_write_equity_plot_creates_png(tmp_path) -> None:
    out = tmp_path / "equity.png"
    write_equity_plot(make_curve(), out)
    assert out.exists() and out.stat().st_size > 10_000
