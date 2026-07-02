"""Offline tests for src.heston.data (no WRDS connection required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.data import (
    DAYS_PER_MONTH,
    MONEYNESS_MAX,
    MONEYNESS_MIN,
    SPX_SECID,
    add_derived_columns,
    build_option_query,
    drop_crossed_quotes,
    select_nearest_expiries,
    target_tenor_days,
    validate_chain,
    zero_rate,
)


def make_chain() -> pd.DataFrame:
    """Build a minimal valid two-row chain fixture."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-14", "2024-06-14"]),
            "exdate": pd.to_datetime(["2024-07-19", "2024-09-20"]),
            "cp_flag": ["C", "P"],
            "strike": [5400.0, 5000.0],
            "best_bid": [100.0, 40.0],
            "best_offer": [102.0, 42.0],
            "impl_volatility": [0.12, 0.15],
            "volume": [10.0, 5.0],
            "open_interest": [100.0, 50.0],
            "spot": [5431.6, 5431.6],
        }
    )


def test_add_derived_columns_mid_and_spreads() -> None:
    out = add_derived_columns(make_chain())
    assert out.loc[0, "mid"] == pytest.approx(101.0)
    assert out.loc[0, "spread"] == pytest.approx(2.0)
    assert out.loc[0, "rel_spread"] == pytest.approx(2.0 / 101.0)


def test_add_derived_columns_zero_mid_gives_nan_rel_spread() -> None:
    chain = make_chain()
    chain.loc[0, ["best_bid", "best_offer"]] = 0.0
    out = add_derived_columns(chain)
    assert np.isnan(out.loc[0, "rel_spread"])


def test_add_derived_columns_dte_and_moneyness() -> None:
    out = add_derived_columns(make_chain())
    assert out.loc[0, "days_to_expiry"] == 35
    assert out.loc[1, "days_to_expiry"] == 98
    assert out.loc[0, "moneyness"] == pytest.approx(5400.0 / 5431.6)


def test_target_tenor_days_grid() -> None:
    days = target_tenor_days((1, 12))
    assert days[0] == pytest.approx(DAYS_PER_MONTH)
    assert days[1] == pytest.approx(12 * DAYS_PER_MONTH)


def test_select_nearest_expiries_picks_nearest() -> None:
    listed = [28, 63, 91, 175, 280, 371]
    selection = select_nearest_expiries(listed)
    targets = target_tenor_days()
    assert selection[float(targets[0])] == 28
    assert selection[float(targets[2])] == 91
    assert selection[float(targets[5])] == 371


def test_select_nearest_expiries_respects_tolerance() -> None:
    # Only a ~1M expiry listed: distant targets must not be force-mapped.
    selection = select_nearest_expiries([30], tolerance_days=10.0)
    assert list(selection.values()) == [30]
    assert len(selection) == 1


def test_select_nearest_expiries_no_double_assignment() -> None:
    # One listed expiry cannot serve two targets.
    selection = select_nearest_expiries([45], tolerance_days=46.0)
    assert len(selection) == 1


def test_zero_rate_interpolates_linearly() -> None:
    curve = pd.DataFrame({"days": [30, 90], "rate": [4.0, 5.0]})
    assert zero_rate(curve, 60.0) == pytest.approx(4.5)


def test_zero_rate_flat_extrapolation_and_empty_raises() -> None:
    curve = pd.DataFrame({"days": [30, 90], "rate": [4.0, 5.0]})
    assert zero_rate(curve, 10.0) == pytest.approx(4.0)
    assert zero_rate(curve, 400.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        zero_rate(curve.iloc[0:0], 30.0)


def test_validate_chain_accepts_good_frame() -> None:
    validate_chain(make_chain())


def test_validate_chain_rejects_bad_frames() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_chain(make_chain().drop(columns=["spot"]))
    bad_strike = make_chain()
    bad_strike.loc[0, "strike"] = -1.0
    with pytest.raises(ValueError, match="strike"):
        validate_chain(bad_strike)
    crossed = make_chain()
    crossed.loc[0, "best_bid"] = 200.0
    with pytest.raises(ValueError, match="crossed"):
        validate_chain(crossed)


def test_drop_crossed_quotes_removes_only_crossed() -> None:
    chain = make_chain()
    chain.loc[1, "best_bid"] = 100.0  # crossed: bid 100 > offer 42
    out = drop_crossed_quotes(chain)
    assert len(out) == 1
    assert out.loc[0, "cp_flag"] == "C"
    untouched = drop_crossed_quotes(make_chain())
    assert len(untouched) == 2


def test_build_option_query_filters() -> None:
    sql = build_option_query(2024)
    assert f"opprcd2024" in sql
    assert str(SPX_SECID) in sql
    assert "am_settlement = 1" in sql
    assert "expiry_indicator is null" in sql
    assert str(MONEYNESS_MIN) in sql and str(MONEYNESS_MAX) in sql
