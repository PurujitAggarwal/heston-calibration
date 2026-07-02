"""Tests for src.heston.signals — deviation signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.signals import (
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    LONG_VOL_ENABLED,
    MAX_SPREAD_IV,
    add_signal_flags,
    build_signal_panel,
    entry_events,
    summarize_panel,
)
from test_calibration import make_chain, make_div_yield, make_zero_curve
from test_surface import make_params_table


def make_panel() -> pd.DataFrame:
    """Hand-built deviation panel covering every flag branch."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-08-03"] * 6),
            "optionid": [1, 2, 3, 4, 5, 6],
            "deviation": [0.030, -0.030, 0.030, 0.004, 0.020, np.nan],
            "spread_iv": [0.001, 0.002, 0.008, 0.001, 0.001, 0.001],
        }
    )


def test_add_signal_flags_directions() -> None:
    panel = add_signal_flags(make_panel())
    # rich vol + liquid -> short entry
    assert bool(panel.loc[0, "entry_short"]) and not bool(panel.loc[0, "entry_long"])
    # cheap vol + liquid would be a long entry, but the leg is disabled
    assert not bool(panel.loc[1, "entry_long"])
    assert not bool(panel.loc[1, "entry_short"])


def test_long_vol_leg_permanently_disabled() -> None:
    assert LONG_VOL_ENABLED is False
    panel = make_panel()
    panel["deviation"] = -0.10  # extreme cheap-vol deviation on every row
    flagged = add_signal_flags(panel)
    assert not flagged["entry_long"].any()


def test_add_signal_flags_liquidity_blocks_entry() -> None:
    panel = add_signal_flags(make_panel())
    # same deviation as row 0 but spread too wide
    assert not bool(panel.loc[2, "liquid"])
    assert not bool(panel.loc[2, "entry_short"])


def test_add_signal_flags_exit_and_thresholds() -> None:
    panel = add_signal_flags(make_panel())
    # |dev| < 0.5 pts -> exit flag
    assert bool(panel.loc[3, "exit_signal"])
    # dev exactly at the entry threshold does not enter (strict inequality)
    assert not bool(panel.loc[4, "entry_short"])
    # nan model IV -> no flags at all
    assert not panel.loc[5, ["liquid", "entry_short", "entry_long", "exit_signal"]].any()


def test_threshold_constants_match_spec() -> None:
    assert ENTRY_THRESHOLD == pytest.approx(0.02)
    assert EXIT_THRESHOLD == pytest.approx(0.005)
    assert MAX_SPREAD_IV == pytest.approx(0.005)


def test_entry_events_direction_and_ordering() -> None:
    panel = add_signal_flags(make_panel())
    panel.loc[4, "deviation"] = 0.050  # push row 4 over the entry threshold
    panel.loc[4, "entry_short"] = True
    events = entry_events(panel)
    assert len(events) == 2
    # strongest |deviation| first; all remaining entries are short vol
    assert events.loc[0, "optionid"] == 5 and events.loc[0, "direction"] == -1
    assert events.loc[1, "optionid"] == 1 and events.loc[1, "direction"] == -1


def test_build_signal_panel_no_lookahead_and_columns() -> None:
    panel = build_signal_panel(
        make_chain(), make_zero_curve(), make_div_yield(), make_params_table()
    )
    # chain starts 2020-06-01 but first calibrated quarter is 2020-07-01:
    # earlier dates must be absent (no parameters -> no signals)
    assert panel["date"].min() >= pd.Timestamp("2020-07-01")
    for col in ("optionid", "cp_flag", "model_iv", "deviation", "spread_iv", "mid"):
        assert col in panel.columns
    # deviation is market - model wherever the model inverted
    ok = panel["model_iv"].notna()
    assert ok.any()
    np.testing.assert_allclose(
        panel.loc[ok, "deviation"],
        panel.loc[ok, "market_iv"] - panel.loc[ok, "model_iv"],
    )


def test_build_signal_panel_uses_fallback_params_for_failed_quarter() -> None:
    # Q4-2020 failed to converge; params_for_date falls back to Q3 params,
    # so Q4 dates must still be present in the panel.
    panel = build_signal_panel(
        make_chain(), make_zero_curve(), make_div_yield(), make_params_table()
    )
    assert (panel["date"] >= pd.Timestamp("2020-10-01")).any()


def test_summarize_panel_counts() -> None:
    panel = add_signal_flags(make_panel())
    summary = summarize_panel(panel)
    assert summary.loc[2020, "quotes"] == 6
    assert summary.loc[2020, "entries_short"] == 1
    assert summary.loc[2020, "entries_long"] == 0  # leg disabled
