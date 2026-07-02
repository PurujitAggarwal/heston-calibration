"""Tests for src.heston.backtest — delta-hedged simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.heston.backtest import (
    CALM_ALLOCATION_FRACTION,
    CALM_MAX_POSITIONS,
    HIGH_VOL_ALLOCATION_FRACTION,
    HIGH_VOL_MAX_POSITIONS,
    MAX_ALLOCATION_FRACTION,
    MAX_NEW_POSITIONS_PER_DAY,
    REGIME_MIN_PERIODS,
    SIZING_REF_VOL,
    STARTING_CAPITAL,
    STOP_LOSS_FRACTION,
    high_vol_regime,
    run_backtest,
    size_position,
    trade_cost,
)
from test_surface import make_params_table


def panel_row(
    date: str,
    optionid: int,
    mid: float,
    spot: float,
    deviation: float,
    market_iv: float = 0.2,
    entry_long: bool = False,
    entry_short: bool = False,
    exit_signal: bool = False,
    days_to_expiry: int = 90,
    strike: float = 3000.0,
    cp_flag: str = "P",
) -> dict[str, object]:
    """One synthetic signal-panel row with sane defaults."""
    return {
        "date": pd.Timestamp(date),
        "optionid": optionid,
        "cp_flag": cp_flag,
        "exdate": pd.Timestamp(date) + pd.Timedelta(days=days_to_expiry),
        "strike": strike,
        "spot": spot,
        "mid": mid,
        "best_bid": mid - 0.5,
        "best_offer": mid + 0.5,
        "spread_iv": 0.001,
        "days_to_expiry": days_to_expiry,
        "maturity_years": days_to_expiry / 365.0,
        "rate": 0.01,
        "div_yield": 0.018,
        "market_iv": market_iv,
        "model_iv": market_iv - deviation,
        "deviation": deviation,
        "liquid": True,
        "entry_long": entry_long,
        "entry_short": entry_short,
        "exit_signal": exit_signal,
    }


def test_size_position_inverse_vol_and_cap() -> None:
    equity = 100_000.0
    # IV 20% -> exactly the regime base fraction
    assert size_position(equity, 0.20, CALM_ALLOCATION_FRACTION) == pytest.approx(
        equity * CALM_ALLOCATION_FRACTION
    )
    # IV 40% -> half the base fraction
    assert size_position(equity, 0.40, CALM_ALLOCATION_FRACTION) == pytest.approx(
        equity * CALM_ALLOCATION_FRACTION / 2.0
    )
    # high-vol regime halves the base
    assert size_position(equity, 0.20, HIGH_VOL_ALLOCATION_FRACTION) == pytest.approx(
        equity * HIGH_VOL_ALLOCATION_FRACTION
    )
    # IV 2% -> inverse-vol multiple of 10 would exceed the 5% cap
    assert size_position(equity, 0.02, CALM_ALLOCATION_FRACTION) == pytest.approx(
        equity * MAX_ALLOCATION_FRACTION
    )
    assert SIZING_REF_VOL == pytest.approx(0.20)


def test_trade_cost_is_3bps() -> None:
    assert trade_cost(10_000.0) == pytest.approx(3.0)
    assert trade_cost(-10_000.0) == pytest.approx(3.0)


def test_revert_exit_books_profitable_long_vol_trade() -> None:
    # Long vol entry (market 3 pts under model); option mid then rises and
    # the deviation reverts -> exit with a gain.
    # OTM strike keeps the model delta small enough that the leverage cap
    # does not bind, so units = allocation / entry mid exactly.
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      entry_long=True, strike=2600.0),
            panel_row("2020-08-04", 1, mid=60.0, spot=3000.0, deviation=-0.02,
                      strike=2600.0),
            panel_row("2020-08-05", 1, mid=65.0, spot=3000.0, deviation=0.001,
                      exit_signal=True, strike=2600.0),
        ]
    )
    curve, trades = run_backtest(panel, make_params_table())
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "revert"
    assert trade["direction"] == 1
    assert trade["holding_days"] == 2
    # spot never moved, so hedge P&L is zero; P&L is the option gain less
    # transaction costs
    units = trade["allocation"] / 50.0
    assert trade["units"] == pytest.approx(units)
    gross = units * 15.0
    assert gross * 0.90 < trade["pnl"] < gross
    assert curve["equity"].iloc[-1] > STARTING_CAPITAL


def test_stop_loss_triggers_on_short_vol_blowup() -> None:
    # Short vol entry; option mid doubles day after day -> stop loss.
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=0.03,
                      entry_short=True),
            panel_row("2020-08-04", 1, mid=100.0, spot=3000.0, deviation=0.04),
            panel_row("2020-08-05", 1, mid=200.0, spot=3000.0, deviation=0.05),
        ]
    )
    curve, trades = run_backtest(panel, make_params_table())
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["pnl"] < -STOP_LOSS_FRACTION * trade["allocation"] * 0.9
    assert curve["equity"].iloc[-1] < STARTING_CAPITAL


def test_daily_entry_cap_and_no_duplicate_positions() -> None:
    rows = [
        panel_row("2020-08-03", oid, mid=50.0, spot=3000.0, deviation=-0.03,
                  entry_long=True, strike=2500.0 + oid)
        for oid in range(1, 31)
    ]
    # same options still flagged the next day: must not re-enter duplicates
    rows += [
        panel_row("2020-08-04", oid, mid=50.0, spot=3000.0, deviation=-0.03,
                  entry_long=True, strike=2500.0 + oid)
        for oid in range(1, 31)
    ]
    panel = pd.DataFrame(rows)
    curve, trades = run_backtest(panel, make_params_table())
    assert curve.loc[0, "n_positions"] == MAX_NEW_POSITIONS_PER_DAY
    assert curve.loc[1, "n_positions"] == 2 * MAX_NEW_POSITIONS_PER_DAY
    assert int(curve["n_positions"].max()) <= CALM_MAX_POSITIONS


def test_strongest_signals_enter_first() -> None:
    rows = [
        panel_row("2020-08-03", oid, mid=50.0, spot=3000.0,
                  deviation=-(0.02 + oid / 1000.0), entry_long=True)
        for oid in range(1, 21)
    ]
    panel = pd.DataFrame(rows)
    _, trades = run_backtest(panel, make_params_table())
    # only the sample end closes these ("final"); the 10 largest |dev| win
    entered = set(trades["optionid"])
    assert entered == set(range(11, 21))


def test_delisted_position_closed_at_last_mark() -> None:
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      entry_long=True),
            panel_row("2020-08-04", 2, mid=10.0, spot=3000.0, deviation=0.0),
        ]
    )
    _, trades = run_backtest(panel, make_params_table())
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "delisted"


def test_expiry_exit_when_dte_hits_floor() -> None:
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      entry_long=True, days_to_expiry=12),
            panel_row("2020-08-10", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      days_to_expiry=5),
        ]
    )
    _, trades = run_backtest(panel, make_params_table())
    assert trades.iloc[0]["exit_reason"] == "expiry"


def test_min_premium_blocks_entry() -> None:
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=1.5, spot=3000.0, deviation=-0.03,
                      entry_long=True),
            panel_row("2020-08-04", 1, mid=1.5, spot=3000.0, deviation=-0.03),
        ]
    )
    curve, trades = run_backtest(panel, make_params_table())
    assert trades.empty
    assert int(curve["n_positions"].max()) == 0


def test_leverage_cap_scales_down_atm_units() -> None:
    # ATM put has |delta| ~0.45: uncapped hedge notional would be ~27x the
    # allocation, so units must come out well below allocation / mid.
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      entry_long=True),
            panel_row("2020-08-04", 1, mid=50.0, spot=3000.0, deviation=0.0,
                      exit_signal=True),
        ]
    )
    _, trades = run_backtest(panel, make_params_table())
    trade = trades.iloc[0]
    assert trade["units"] < trade["allocation"] / 50.0


def test_stop_cooldown_blocks_reentry_for_five_days() -> None:
    rows = [
        panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                  entry_long=True),
        panel_row("2020-08-04", 1, mid=20.0, spot=3000.0, deviation=-0.04),
    ]
    # five trading days of persistent signal after the stop: barred
    for day in ("2020-08-05", "2020-08-06", "2020-08-07", "2020-08-10",
                "2020-08-11"):
        rows.append(panel_row(day, 1, mid=20.0, spot=3000.0, deviation=-0.04,
                              entry_long=True))
    # sixth trading day after the stop: allowed again
    rows.append(panel_row("2020-08-12", 1, mid=20.0, spot=3000.0,
                          deviation=-0.04, entry_long=True))
    _, trades = run_backtest(pd.DataFrame(rows), make_params_table())
    assert len(trades) == 2
    assert trades.iloc[0]["exit_reason"] == "stop"
    assert trades.iloc[1]["entry_date"] == pd.Timestamp("2020-08-12")


def test_high_vol_regime_lags_one_day() -> None:
    # 300 flat days (zero realised vol), then a 10% jump: the jump day
    # itself must still be calm (only t-1 information), the next day high.
    dates = pd.bdate_range("2019-01-01", periods=303)
    closes = np.full(len(dates), 100.0)
    closes[301:] = 110.0
    spot = pd.DataFrame({"date": dates, "close": closes})
    regime = high_vol_regime(spot)
    assert not bool(regime.iloc[301])  # jump day: classified with old info
    assert bool(regime.iloc[302])  # day after: jump visible in lagged vol
    # warm-up period is always calm
    assert not regime.iloc[:REGIME_MIN_PERIODS].any()


def test_high_vol_regime_percentile_calculation() -> None:
    # Wiring check against an independently computed spec on a seeded walk,
    # plus a behavioural check that roughly the top quintile is flagged.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2017-01-01", periods=800)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    spot = pd.DataFrame({"date": dates, "close": closes})
    regime = high_vol_regime(spot)

    series = pd.Series(closes, index=dates)
    vol = np.log(series).diff().rolling(21).std() * np.sqrt(252.0)
    lagged = vol.shift(1)
    threshold = lagged.rolling(252, min_periods=126).quantile(0.80)
    expected = (lagged > threshold).fillna(False)
    assert regime.equals(expected)
    share = regime.iloc[300:].mean()
    assert 0.05 < share < 0.35


def test_regime_switch_halves_allocation() -> None:
    regime = pd.Series(True, index=[pd.Timestamp("2020-08-04")])
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=0.03,
                      entry_short=True, strike=2600.0),
            panel_row("2020-08-04", 1, mid=50.0, spot=3000.0, deviation=0.03,
                      strike=2600.0),
            panel_row("2020-08-04", 2, mid=50.0, spot=3000.0, deviation=0.03,
                      entry_short=True, strike=2600.0),
        ]
    )
    _, trades = run_backtest(panel, make_params_table(), high_regime=regime)
    trades = trades.set_index("optionid")
    # calm day: 1% of exactly $100,000 at IV 20%
    assert trades.loc[1, "allocation"] == pytest.approx(
        STARTING_CAPITAL * CALM_ALLOCATION_FRACTION
    )
    # high-vol day: 0.5% of (near-)unchanged equity
    assert trades.loc[2, "allocation"] == pytest.approx(
        STARTING_CAPITAL * HIGH_VOL_ALLOCATION_FRACTION, rel=0.01
    )


def test_regime_switch_shrinks_position_cap() -> None:
    dates = ("2020-08-03", "2020-08-04", "2020-08-05")
    regime = pd.Series(True, index=pd.to_datetime(dates))
    rows = [
        panel_row(day, oid, mid=50.0, spot=3000.0, deviation=0.03,
                  entry_short=True, strike=2500.0 + oid)
        for day in dates
        for oid in range(1, 31)
    ]
    curve, _ = run_backtest(pd.DataFrame(rows), make_params_table(),
                            high_regime=regime)
    # 10/day fills toward the high-vol cap of 20, then entries stop
    assert list(curve["n_positions"]) == [10, 20, HIGH_VOL_MAX_POSITIONS]


def test_transaction_costs_reduce_equity_on_flat_prices() -> None:
    # Prices never move and the trade exits by revert: the only equity
    # change is transaction costs, so final equity must be slightly below
    # start but within a few bps of it.
    panel = pd.DataFrame(
        [
            panel_row("2020-08-03", 1, mid=50.0, spot=3000.0, deviation=-0.03,
                      entry_long=True),
            panel_row("2020-08-04", 1, mid=50.0, spot=3000.0, deviation=0.0,
                      exit_signal=True),
        ]
    )
    curve, trades = run_backtest(panel, make_params_table())
    final = curve["equity"].iloc[-1]
    assert final < STARTING_CAPITAL
    assert final > STARTING_CAPITAL * 0.999
    assert trades.iloc[0]["pnl"] == pytest.approx(
        final - STARTING_CAPITAL, abs=1e-6
    )
