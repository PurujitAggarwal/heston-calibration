"""Offline tests for src.paper_trading.execution (no ib_insync, no TWS).

The end-to-end tests run execute_daily in paper-simulation mode (ib=None), so
fills book at the quoted mid and real Carr-Madan FFT deltas are computed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.heston.fft import HestonParams
from src.paper_trading.execution import (
    CALM_MAX_POSITIONS,
    EXIT_DTE_THRESHOLD,
    HIGH_VOL_MAX_POSITIONS,
    classify_exit,
    execute_daily,
    hedge_target_shares,
    regime_config,
    select_entries,
    size_contracts,
)
from src.paper_trading.portfolio import CONTRACT_MULTIPLIER, LivePosition, Portfolio

PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.03)
TODAY = pd.Timestamp("2026-01-15")


def make_position(optionid: int = 1, allocation: float = 1000.0) -> LivePosition:
    """A short 1-contract put position at entry."""
    return LivePosition(
        optionid=optionid,
        direction=-1,
        units=1.0,
        allocation=allocation,
        entry_date=TODAY,
        entry_mid=5.0,
        model_iv_entry=0.17,
        market_iv_entry=0.20,
        delta_entry=-0.30,
        entry_deviation=0.03,
        strike=4900.0,
        cp_flag="P",
        exdate=pd.Timestamp("2026-03-20"),
        last_mid=5.0,
        last_spot=5000.0,
        multiplier=CONTRACT_MULTIPLIER,
    )


def make_quotes(
    exit_signal: bool = False,
    entry_short: bool = True,
    days_to_expiry: int = 64,
    n: int = 2,
) -> pd.DataFrame:
    """A flagged, model-priced panel indexed by optionid (constant $5 mid)."""
    spot = 5000.0
    exdate = TODAY + pd.Timedelta(days=days_to_expiry)
    rows = []
    for i in range(n):
        strike = 4700.0 + 10.0 * i  # OTM puts below spot
        rows.append(
            {
                "optionid": 100 + i,
                "cp_flag": "P",
                "strike": strike,
                "spot": spot,
                "mid": 5.0,
                "market_iv": 0.20,
                "model_iv": 0.17,
                "deviation": 0.03 + 0.0001 * i,  # distinct, for signal ranking
                "maturity_years": days_to_expiry / 365.0,
                "rate": 0.045,
                "div_yield": 0.018,
                "days_to_expiry": days_to_expiry,
                "exit_signal": exit_signal,
                "entry_short": entry_short,
                "exdate": exdate,
            }
        )
    return pd.DataFrame(rows).set_index("optionid")


def test_size_contracts_floors_and_skips_unaffordable() -> None:
    assert size_contracts(1000.0, 5.0, 100.0) == 2  # 1000 / 500
    assert size_contracts(1000.0, 40.0, 100.0) == 0  # one contract = 4000 > alloc
    assert size_contracts(1000.0, 0.0, 100.0) == 0


def test_classify_exit_precedence() -> None:
    pos = make_position(allocation=1000.0)
    pos.cum_pnl = -60.0  # < -5% * 1000 = -50 -> stop wins
    assert classify_exit(pos, mid=5.0, days_to_expiry=64, exit_signal=True) == "stop"
    pos.cum_pnl = 0.0
    assert classify_exit(pos, mid=5.0, days_to_expiry=64, exit_signal=True) == "revert"
    assert (
        classify_exit(pos, mid=5.0, days_to_expiry=EXIT_DTE_THRESHOLD, exit_signal=False)
        == "expiry"
    )
    assert classify_exit(pos, mid=5.0, days_to_expiry=64, exit_signal=False) is None


def test_regime_config_and_hedge_sign() -> None:
    assert regime_config(False) == (0.01, 50)
    assert regime_config(True) == (0.005, 20)
    # short put (direction -1) with negative delta -> short the underlying hedge
    assert hedge_target_shares(-1, 2.0, 100.0, -0.30) == pytest.approx(-60.0)


def test_select_entries_respects_regime_position_cap() -> None:
    book = Portfolio()
    quotes = make_quotes(n=25)  # all affordable at the constant $5 mid
    deltas = {oid: -0.3 for oid in quotes.index}
    calm = select_entries(book, quotes, 100_000.0, deltas, is_high_vol=False)
    high = select_entries(book, quotes, 100_000.0, deltas, is_high_vol=True)
    assert len(calm) == 25  # under the calm cap of 50
    assert len(high) == HIGH_VOL_MAX_POSITIONS  # capped at 20
    assert all(p.units >= 1 for p in calm)


def test_select_entries_skips_below_min_mid() -> None:
    book = Portfolio()
    quotes = make_quotes(n=1)
    quotes.loc[100, "mid"] = 1.0  # below MIN_ENTRY_MID (2.0)
    plans = select_entries(book, quotes, 100_000.0, {100: -0.3}, is_high_vol=False)
    assert plans == []


def test_execute_daily_opens_then_closes_on_revert() -> None:
    book = Portfolio()
    day1 = execute_daily(book, make_quotes(entry_short=True), PARAMS, False, TODAY)
    assert day1["n_opened"] == 2
    assert len(book.open_positions) == 2
    assert book.equity < book.starting_capital  # entry + hedge costs
    for pos in book.open_positions.values():
        assert pos.market_iv_entry == pytest.approx(0.20)
        assert pos.hedge_shares != 0.0  # hedged at a nonzero model delta

    day2 = execute_daily(
        book,
        make_quotes(exit_signal=True, entry_short=False),
        PARAMS,
        False,
        TODAY + pd.Timedelta(days=1),
    )
    assert day2["n_closed"] == 2
    assert book.open_positions == {}
    assert len(book.closed_trades) == 2
    assert all(t["exit_reason"] == "revert" for t in book.closed_trades)


def test_execute_daily_closes_delisted_when_quote_missing() -> None:
    book = Portfolio()
    execute_daily(book, make_quotes(entry_short=True), PARAMS, False, TODAY)
    assert len(book.open_positions) == 2

    quotes = make_quotes(entry_short=False, n=2).drop(index=101)
    execute_daily(book, quotes, PARAMS, False, TODAY + pd.Timedelta(days=1))
    reasons = {t["optionid"]: t["exit_reason"] for t in book.closed_trades}
    assert reasons.get(101) == "delisted"
    assert 100 in book.open_positions  # still quoted, no exit rule hit


def test_high_vol_uses_smaller_size_than_calm() -> None:
    quotes = make_quotes(n=2)
    calm_book, high_book = Portfolio(), Portfolio()
    execute_daily(calm_book, quotes, PARAMS, False, TODAY)
    execute_daily(high_book, quotes.copy(), PARAMS, True, TODAY)
    calm_units = sum(p.units for p in calm_book.open_positions.values())
    high_units = sum(p.units for p in high_book.open_positions.values())
    assert calm_units > high_units  # 1% vs 0.5% allocation -> fewer contracts
    assert CALM_MAX_POSITIONS == 50
