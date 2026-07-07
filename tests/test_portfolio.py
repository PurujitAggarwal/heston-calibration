"""Tests for src.paper_trading.portfolio (pure state, no IBKR)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading.portfolio import (
    CONTRACT_MULTIPLIER,
    STARTING_CAPITAL,
    LivePosition,
    Portfolio,
)


def make_position(optionid: int = 1, cum_pnl: float = 0.0) -> LivePosition:
    """Build a short-vol position fixture (short 1 put contract)."""
    return LivePosition(
        optionid=optionid,
        direction=-1,
        units=1.0,
        allocation=1000.0,
        entry_date=pd.Timestamp("2026-01-15"),
        entry_mid=40.0,
        model_iv_entry=0.16,
        market_iv_entry=0.19,
        delta_entry=-0.30,
        entry_deviation=0.03,
        strike=4900.0,
        cp_flag="P",
        exdate=pd.Timestamp("2026-03-20"),
        last_mid=40.0,
        last_spot=5000.0,
        multiplier=CONTRACT_MULTIPLIER,
        cum_pnl=cum_pnl,
    )


def test_new_portfolio_starts_at_capital() -> None:
    book = Portfolio()
    assert book.cash == STARTING_CAPITAL
    assert book.equity == STARTING_CAPITAL
    assert book.unrealized_pnl == 0.0
    assert book.current_drawdown == 0.0
    assert book.open_positions == {}


def test_mark_position_moves_equity_not_cash() -> None:
    book = Portfolio()
    book.open_position(make_position())
    # short put: option falls 40 -> 38, profit = -1 * 1 * 100 * (38 - 40) = +200
    step = book.mark_position(1, mid=38.0, spot=5000.0)
    assert step == pytest.approx(200.0)
    assert book.unrealized_pnl == pytest.approx(200.0)
    assert book.equity == pytest.approx(STARTING_CAPITAL + 200.0)
    assert book.cash == pytest.approx(STARTING_CAPITAL)  # unchanged until close


def test_hedge_pnl_included_in_mark() -> None:
    book = Portfolio()
    book.open_position(make_position())
    book.set_hedge(1, hedge_shares=0.30)  # long 0.30 units of underlying
    # option flat, spot +10: hedge P&L = 0.30 * 10 = +3
    step = book.mark_position(1, mid=40.0, spot=5010.0)
    assert step == pytest.approx(3.0)


def test_apply_cost_reduces_equity() -> None:
    book = Portfolio()
    book.open_position(make_position())
    book.apply_cost(1, cost=12.5)
    assert book.equity == pytest.approx(STARTING_CAPITAL - 12.5)


def test_close_position_realizes_pnl_into_cash() -> None:
    book = Portfolio()
    book.open_position(make_position())
    book.mark_position(1, mid=38.0, spot=5000.0)  # +200 unrealized
    record = book.close_position(
        1, pd.Timestamp("2026-02-01"), "revert", exit_cost=5.0
    )
    assert record["pnl"] == pytest.approx(195.0)
    assert record["return_on_allocation"] == pytest.approx(195.0 / 1000.0)
    assert record["exit_reason"] == "revert"
    assert record["model_iv_entry"] == pytest.approx(0.16)
    assert book.open_positions == {}
    assert book.cash == pytest.approx(STARTING_CAPITAL + 195.0)
    assert book.unrealized_pnl == 0.0


def test_record_equity_tracks_peak_and_drawdown() -> None:
    book = Portfolio()
    book.record_equity(pd.Timestamp("2026-01-15"))  # 100k, peak 100k
    # book a 5k loss via a closed trade
    book.open_position(make_position())
    book.mark_position(1, mid=90.0, spot=5000.0)  # short put rises -> big loss
    book.close_position(1, pd.Timestamp("2026-01-16"), "stop")
    snap = book.record_equity(pd.Timestamp("2026-01-16"))
    assert snap["equity"] < STARTING_CAPITAL
    assert snap["peak_equity"] == pytest.approx(STARTING_CAPITAL)
    assert snap["drawdown"] == pytest.approx(book.equity / STARTING_CAPITAL - 1.0)
    assert snap["drawdown"] < 0.0


def test_open_duplicate_optionid_raises() -> None:
    book = Portfolio()
    book.open_position(make_position(optionid=7))
    with pytest.raises(ValueError, match="already open"):
        book.open_position(make_position(optionid=7))


def test_save_load_roundtrip(tmp_path) -> None:
    book = Portfolio()
    book.open_position(make_position(optionid=1))
    book.mark_position(1, mid=38.0, spot=5000.0)
    book.open_position(make_position(optionid=2))
    book.close_position(2, pd.Timestamp("2026-02-01"), "revert", exit_cost=1.0)
    book.record_equity(pd.Timestamp("2026-02-01"), spx_close=5010.0)

    path = book.save(tmp_path / "paper_portfolio.json")
    reloaded = Portfolio.load(path)
    assert reloaded.equity == pytest.approx(book.equity)
    assert reloaded.cash == pytest.approx(book.cash)
    assert set(reloaded.open_positions) == {1}
    assert len(reloaded.closed_trades) == 1
    assert reloaded.open_positions[1].cum_pnl == pytest.approx(
        book.open_positions[1].cum_pnl
    )
    assert reloaded.equity_curve[-1]["spx_close"] == pytest.approx(5010.0)


def test_load_or_new_when_missing(tmp_path) -> None:
    book = Portfolio.load_or_new(tmp_path / "does_not_exist.json")
    assert book.equity == STARTING_CAPITAL
    assert book.closed_trades == []
