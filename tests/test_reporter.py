"""Offline tests for src.paper_trading.reporter (no SMTP, no network).

Only report construction is exercised; send_message / send_daily_report are
never called, so no email is sent.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.paper_trading.portfolio import STARTING_CAPITAL, LivePosition, Portfolio
from src.paper_trading.reporter import (
    EMAIL_ADDRESS,
    GMAIL_APP_PASSWORD_KEY,
    SUBJECT_PREFIX,
    build_daily_message,
    build_summary,
    format_summary,
    read_secret,
)


def make_position(optionid: int, entry_date: str) -> LivePosition:
    """A short 1-contract put opened on ``entry_date``."""
    return LivePosition(
        optionid=optionid,
        direction=-1,
        units=1.0,
        allocation=1000.0,
        entry_date=pd.Timestamp(entry_date),
        entry_mid=5.0,
        model_iv_entry=0.17,
        market_iv_entry=0.20,
        delta_entry=-0.30,
        entry_deviation=0.03,
        strike=4900.0,
        cp_flag="P",
        exdate=pd.Timestamp("2026-09-18"),
        last_mid=5.0,
        last_spot=5000.0,
    )


def make_trade(entry_date: str, exit_date: str, pnl: float, holding_days: int) -> dict:
    """A closed-trade record with the fields the reporter reads."""
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "pnl": pnl,
        "holding_days": holding_days,
    }


def make_portfolio() -> Portfolio:
    """A book with two equity snapshots, two closed trades and one open position.

    Report date is 2026-07-06: one position was opened that day and is still
    open; one trade was opened and closed the same day; another trade closed
    that day but was opened earlier.
    """
    book = Portfolio()
    book.equity_curve = [
        {"date": "2026-07-03", "equity": 100_500.0},
        {"date": "2026-07-05", "equity": 101_000.0},
        {"date": "2026-07-06", "equity": 102_000.0},
    ]
    book.closed_trades = [
        make_trade("2026-06-01", "2026-07-06", 500.0, 35),  # closed today
        make_trade("2026-07-06", "2026-07-06", -120.0, 0),  # opened + closed today
        make_trade("2026-05-01", "2026-06-15", 300.0, 45),  # earlier
    ]
    book.open_positions = {7: make_position(7, "2026-07-06")}  # opened today, open
    return book


def test_read_secret_parses_value(tmp_path) -> None:
    secrets = tmp_path / "secrets"
    secrets.write_text(
        "# comment line\n"
        "\n"
        'GMAIL_APP_PASSWORD="abcd efgh ijkl mnop"\n'
        "OTHER=ignored\n"
    )
    assert read_secret(GMAIL_APP_PASSWORD_KEY, secrets) == "abcd efgh ijkl mnop"


def test_read_secret_missing_key_raises(tmp_path) -> None:
    secrets = tmp_path / "secrets"
    secrets.write_text("OTHER=value\n")
    with pytest.raises(KeyError):
        read_secret(GMAIL_APP_PASSWORD_KEY, secrets)


def test_build_summary_day_pnl_and_activity() -> None:
    summary = build_summary(make_portfolio())
    assert summary["date"] == "2026-07-06"
    assert summary["equity"] == pytest.approx(102_000.0)
    assert summary["day_pnl"] == pytest.approx(1_000.0)  # 102k - 101k
    assert summary["total_return"] == pytest.approx(102_000.0 / STARTING_CAPITAL - 1.0)
    assert summary["n_opened"] == 2  # open-today (1) + closed-that-entered-today (1)
    assert summary["n_closed"] == 2  # two trades exited today
    assert summary["n_open_positions"] == 1
    assert summary["n_trades"] == 3
    assert summary["win_rate"] == pytest.approx(2.0 / 3.0)  # two of three trades > 0
    assert not math.isnan(summary["sharpe"])  # two equity points -> defined


def test_build_summary_empty_book_is_nan_safe() -> None:
    summary = build_summary(Portfolio(), as_of=pd.Timestamp("2026-07-06"))
    assert summary["equity"] == pytest.approx(STARTING_CAPITAL)
    assert summary["day_pnl"] == pytest.approx(0.0)
    assert summary["n_opened"] == 0 and summary["n_closed"] == 0
    assert summary["n_trades"] == 0
    assert math.isnan(summary["win_rate"])
    assert math.isnan(summary["sharpe"])


def test_format_summary_contains_key_lines() -> None:
    body = format_summary(build_summary(make_portfolio()))
    assert "2026-07-06" in body
    assert "Day P&L" in body
    assert "Sharpe (ann.)" in body
    assert "Open positions" in body


def test_build_daily_message_attaches_png(tmp_path) -> None:
    message = build_daily_message(
        make_portfolio(), plot_path=tmp_path / "equity.png"
    )
    assert message["Subject"] == f"{SUBJECT_PREFIX} — 2026-07-06"
    assert message["From"] == EMAIL_ADDRESS
    assert message["To"] == EMAIL_ADDRESS
    attachments = list(message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_content_type() == "image/png"


def test_build_daily_message_no_plot_has_no_attachment() -> None:
    message = build_daily_message(make_portfolio(), attach_plot=False)
    assert list(message.iter_attachments()) == []
    assert message.get_content_type() == "text/plain"
