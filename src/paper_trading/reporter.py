"""Daily paper-trading email report for the Heston short-vol strategy.

Turns the persistent :class:`~src.paper_trading.portfolio.Portfolio` into a
plain-text daily summary — today's P&L, entries/exits, open book and the
cumulative performance metrics — with the equity curve attached as a PNG, and
sends it from and to the configured Gmail address over SMTP-SSL.

Performance metrics (Sharpe, Sortino, max drawdown, win rate) and the equity
plot are computed by reusing :mod:`src.heston.reporting` unchanged, applied to
the live equity curve and closed-trade records.

The Gmail app password is read at send time from ``~/.heston_secrets`` (a
``KEY=VALUE`` file); it is never stored in code. Report construction
(:func:`build_daily_message`) does no network I/O, so it is fully testable
offline — only :func:`send_message` / :func:`send_daily_report` touch SMTP.

Run:
    python -m src.paper_trading.reporter
"""

from __future__ import annotations

import logging
import math
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd

from src.heston.backtest import STARTING_CAPITAL
from src.heston.reporting import (
    annualised_sharpe,
    annualised_sortino,
    daily_returns,
    max_drawdown,
    write_equity_plot,
)
from src.heston.surface import REPORTS_DIR
from src.paper_trading.portfolio import Portfolio

logger = logging.getLogger("paper_trading.reporter")

# --- Secrets ----------------------------------------------------------------------
SECRETS_PATH: Path = Path.home() / ".heston_secrets"
GMAIL_APP_PASSWORD_KEY: str = "GMAIL_APP_PASSWORD"

# --- Email delivery ---------------------------------------------------------------
EMAIL_ADDRESS: str = "purujitaggarwal@gmail.com"  # sender and recipient
SMTP_HOST: str = "smtp.gmail.com"
SMTP_SSL_PORT: int = 465
SUBJECT_PREFIX: str = "Heston paper-trading"

# --- Report layout ----------------------------------------------------------------
LIVE_EQUITY_PNG: Path = REPORTS_DIR / "paper_equity_curve.png"
DATE_FORMAT: str = "%Y-%m-%d"
REPORT_RULE_WIDTH: int = 44
NOT_AVAILABLE: str = "n/a"


def read_secret(key: str, path: Path = SECRETS_PATH) -> str:
    """Read one value from a ``KEY=VALUE`` secrets file.

    Blank lines and ``#`` comments are ignored; surrounding whitespace and a
    single layer of matching quotes are stripped from the value.

    Args:
        key: Secret name to look up.
        path: Path to the secrets file.

    Returns:
        The secret value.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If ``key`` is not present in the file.
    """
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return value.strip().strip('"').strip("'")
    raise KeyError(f"{key} not found in {path}")


def _iso(date: pd.Timestamp) -> str:
    """Format a timestamp as a ``YYYY-MM-DD`` string."""
    return pd.Timestamp(date).strftime(DATE_FORMAT)


def _day_activity(portfolio: Portfolio, date_iso: str) -> tuple[int, int]:
    """Count positions opened and closed on a given date.

    Opened counts positions entered on the date whether they are still open or
    already closed; closed counts trades exited on the date.

    Args:
        portfolio: The book.
        date_iso: Trading date as ``YYYY-MM-DD``.

    Returns:
        ``(n_opened, n_closed)`` for the date.
    """
    opened = sum(
        1 for pos in portfolio.open_positions.values() if _iso(pos.entry_date) == date_iso
    )
    opened += sum(1 for trade in portfolio.closed_trades if trade["entry_date"] == date_iso)
    closed = sum(1 for trade in portfolio.closed_trades if trade["exit_date"] == date_iso)
    return opened, closed


def build_summary(portfolio: Portfolio, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    """Daily figures and cumulative performance metrics for the book.

    Metrics reuse :mod:`src.heston.reporting`: Sharpe/Sortino on daily equity
    returns against a zero benchmark, max drawdown on the equity curve, win
    rate and average holding period on the closed trades. Values that need at
    least two equity points (day P&L, ratios) or one closed trade (win rate,
    average holding) are NaN until enough history exists.

    Args:
        portfolio: The book to summarise.
        as_of: Report date; defaults to the latest equity-curve date.

    Returns:
        Summary dict consumed by :func:`format_summary`.
    """
    equity_curve = pd.DataFrame(portfolio.equity_curve)
    trades = pd.DataFrame(portfolio.closed_trades)

    if equity_curve.empty:
        equity = pd.Series([portfolio.equity], dtype=float)
        report_date = pd.Timestamp(as_of) if as_of is not None else None
    else:
        equity = equity_curve["equity"].astype(float)
        report_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(
            equity_curve["date"].iloc[-1]
        )

    equity_now = float(equity.iloc[-1])
    if len(equity) >= 2:
        day_pnl = float(equity.iloc[-1] - equity.iloc[-2])
    else:
        day_pnl = float(equity_now - STARTING_CAPITAL)

    date_iso = _iso(report_date) if report_date is not None else None
    n_opened, n_closed = _day_activity(portfolio, date_iso) if date_iso else (0, 0)
    returns = daily_returns(equity)

    return {
        "date": date_iso,
        "starting_capital": float(STARTING_CAPITAL),
        "equity": equity_now,
        "day_pnl": day_pnl,
        "total_return": float(equity_now / STARTING_CAPITAL - 1.0),
        "n_opened": n_opened,
        "n_closed": n_closed,
        "n_open_positions": len(portfolio.open_positions),
        "sharpe": annualised_sharpe(returns),
        "sortino": annualised_sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else float("nan"),
        "n_trades": int(len(trades)),
        "avg_holding_days": (
            float(trades["holding_days"].mean()) if not trades.empty else float("nan")
        ),
    }


def _fmt(value: float | None, spec: str) -> str:
    """Format a number, rendering None/NaN as :data:`NOT_AVAILABLE`."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NOT_AVAILABLE
    return format(value, spec)


def format_summary(summary: dict[str, Any]) -> str:
    """Render a summary dict as the plain-text email body.

    Args:
        summary: Output of :func:`build_summary`.

    Returns:
        Formatted multi-line report string.
    """
    return "\n".join(
        [
            f"{SUBJECT_PREFIX} — daily report",
            f"Date            : {summary['date'] or NOT_AVAILABLE}",
            "=" * REPORT_RULE_WIDTH,
            f"Equity          : ${_fmt(summary['equity'], '>15,.2f')}",
            f"Day P&L         : ${_fmt(summary['day_pnl'], '>+15,.2f')}",
            f"Total return    : {_fmt(summary['total_return'], '>16.2%')}",
            f"Opened / Closed : {summary['n_opened']:>8} / {summary['n_closed']}",
            f"Open positions  : {summary['n_open_positions']:>16}",
            "-" * REPORT_RULE_WIDTH,
            f"Sharpe (ann.)   : {_fmt(summary['sharpe'], '>16.2f')}",
            f"Sortino (ann.)  : {_fmt(summary['sortino'], '>16.2f')}",
            f"Max drawdown    : {_fmt(summary['max_drawdown'], '>16.1%')}",
            f"Win rate        : {_fmt(summary['win_rate'], '>16.1%')}",
            f"Trades (closed) : {summary['n_trades']:>16}",
            f"Avg holding     : {_fmt(summary['avg_holding_days'], '>11.1f')} days",
        ]
    )


def write_live_equity_png(portfolio: Portfolio, path: Path = LIVE_EQUITY_PNG) -> Path:
    """Write the live equity curve (with drawdown panel) to a PNG.

    Reuses :func:`src.heston.reporting.write_equity_plot` on the portfolio's
    recorded equity curve.

    Args:
        portfolio: Book whose ``equity_curve`` is plotted.
        path: Destination PNG path.

    Returns:
        The path written.
    """
    frame = pd.DataFrame(portfolio.equity_curve)
    frame = frame.assign(date=pd.to_datetime(frame["date"]))
    write_equity_plot(frame, path)
    return path


def build_daily_message(
    portfolio: Portfolio,
    as_of: pd.Timestamp | None = None,
    sender: str = EMAIL_ADDRESS,
    recipient: str = EMAIL_ADDRESS,
    attach_plot: bool = True,
    plot_path: Path = LIVE_EQUITY_PNG,
) -> EmailMessage:
    """Build the daily report email (no network I/O).

    Args:
        portfolio: The book to report on.
        as_of: Report date; defaults to the latest equity-curve date.
        sender: From address.
        recipient: To address.
        attach_plot: Attach the equity-curve PNG when the curve is non-empty.
        plot_path: Where the equity PNG is written before attaching.

    Returns:
        The assembled :class:`email.message.EmailMessage`.
    """
    summary = build_summary(portfolio, as_of)
    message = EmailMessage()
    message["Subject"] = f"{SUBJECT_PREFIX} — {summary['date'] or 'report'}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(format_summary(summary))

    if attach_plot and portfolio.equity_curve:
        png = write_live_equity_png(portfolio, plot_path)
        message.add_attachment(
            png.read_bytes(), maintype="image", subtype="png", filename=png.name
        )
    return message


def send_message(
    message: EmailMessage,
    password: str,
    host: str = SMTP_HOST,
    port: int = SMTP_SSL_PORT,
) -> None:
    """Send a prepared message over SMTP-SSL, authenticating as the sender.

    Args:
        message: The email to send (its ``From`` is the login user).
        password: Gmail app password.
        host: SMTP host.
        port: SMTP-SSL port.
    """
    with smtplib.SMTP_SSL(host, port) as server:
        server.login(message["From"], password)
        server.send_message(message)


def send_daily_report(
    portfolio: Portfolio,
    as_of: pd.Timestamp | None = None,
    password: str | None = None,
    sender: str = EMAIL_ADDRESS,
    recipient: str = EMAIL_ADDRESS,
) -> EmailMessage:
    """Build and send the daily report email.

    Args:
        portfolio: The book to report on.
        as_of: Report date; defaults to the latest equity-curve date.
        password: Gmail app password; read from :data:`SECRETS_PATH` when None.
        sender: From address.
        recipient: To address.

    Returns:
        The message that was sent.
    """
    message = build_daily_message(portfolio, as_of, sender, recipient)
    if password is None:
        password = read_secret(GMAIL_APP_PASSWORD_KEY)
    send_message(message, password)
    logger.info("sent daily report '%s' to %s", message["Subject"], recipient)
    return message


def main() -> None:
    """CLI entry point: ``python -m src.paper_trading.reporter``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    portfolio = Portfolio.load()
    message = send_daily_report(portfolio)
    print(f"sent '{message['Subject']}' to {message['To']}")


if __name__ == "__main__":
    main()
