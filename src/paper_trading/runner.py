"""Autonomous daily runner — the cron entry point for live paper trading.

Wires a live Interactive Brokers trading connection into the daily orchestrator
(:func:`src.paper_trading.daily.run_day`) and drives one full trading day. This
is the process the cron job (``setup_cron.sh``) invokes once per weekday: it
opens the read/write IBKR connection to the paper account, runs the whole
fetch -> price -> signal -> execute -> report cycle with orders routed to IBKR,
emails the daily report, and always disconnects cleanly.

Passing the connected ``ib`` into :func:`run_day` is what makes the day *live*:
with it, entries/exits are routed as limit-at-mid orders to the paper account
and the net delta is hedged in SPY; without it (as in the offline tests) fills
are booked at the quoted mid. The market-data pull inside :func:`run_day` opens
its own separate read-only connection, so this runner owns only the trading
session.

Prerequisites (the runner starts none of them):
    - TWS or IB Gateway running and logged into paper account DUR195917, with
      the API enabled on 127.0.0.1:7497;
    - a Gmail app password in ``~/.heston_secrets`` for the email report.

Logs go to ``logs/paper_trading.log`` and to stderr, which the cron job
captures in ``logs/cron.log``.

Run:
    python -m src.paper_trading.runner
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.heston.data import PROJECT_ROOT
from src.paper_trading.daily import run_day
from src.paper_trading.execution import connect_trading

logger = logging.getLogger("paper_trading.runner")

# --- Logging ----------------------------------------------------------------------
LOG_DIR: Path = PROJECT_ROOT / "logs"
RUNNER_LOG: Path = LOG_DIR / "paper_trading.log"
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_path: Path = RUNNER_LOG) -> None:
    """Send INFO-level logs to both the runner log file and stderr.

    Args:
        log_path: Append-mode log file; its parent directory is created.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
        ],
    )


def main() -> int:
    """Run one live paper-trading day end to end against the IBKR paper account.

    Opens the read/write IBKR trading connection, drives :func:`run_day` with it
    (routing orders and hedging live) and emailing the report, then disconnects.

    Returns:
        Process exit code (0 on success).
    """
    configure_logging()
    logger.info("paper-trading runner starting (IBKR paper account)")
    ib = connect_trading()
    try:
        result = run_day(ib=ib, send=True)
    finally:
        ib.disconnect()
        logger.info("disconnected from IBKR")
    print(
        f"{result['date']:%Y-%m-%d}: opened {result['n_opened']}, "
        f"closed {result['n_closed']}, equity ${result['equity']:,.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
