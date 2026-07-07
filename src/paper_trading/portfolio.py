"""Paper-trading portfolio state — cash, open positions, closed trades, equity.

Maintains the book in ``data/processed/paper_portfolio.json`` and is the single
source of truth for P&L accounting. The model mirrors the historical backtest:
every position carries a marked-to-market ``cum_pnl`` (option leg + delta-hedge
leg + all transaction costs incurred so far), and

    equity   = starting_capital + realized_pnl + unrealized_pnl
    cash     = starting_capital + realized_pnl          (settled, closed trades)
    unrealized_pnl = sum(cum_pnl over open positions)
    realized_pnl   = sum(pnl over closed trades)

so entry/exit costs and daily hedge costs hit equity immediately (through a
position's ``cum_pnl``) and roll into cash only when the trade closes.

Position P&L on a mark:
    step = direction * units * multiplier * (mid - last_mid)
           + hedge_shares * (spot - last_spot)

where ``spot`` is the underlying price used for delta hedging (SPX or its
proxy), kept consistent across marks by the execution layer.

All dates are stored as ``YYYY-MM-DD`` strings so the JSON is human-readable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.heston.backtest import STARTING_CAPITAL
from src.heston.data import PROJECT_ROOT

logger = logging.getLogger("paper_trading.portfolio")

# --- Constants --------------------------------------------------------------------
PORTFOLIO_PATH: Path = PROJECT_ROOT / "data" / "processed" / "paper_portfolio.json"
CONTRACT_MULTIPLIER: float = 100.0  # SPX option contract multiplier
DATE_FORMAT: str = "%Y-%m-%d"


def _iso(date: pd.Timestamp) -> str:
    """Format a timestamp as a ``YYYY-MM-DD`` string."""
    return pd.Timestamp(date).strftime(DATE_FORMAT)


@dataclass
class LivePosition:
    """One open delta-hedged paper option position.

    Attributes:
        optionid: IBKR contract id (unique key).
        direction: +1 long the option, -1 short (short vol).
        units: Number of option contracts held.
        multiplier: Contract multiplier (100 for SPX).
        allocation: Equity allocated at entry (stop-loss base).
        entry_date: Entry trading date.
        entry_mid: Option mid price paid/received at entry.
        model_iv_entry: Heston model IV at entry.
        market_iv_entry: Market IV at entry.
        delta_entry: Model delta at entry.
        entry_deviation: market_iv_entry - model_iv_entry at entry.
        strike: Strike price.
        cp_flag: "C" or "P".
        exdate: Contract expiry date.
        hedge_shares: Signed underlying holding hedging the option delta.
        last_mid: Latest option mark.
        last_spot: Latest underlying (hedge) price.
        cum_pnl: Marked P&L including hedge and all costs to date.
    """

    optionid: int
    direction: int
    units: float
    allocation: float
    entry_date: pd.Timestamp
    entry_mid: float
    model_iv_entry: float
    market_iv_entry: float
    delta_entry: float
    entry_deviation: float
    strike: float
    cp_flag: str
    exdate: pd.Timestamp
    last_mid: float
    last_spot: float
    multiplier: float = CONTRACT_MULTIPLIER
    hedge_shares: float = 0.0
    cum_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (timestamps as ISO strings)."""
        return {
            "optionid": int(self.optionid),
            "direction": int(self.direction),
            "units": float(self.units),
            "allocation": float(self.allocation),
            "entry_date": _iso(self.entry_date),
            "entry_mid": float(self.entry_mid),
            "model_iv_entry": float(self.model_iv_entry),
            "market_iv_entry": float(self.market_iv_entry),
            "delta_entry": float(self.delta_entry),
            "entry_deviation": float(self.entry_deviation),
            "strike": float(self.strike),
            "cp_flag": str(self.cp_flag),
            "exdate": _iso(self.exdate),
            "last_mid": float(self.last_mid),
            "last_spot": float(self.last_spot),
            "multiplier": float(self.multiplier),
            "hedge_shares": float(self.hedge_shares),
            "cum_pnl": float(self.cum_pnl),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LivePosition":
        """Rebuild a position from :meth:`to_dict` output."""
        return cls(
            optionid=int(data["optionid"]),
            direction=int(data["direction"]),
            units=float(data["units"]),
            allocation=float(data["allocation"]),
            entry_date=pd.Timestamp(data["entry_date"]),
            entry_mid=float(data["entry_mid"]),
            model_iv_entry=float(data["model_iv_entry"]),
            market_iv_entry=float(data["market_iv_entry"]),
            delta_entry=float(data["delta_entry"]),
            entry_deviation=float(data["entry_deviation"]),
            strike=float(data["strike"]),
            cp_flag=str(data["cp_flag"]),
            exdate=pd.Timestamp(data["exdate"]),
            last_mid=float(data["last_mid"]),
            last_spot=float(data["last_spot"]),
            multiplier=float(data["multiplier"]),
            hedge_shares=float(data["hedge_shares"]),
            cum_pnl=float(data["cum_pnl"]),
        )


@dataclass
class Portfolio:
    """The paper-trading book: cash, open positions, closed trades, equity curve.

    Attributes:
        starting_capital: Initial capital ($100,000).
        open_positions: Open positions keyed by optionid.
        closed_trades: Closed-trade records (P&L per trade).
        equity_curve: Daily equity snapshots.
    """

    starting_capital: float = STARTING_CAPITAL
    open_positions: dict[int, LivePosition] = field(default_factory=dict)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    # --- derived quantities -------------------------------------------------------
    @property
    def realized_pnl(self) -> float:
        """P&L already booked from closed trades."""
        return float(sum(trade["pnl"] for trade in self.closed_trades))

    @property
    def unrealized_pnl(self) -> float:
        """Marked P&L across all open positions."""
        return float(sum(pos.cum_pnl for pos in self.open_positions.values()))

    @property
    def cash(self) -> float:
        """Settled cash: starting capital plus realized P&L."""
        return self.starting_capital + self.realized_pnl

    @property
    def equity(self) -> float:
        """Total marked equity: cash plus unrealized P&L."""
        return self.cash + self.unrealized_pnl

    @property
    def peak_equity(self) -> float:
        """Highest equity seen across the recorded curve and the current mark."""
        recorded = [float(snap["equity"]) for snap in self.equity_curve]
        return max([*recorded, self.equity, self.starting_capital])

    @property
    def current_drawdown(self) -> float:
        """Current equity drawdown from the running peak (<= 0)."""
        peak = self.peak_equity
        return self.equity / peak - 1.0 if peak > 0.0 else 0.0

    # --- mutations ----------------------------------------------------------------
    def open_position(self, position: LivePosition) -> None:
        """Add a new open position (its ``cum_pnl`` already nets entry costs).

        Args:
            position: The position to open.

        Raises:
            ValueError: If a position with the same optionid is already open.
        """
        if position.optionid in self.open_positions:
            raise ValueError(f"position {position.optionid} is already open")
        self.open_positions[position.optionid] = position
        logger.info(
            "opened %s %d x %.0f %s (optionid=%d)",
            "long" if position.direction > 0 else "short",
            position.units,
            position.strike,
            position.cp_flag,
            position.optionid,
        )

    def mark_position(self, optionid: int, mid: float, spot: float) -> float:
        """Mark an open position to a new option mid and underlying price.

        Args:
            optionid: Position key.
            mid: Latest option mid price.
            spot: Latest underlying (hedge) price.

        Returns:
            The step P&L added to the position's cumulative P&L.
        """
        position = self.open_positions[optionid]
        step = position.direction * position.units * position.multiplier * (
            mid - position.last_mid
        ) + position.hedge_shares * (spot - position.last_spot)
        position.cum_pnl += step
        position.last_mid = mid
        position.last_spot = spot
        return float(step)

    def apply_cost(self, optionid: int, cost: float) -> None:
        """Charge a transaction cost against an open position's P&L.

        Args:
            optionid: Position key.
            cost: Dollar cost (non-negative); subtracted from ``cum_pnl``.
        """
        self.open_positions[optionid].cum_pnl -= abs(cost)

    def set_hedge(self, optionid: int, hedge_shares: float) -> None:
        """Set an open position's underlying hedge holding.

        Args:
            optionid: Position key.
            hedge_shares: New signed underlying holding.
        """
        self.open_positions[optionid].hedge_shares = hedge_shares

    def close_position(
        self,
        optionid: int,
        exit_date: pd.Timestamp,
        exit_reason: str,
        exit_cost: float = 0.0,
    ) -> dict[str, Any]:
        """Close a position, charging any exit cost, and record the trade.

        Args:
            optionid: Position key.
            exit_date: Closing date.
            exit_reason: One of revert/stop/expiry/delisted/final.
            exit_cost: Transaction cost of the exit (option + hedge unwind).

        Returns:
            The closed-trade record appended to ``closed_trades``.
        """
        position = self.open_positions.pop(optionid)
        position.cum_pnl -= abs(exit_cost)
        record = {
            "optionid": int(position.optionid),
            "direction": int(position.direction),
            "units": float(position.units),
            "multiplier": float(position.multiplier),
            "allocation": float(position.allocation),
            "entry_date": _iso(position.entry_date),
            "exit_date": _iso(exit_date),
            "holding_days": int(
                (pd.Timestamp(exit_date) - position.entry_date).days
            ),
            "strike": float(position.strike),
            "cp_flag": str(position.cp_flag),
            "exdate": _iso(position.exdate),
            "model_iv_entry": float(position.model_iv_entry),
            "market_iv_entry": float(position.market_iv_entry),
            "delta_entry": float(position.delta_entry),
            "entry_deviation": float(position.entry_deviation),
            "pnl": float(position.cum_pnl),
            "return_on_allocation": (
                float(position.cum_pnl / position.allocation)
                if position.allocation
                else float("nan")
            ),
            "exit_reason": exit_reason,
        }
        self.closed_trades.append(record)
        logger.info(
            "closed optionid=%d reason=%s pnl=%.2f",
            position.optionid,
            exit_reason,
            position.cum_pnl,
        )
        return record

    def record_equity(
        self, date: pd.Timestamp, spx_close: float | None = None
    ) -> dict[str, Any]:
        """Append a daily equity snapshot and return it.

        Args:
            date: Trading date of the snapshot.
            spx_close: SPX close for the day (optional, for reporting).

        Returns:
            The snapshot appended to ``equity_curve``.
        """
        snapshot = {
            "date": _iso(date),
            "equity": self.equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "n_positions": len(self.open_positions),
            "peak_equity": self.peak_equity,
            "drawdown": self.current_drawdown,
            "spx_close": float(spx_close) if spx_close is not None else None,
        }
        self.equity_curve.append(snapshot)
        return snapshot

    # --- persistence --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole book to a JSON-safe dict."""
        return {
            "starting_capital": float(self.starting_capital),
            "open_positions": {
                str(oid): pos.to_dict() for oid, pos in self.open_positions.items()
            },
            "closed_trades": self.closed_trades,
            "equity_curve": self.equity_curve,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        """Rebuild a portfolio from :meth:`to_dict` output."""
        return cls(
            starting_capital=float(data["starting_capital"]),
            open_positions={
                int(oid): LivePosition.from_dict(pos)
                for oid, pos in data["open_positions"].items()
            },
            closed_trades=list(data["closed_trades"]),
            equity_curve=list(data["equity_curve"]),
        )

    def save(self, path: Path = PORTFOLIO_PATH) -> Path:
        """Write the book to JSON, stamping the update time in the payload.

        Args:
            path: Destination JSON path.

        Returns:
            The path written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["last_updated"] = (
            self.equity_curve[-1]["date"] if self.equity_curve else None
        )
        path.write_text(json.dumps(payload, indent=2))
        logger.info("saved portfolio to %s (equity=%.2f)", path, self.equity)
        return path

    @classmethod
    def load(cls, path: Path = PORTFOLIO_PATH) -> "Portfolio":
        """Load a portfolio from JSON.

        Args:
            path: Source JSON path.

        Returns:
            The reconstructed portfolio.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def load_or_new(cls, path: Path = PORTFOLIO_PATH) -> "Portfolio":
        """Load the portfolio if it exists, else start a fresh one at capital.

        Args:
            path: Source JSON path.

        Returns:
            The loaded or freshly initialised portfolio.
        """
        if path.exists():
            return cls.load(path)
        logger.info("no portfolio at %s; starting fresh at $%.0f", path, STARTING_CAPITAL)
        return cls()
