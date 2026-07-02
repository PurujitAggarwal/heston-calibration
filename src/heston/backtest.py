"""Delta-hedged backtest of the Heston IV mean-reversion strategy.

Consumes the flagged signal panel and simulates from $100,000 (production
configuration: regime-adaptive short-vol only):

    - entries: strongest liquid deviations first, at most
      MAX_NEW_POSITIONS_PER_DAY per day; short the option when market IV is
      rich vs the model (the long-vol leg is disabled in the signals stage)
    - regime: 21-day realised vol of SPX closes, lagged one day, vs the
      80th percentile of its trailing 252-day distribution — high-vol days
      run half sizing and a smaller book (no lookahead: only information
      through the prior close feeds the classifier)
    - sizing: inverse vol — premium notional = equity * base_fraction *
      (20% / market IV), capped at 5% of equity, where base_fraction and
      the concurrent-position cap follow the regime (calm: 1% / 50 names;
      high vol: 0.5% / 20 names); equity is marked to market daily
    - hedging: delta-neutral versus the Heston model delta, rebalanced daily
    - costs: 3bps per side on every option and hedge notional traded
    - exits: deviation reverted within 0.5 vol pts, stop loss at -5% of the
      capital allocated to the position, approaching expiry, or the quote
      leaving the tradable universe (closed at its last mark)
    - guards: entries require mid >= $2; units are scaled down so the
      underlying-equivalent hedge notional stays within 10x the allocation;
      a contract that stops out cannot re-enter for 5 trading days

Outputs data/processed/equity_curve.parquet and trades.parquet for the
reporting stage.

Run:
    python -m src.heston.backtest
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.heston.calibration import PARAMS_PARQUET
from src.heston.data import PROJECT_ROOT, SPOT_PARQUET
from src.heston.fft import HestonParams, heston_deltas_bulk
from src.heston.signals import PANEL_PARQUET
from src.heston.surface import params_for_date

# --- Account ----------------------------------------------------------------------
STARTING_CAPITAL: float = 100_000.0
TRANSACTION_COST_RATE: float = 3e-4  # 3 bps per side on traded notional

# --- Position sizing (regime-adaptive) ---------------------------------------------
SIZING_REF_VOL: float = 0.20
MAX_ALLOCATION_FRACTION: float = 0.05
CALM_ALLOCATION_FRACTION: float = 0.01  # base sizing in the calm regime
CALM_MAX_POSITIONS: int = 50
HIGH_VOL_ALLOCATION_FRACTION: float = 0.005  # half sizing in the high-vol regime
HIGH_VOL_MAX_POSITIONS: int = 20

# --- Regime classifier --------------------------------------------------------------
REALISED_VOL_WINDOW: int = 21  # trading days of returns per vol estimate
REALISED_VOL_ANNUALISATION: float = 252.0
REGIME_LAG_DAYS: int = 1  # only information through the prior close is used
REGIME_LOOKBACK: int = 252  # trailing distribution window
REGIME_MIN_PERIODS: int = 126  # allow classification once half a window exists
REGIME_PERCENTILE: float = 0.80  # above this trailing percentile = high vol

# --- Risk limits ------------------------------------------------------------------
STOP_LOSS_FRACTION: float = 0.05  # of the capital allocated at entry
MAX_NEW_POSITIONS_PER_DAY: int = 10
MIN_EXIT_DTE: int = 8  # close before the quote drops out of the 7-day floor
MIN_ENTRY_MID: float = 2.0  # no entries in sub-$2 wing options
MAX_UNDERLYING_LEVERAGE: float = 10.0  # |delta| * units * spot <= 10x allocation
STOP_COOLDOWN_DAYS: int = 5  # trading days a stopped contract stays barred

# --- Output -----------------------------------------------------------------------
EQUITY_CURVE_PARQUET = PROJECT_ROOT / "data" / "processed" / "equity_curve.parquet"
TRADES_PARQUET = PROJECT_ROOT / "data" / "processed" / "trades.parquet"


@dataclass
class Position:
    """One open delta-hedged option position.

    Attributes:
        optionid: OptionMetrics contract id.
        direction: +1 long the option (long vol), -1 short (short vol).
        units: Option units held (premium notional / entry mid).
        allocation: Equity allocated at entry (stop-loss base).
        entry_date: Entry trading date.
        entry_deviation: Market-minus-model IV at entry.
        strike: Strike price.
        cp_flag: "C" or "P".
        exdate: Contract expiry date.
        last_mid: Latest option mark.
        last_spot: Latest index level.
        hedge_shares: Signed underlying holding (delta hedge).
        cum_pnl: Cumulative P&L including hedge and costs.
    """

    optionid: int
    direction: int
    units: float
    allocation: float
    entry_date: pd.Timestamp
    entry_deviation: float
    strike: float
    cp_flag: str
    exdate: pd.Timestamp
    last_mid: float
    last_spot: float
    hedge_shares: float = 0.0
    cum_pnl: float = field(default=0.0)


def high_vol_regime(spot: pd.DataFrame) -> pd.Series:
    """Daily high-vol regime flags from SPX closes, with no lookahead.

    Realised vol is the 21-day rolling standard deviation of log returns,
    annualised. It is lagged REGIME_LAG_DAYS so the flag for day t uses
    only information through day t-1, and compared against the
    REGIME_PERCENTILE quantile of its own trailing REGIME_LOOKBACK-day
    distribution (built from the same lagged series).

    Args:
        spot: Frame with ``date`` and ``close`` columns.

    Returns:
        Boolean Series indexed by date; True on high-vol days. Days without
        enough history are classified calm.
    """
    closes = spot.sort_values("date").set_index("date")["close"]
    returns = np.log(closes).diff()
    realised = returns.rolling(REALISED_VOL_WINDOW).std() * np.sqrt(
        REALISED_VOL_ANNUALISATION
    )
    lagged = realised.shift(REGIME_LAG_DAYS)
    threshold = lagged.rolling(
        REGIME_LOOKBACK, min_periods=REGIME_MIN_PERIODS
    ).quantile(REGIME_PERCENTILE)
    return (lagged > threshold).fillna(False)


def size_position(equity: float, market_iv: float, base_fraction: float) -> float:
    """Inverse-vol premium allocation for one position.

    Args:
        equity: Current marked portfolio equity.
        market_iv: Market implied vol of the option being entered.
        base_fraction: Regime base allocation fraction
            (CALM_ALLOCATION_FRACTION or HIGH_VOL_ALLOCATION_FRACTION).

    Returns:
        Dollar premium notional to deploy.
    """
    fraction = base_fraction * (SIZING_REF_VOL / market_iv)
    return equity * min(fraction, MAX_ALLOCATION_FRACTION)


def trade_cost(notional: float) -> float:
    """Transaction cost for one side of a trade.

    Args:
        notional: Absolute dollar notional traded.

    Returns:
        Cost in dollars.
    """
    return abs(notional) * TRANSACTION_COST_RATE


def compute_deltas(
    params: HestonParams, day: pd.DataFrame, positions: dict[int, Position]
) -> dict[int, float]:
    """Heston deltas for every open position quoted today.

    Groups positions by expiry so each (date, expiry) costs two FFTs.

    Args:
        params: Parameters governing the current quarter.
        day: Today's panel slice indexed by optionid.
        positions: Open positions.

    Returns:
        Mapping optionid -> model delta.
    """
    quoted = [oid for oid in positions if oid in day.index]
    deltas: dict[int, float] = {}
    if not quoted:
        return deltas
    rows = day.loc[quoted]
    for _, group in rows.groupby("maturity_years"):
        values = heston_deltas_bulk(
            params,
            float(group["spot"].iloc[0]),
            group["strike"].to_numpy(float),
            float(group["maturity_years"].iloc[0]),
            float(group["rate"].iloc[0]),
            float(group["div_yield"].iloc[0]),
            group["cp_flag"].to_numpy(),
        )
        for oid, delta in zip(group.index, values):
            deltas[int(oid)] = float(delta)
    return deltas


def close_position(
    position: Position,
    exit_date: pd.Timestamp,
    exit_reason: str,
    trades: list[dict[str, object]],
) -> float:
    """Close a position at its latest marks and log the trade.

    Args:
        position: Position to close (already marked to today).
        exit_date: Closing date.
        exit_reason: One of revert/stop/expiry/delisted/final.
        trades: Trade log to append to.

    Returns:
        Exit transaction costs (option leg + hedge unwind).
    """
    costs = trade_cost(position.units * position.last_mid) + trade_cost(
        position.hedge_shares * position.last_spot
    )
    position.cum_pnl -= costs
    trades.append(
        {
            "optionid": position.optionid,
            "units": position.units,
            "direction": position.direction,
            "entry_date": position.entry_date,
            "exit_date": exit_date,
            "holding_days": int((exit_date - position.entry_date).days),
            "strike": position.strike,
            "cp_flag": position.cp_flag,
            "exdate": position.exdate,
            "allocation": position.allocation,
            "entry_deviation": position.entry_deviation,
            "pnl": position.cum_pnl,
            "return_on_allocation": position.cum_pnl / position.allocation,
            "exit_reason": exit_reason,
        }
    )
    return costs


def run_backtest(
    panel: pd.DataFrame,
    params_table: pd.DataFrame,
    high_regime: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the strategy over the full flagged signal panel.

    Args:
        panel: Flagged signal panel (signals stage output).
        params_table: Rolling calibration output (for hedge deltas).
        high_regime: Boolean Series (indexed by date) marking high-vol days,
            from :func:`high_vol_regime`. Days absent from the series — and
            every day when None is passed (synthetic tests) — are calm.

    Returns:
        Tuple ``(equity_curve, trades)``: daily equity frame and one row
        per closed trade.
    """
    panel = panel.sort_values("date")
    panel = panel.assign(optionid=panel["optionid"].astype("int64"))
    equity = STARTING_CAPITAL
    positions: dict[int, Position] = {}
    trades: list[dict[str, object]] = []
    curve: list[dict[str, object]] = []
    last_stop_day: dict[int, int] = {}

    current_quarter: pd.Timestamp | None = None
    params: HestonParams | None = None

    for day_index, (date, day) in enumerate(panel.groupby("date", sort=True)):
        day = day.set_index("optionid")
        day = day[~day.index.duplicated(keep="first")]
        quarter = pd.Timestamp(date).to_period("Q").start_time
        if quarter != current_quarter:
            params = params_for_date(params_table, pd.Timestamp(date))
            current_quarter = quarter
        if params is None:
            continue
        is_high_vol = high_regime is not None and bool(
            high_regime.get(pd.Timestamp(date), False)
        )
        base_fraction = (
            HIGH_VOL_ALLOCATION_FRACTION if is_high_vol else CALM_ALLOCATION_FRACTION
        )
        max_positions = HIGH_VOL_MAX_POSITIONS if is_high_vol else CALM_MAX_POSITIONS
        costs_today = 0.0
        pnl_today = 0.0

        # --- mark open positions and process exits -------------------------
        for oid in list(positions):
            position = positions[oid]
            if oid not in day.index:
                costs_today += close_position(
                    position, pd.Timestamp(date), "delisted", trades
                )
                del positions[oid]
                continue
            row = day.loc[oid]
            mid, spot = float(row["mid"]), float(row["spot"])
            step_pnl = position.direction * position.units * (
                mid - position.last_mid
            ) + position.hedge_shares * (spot - position.last_spot)
            position.cum_pnl += step_pnl
            pnl_today += step_pnl
            position.last_mid, position.last_spot = mid, spot

            if position.cum_pnl <= -STOP_LOSS_FRACTION * position.allocation:
                reason = "stop"
            elif bool(row["exit_signal"]):
                reason = "revert"
            elif int(row["days_to_expiry"]) <= MIN_EXIT_DTE:
                reason = "expiry"
            else:
                reason = ""
            if reason:
                if reason == "stop":
                    last_stop_day[oid] = day_index
                costs_today += close_position(
                    position, pd.Timestamp(date), reason, trades
                )
                del positions[oid]

        # --- rehedge survivors to the model delta --------------------------
        deltas = compute_deltas(params, day, positions)
        for oid, delta in deltas.items():
            position = positions[oid]
            target = -position.direction * position.units * delta
            rehedge_cost = trade_cost(
                (target - position.hedge_shares) * position.last_spot
            )
            costs_today += rehedge_cost
            position.cum_pnl -= rehedge_cost
            position.hedge_shares = target

        # --- entries --------------------------------------------------------
        candidates = day.loc[
            (day["entry_long"] | day["entry_short"])
            & ~day.index.isin(positions.keys())
        ].copy()
        candidates = candidates.reindex(
            candidates["deviation"].abs().sort_values(ascending=False).index
        )
        n_new = 0
        equity_marked = equity + pnl_today - costs_today
        new_ids: list[int] = []
        for oid, row in candidates.iterrows():
            if (
                n_new >= MAX_NEW_POSITIONS_PER_DAY
                or len(positions) >= max_positions
            ):
                break
            mid = float(row["mid"])
            if mid < MIN_ENTRY_MID or equity_marked <= 0.0:
                continue
            stopped = last_stop_day.get(int(oid))
            if stopped is not None and day_index - stopped <= STOP_COOLDOWN_DAYS:
                continue
            allocation = size_position(
                equity_marked, float(row["market_iv"]), base_fraction
            )
            direction = 1 if bool(row["entry_long"]) else -1
            positions[int(oid)] = Position(
                optionid=int(oid),
                direction=direction,
                units=allocation / mid,
                allocation=allocation,
                entry_date=pd.Timestamp(date),
                entry_deviation=float(row["deviation"]),
                strike=float(row["strike"]),
                cp_flag=str(row["cp_flag"]),
                exdate=pd.Timestamp(row["exdate"]),
                last_mid=mid,
                last_spot=float(row["spot"]),
            )
            new_ids.append(int(oid))
            n_new += 1

        # size-cap, cost and hedge the new entries at today's deltas
        new_deltas = compute_deltas(params, day, {oid: positions[oid] for oid in new_ids})
        for oid in new_ids:
            position = positions[oid]
            delta = new_deltas.get(oid, 0.0)
            hedge_notional = abs(delta) * position.units * position.last_spot
            leverage_cap = MAX_UNDERLYING_LEVERAGE * position.allocation
            if hedge_notional > leverage_cap:
                position.units *= leverage_cap / hedge_notional
            entry_cost = trade_cost(position.units * position.last_mid)
            position.hedge_shares = -position.direction * position.units * delta
            hedge_cost = trade_cost(position.hedge_shares * position.last_spot)
            costs_today += entry_cost + hedge_cost
            position.cum_pnl -= entry_cost + hedge_cost

        equity += pnl_today - costs_today
        curve.append(
            {
                "date": pd.Timestamp(date),
                "equity": equity,
                "n_positions": len(positions),
                "pnl": pnl_today,
                "costs": costs_today,
            }
        )

    # close whatever is still open at the end of the sample
    final_costs = 0.0
    if curve:
        last_date = pd.Timestamp(curve[-1]["date"])
        for oid in list(positions):
            final_costs += close_position(positions[oid], last_date, "final", trades)
            del positions[oid]
        equity -= final_costs
        curve[-1]["equity"] = equity
        curve[-1]["costs"] = float(curve[-1]["costs"]) + final_costs

    return pd.DataFrame(curve), pd.DataFrame(trades)


def main() -> None:
    """CLI entry point: ``python -m src.heston.backtest``."""
    panel = pd.read_parquet(PANEL_PARQUET)
    params_table = pd.read_parquet(PARAMS_PARQUET)
    regime = high_vol_regime(pd.read_parquet(SPOT_PARQUET))
    print(f"high-vol regime share: {regime.mean():.1%} of days")
    equity_curve, trades = run_backtest(panel, params_table, high_regime=regime)

    EQUITY_CURVE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    equity_curve.to_parquet(EQUITY_CURVE_PARQUET, index=False)
    trades.to_parquet(TRADES_PARQUET, index=False)

    final = float(equity_curve["equity"].iloc[-1])
    print(f"\nfinal equity: ${final:,.0f} from ${STARTING_CAPITAL:,.0f}")
    print(f"closed trades: {len(trades):,}")
    print(trades["exit_reason"].value_counts().to_string())
    print(f"wrote {EQUITY_CURVE_PARQUET} and {TRADES_PARQUET}")


if __name__ == "__main__":
    main()
