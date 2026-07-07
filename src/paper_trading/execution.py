"""Trade execution — turn Heston short-vol signals into IBKR paper trades.

Consumes a flagged, model-priced quote panel for one trading day (indexed by
optionid) and drives the :class:`~src.paper_trading.portfolio.Portfolio`:

    1. mark every open position to today's mid and index level;
    2. close positions on revert / -5% stop / <=30 days to expiry / delisting;
    3. re-hedge survivors to the Heston model delta;
    4. open new short-vol positions on the strongest liquid signals, sized to
       whole contracts under the regime-adaptive production config.

Reused, unchanged, from :mod:`src.heston.backtest`: the sizing rule
(:func:`size_position`), cost model (:func:`trade_cost`), bulk delta pricing
(:func:`compute_deltas`) and the regime sizing/position/stop/min-mid constants.
Only the execution mechanics (whole-contract sizing, IBKR order routing, 30-DTE
exit) are new. The backtest's underlying-leverage cap is deliberately omitted:
it is incompatible with whole-contract SPX granularity (one contract already
exceeds it) and its risk purpose is served by the daily delta hedge and the
-5% stop.

Order routing is dependency-injected through the ``ib`` argument:
    - ``ib is None``  -> paper simulation, fills booked at the quoted mid
      (deterministic, fully offline-testable);
    - ``ib`` given    -> limit-at-mid orders to the IBKR paper account, with
      the SPY delta hedge reconciled to a single net order per run.

``ib_insync`` is imported lazily, so the offline suite needs neither it nor a
running TWS. The live order helpers require TWS on the paper account.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.heston.backtest import (
    CALM_ALLOCATION_FRACTION,
    CALM_MAX_POSITIONS,
    HIGH_VOL_ALLOCATION_FRACTION,
    HIGH_VOL_MAX_POSITIONS,
    MIN_ENTRY_MID,
    STOP_LOSS_FRACTION,
    compute_deltas,
    size_position,
    trade_cost,
)
from src.heston.fft import HestonParams
from src.paper_trading.data_live import (
    IBKR_HOST,
    IBKR_PORT,
    OPTION_EXCHANGE,
    UNDERLYING_CURRENCY,
)
from src.paper_trading.portfolio import (
    CONTRACT_MULTIPLIER,
    LivePosition,
    Portfolio,
)

logger = logging.getLogger("paper_trading.execution")

# --- Live exit / order config -----------------------------------------------------
EXIT_DTE_THRESHOLD: int = 30  # close at 30 days to expiry (live spec)
SHORT_DIRECTION: int = -1  # production strategy is short-vol only
TRADE_CLIENT_ID: int = 12  # client id reserved for the execution process
ORDER_FILL_TIMEOUT_SECONDS: float = 30.0
ORDER_POLL_SECONDS: float = 1.0
LIMIT_PRICE_DECIMALS: int = 2  # SPX option price tick granularity

# --- Delta-hedge instrument -------------------------------------------------------
HEDGE_SYMBOL: str = "SPY"  # SPX index is not directly tradable; hedge via SPY
HEDGE_EXCHANGE: str = "SMART"

# --- Order actions ----------------------------------------------------------------
ACTION_BUY: str = "BUY"
ACTION_SELL: str = "SELL"

# --- Exit reasons -----------------------------------------------------------------
EXIT_STOP: str = "stop"
EXIT_REVERT: str = "revert"
EXIT_EXPIRY: str = "expiry"
EXIT_DELISTED: str = "delisted"


@dataclass(frozen=True)
class FillResult:
    """Outcome of an order.

    Attributes:
        filled: Whether the order fully filled.
        fill_price: Average fill price (NaN if unfilled).
        quantity: Filled quantity.
    """

    filled: bool
    fill_price: float
    quantity: float


@dataclass(frozen=True)
class EntryPlan:
    """A sized new short-vol entry, before it is routed/booked.

    Attributes:
        optionid: Contract id.
        units: Whole option contracts to short.
        allocation: Dollar equity allocated (stop-loss base).
        mid: Quoted option mid (limit price).
        spot: Index level.
        market_iv: Market implied vol at entry.
        model_iv: Heston model IV at entry.
        deviation: market_iv - model_iv at entry.
        delta: Model delta at entry.
        strike: Strike price.
        cp_flag: "C" or "P".
        exdate: Expiry date.
    """

    optionid: int
    units: float
    allocation: float
    mid: float
    spot: float
    market_iv: float
    model_iv: float
    deviation: float
    delta: float
    strike: float
    cp_flag: str
    exdate: pd.Timestamp


# --- Pure decision logic (no ib_insync, fully testable offline) --------------------


def size_contracts(allocation: float, mid: float, multiplier: float) -> int:
    """Whole option contracts affordable at a target dollar allocation.

    Args:
        allocation: Target dollar premium notional.
        mid: Option mid price (per share).
        multiplier: Contract multiplier.

    Returns:
        ``floor(allocation / (mid * multiplier))`` — zero when a single
        contract exceeds the allocation.
    """
    per_contract = mid * multiplier
    if per_contract <= 0.0:
        return 0
    return int(math.floor(allocation / per_contract))


def classify_exit(
    position: LivePosition, mid: float, days_to_expiry: int, exit_signal: bool
) -> str | None:
    """Exit reason for a marked position, or None to hold.

    Precedence matches the backtest: stop loss, then reversion, then expiry.
    The position must already be marked to ``mid`` (its ``cum_pnl`` current).

    Args:
        position: The open position (already marked).
        mid: Latest option mid.
        days_to_expiry: Days to the contract's expiry.
        exit_signal: Whether the deviation has reverted within tolerance.

    Returns:
        One of :data:`EXIT_STOP`/:data:`EXIT_REVERT`/:data:`EXIT_EXPIRY`, or
        None.
    """
    if position.cum_pnl <= -STOP_LOSS_FRACTION * position.allocation:
        return EXIT_STOP
    if exit_signal:
        return EXIT_REVERT
    if days_to_expiry <= EXIT_DTE_THRESHOLD:
        return EXIT_EXPIRY
    return None


def regime_config(is_high_vol: bool) -> tuple[float, int]:
    """Base sizing fraction and concurrent-position cap for the regime.

    Args:
        is_high_vol: Whether today is classified high-vol.

    Returns:
        ``(base_fraction, max_positions)`` — (0.5%, 20) high-vol, else (1%, 50).
    """
    if is_high_vol:
        return HIGH_VOL_ALLOCATION_FRACTION, HIGH_VOL_MAX_POSITIONS
    return CALM_ALLOCATION_FRACTION, CALM_MAX_POSITIONS


def hedge_target_shares(direction: int, units: float, multiplier: float, delta: float) -> float:
    """Signed underlying holding that neutralises an option position's delta.

    Args:
        direction: +1 long / -1 short the option.
        units: Option contracts held.
        multiplier: Contract multiplier.
        delta: Model delta per share.

    Returns:
        ``-direction * units * multiplier * delta`` in SPX-equivalent shares.
    """
    return -direction * units * multiplier * delta


def select_entries(
    portfolio: Portfolio,
    quotes: pd.DataFrame,
    equity: float,
    deltas: dict[int, float],
    is_high_vol: bool,
) -> list[EntryPlan]:
    """Choose and size today's new short-vol entries.

    Strongest liquid short signals first, honouring the regime position cap,
    the minimum entry mid and whole-contract affordability. Positions are
    sized purely by premium allocation (the underlying-leverage cap is not
    applied — see the module docstring).

    Args:
        portfolio: Current book (its open positions are excluded and counted
            toward the regime cap).
        quotes: Today's flagged quote panel indexed by optionid, with
            ``entry_short``, ``deviation``, ``mid``, ``spot``, ``market_iv``,
            ``model_iv``, ``strike``, ``cp_flag``, ``exdate`` columns.
        equity: Current marked equity (sizing base).
        deltas: Model delta per optionid.
        is_high_vol: Regime flag.

    Returns:
        Sized :class:`EntryPlan` list.
    """
    base_fraction, max_positions = regime_config(is_high_vol)
    open_ids = set(portfolio.open_positions)

    candidates = quotes.loc[
        quotes["entry_short"].fillna(False) & ~quotes.index.isin(open_ids)
    ]
    candidates = candidates.reindex(
        candidates["deviation"].abs().sort_values(ascending=False).index
    )

    plans: list[EntryPlan] = []
    for optionid, row in candidates.iterrows():
        if len(open_ids) + len(plans) >= max_positions:
            break
        mid = float(row["mid"])
        if mid < MIN_ENTRY_MID or equity <= 0.0:
            continue
        market_iv = float(row["market_iv"])
        allocation = size_position(equity, market_iv, base_fraction)
        units = size_contracts(allocation, mid, CONTRACT_MULTIPLIER)
        if units < 1:
            continue
        plans.append(
            EntryPlan(
                optionid=int(optionid),
                units=float(units),
                allocation=allocation,
                mid=mid,
                spot=float(row["spot"]),
                market_iv=market_iv,
                model_iv=float(row["model_iv"]),
                deviation=float(row["deviation"]),
                delta=float(deltas.get(int(optionid), 0.0)),
                strike=float(row["strike"]),
                cp_flag=str(row["cp_flag"]),
                exdate=pd.Timestamp(row["exdate"]),
            )
        )
    return plans


# --- IBKR order routing (thin, lazy, requires TWS) --------------------------------


def connect_trading(
    host: str = IBKR_HOST,
    port: int = IBKR_PORT,
    client_id: int = TRADE_CLIENT_ID,
) -> Any:
    """Open a read/write IBKR connection for order placement.

    Args:
        host: TWS/Gateway host.
        port: TWS/Gateway socket (paper trading is 7497).
        client_id: Client id unique to the execution process.

    Returns:
        A connected ``ib_insync.IB`` instance (caller must ``disconnect``).
    """
    from ib_insync import IB

    ib = IB()
    ib.connect(host=host, port=port, clientId=client_id)
    logger.info("execution connected to IBKR %s:%d (clientId=%d)", host, port, client_id)
    return ib


def _option_contract(ib: Any, optionid: int) -> Any:
    """Resolve an option contract from its IBKR conId."""
    from ib_insync import Contract

    contract = Contract(
        secType="OPT",
        conId=int(optionid),
        exchange=OPTION_EXCHANGE,
        currency=UNDERLYING_CURRENCY,
    )
    ib.qualifyContracts(contract)
    return contract


def place_limit_order(
    ib: Any,
    contract: Any,
    action: str,
    quantity: float,
    limit_price: float,
    timeout: float = ORDER_FILL_TIMEOUT_SECONDS,
) -> FillResult:
    """Place a limit order and wait (up to ``timeout``) for a fill.

    Args:
        ib: Connected IBKR instance.
        contract: Contract to trade.
        action: ``BUY`` or ``SELL``.
        quantity: Contracts/shares.
        limit_price: Limit price (rounded to the price tick).
        timeout: Seconds to wait before cancelling.

    Returns:
        The fill outcome; unfilled orders are cancelled.
    """
    from ib_insync import LimitOrder

    order = LimitOrder(action, quantity, round(limit_price, LIMIT_PRICE_DECIMALS))
    trade = ib.placeOrder(contract, order)
    waited = 0.0
    while not trade.isDone() and waited < timeout:
        ib.sleep(ORDER_POLL_SECONDS)
        waited += ORDER_POLL_SECONDS
    if trade.orderStatus.status == "Filled":
        return FillResult(
            filled=True,
            fill_price=float(trade.orderStatus.avgFillPrice),
            quantity=float(trade.orderStatus.filled),
        )
    ib.cancelOrder(order)
    logger.warning(
        "order not filled (%s %s x%.0f @ %.2f); cancelled",
        action,
        getattr(contract, "conId", "?"),
        quantity,
        limit_price,
    )
    return FillResult(filled=False, fill_price=float("nan"), quantity=0.0)


def _fill_price(
    ib: Any | None, optionid: int, action: str, quantity: float, mid: float
) -> float | None:
    """Fill price for one option trade: quoted mid in sim, real fill live.

    Args:
        ib: Connected IBKR instance, or None for paper simulation.
        optionid: Contract id.
        action: ``BUY`` or ``SELL``.
        quantity: Contracts.
        mid: Quoted mid (limit price).

    Returns:
        The fill price, or None if a live order did not fill.
    """
    if ib is None:
        return mid
    contract = _option_contract(ib, optionid)
    result = place_limit_order(ib, contract, action, quantity, mid)
    return result.fill_price if result.filled else None


def reconcile_hedge(ib: Any, portfolio: Portfolio, spot: float) -> None:
    """Route the net SPX-equivalent delta hedge as one SPY order.

    The portfolio tracks each position's hedge in SPX-equivalent shares; the
    net across the book is converted to SPY at the SPY/SPX price ratio and the
    difference versus the current SPY position is traded.

    Args:
        ib: Connected IBKR instance.
        portfolio: Current book (its ``hedge_shares`` are the target).
        spot: SPX index level.
    """
    from ib_insync import MarketOrder, Stock

    hedge = Stock(HEDGE_SYMBOL, HEDGE_EXCHANGE, UNDERLYING_CURRENCY)
    ib.qualifyContracts(hedge)
    (ticker,) = ib.reqTickers(hedge)
    spy_price = ticker.marketPrice()
    if spy_price is None or not np.isfinite(spy_price) or spy_price <= 0.0:
        logger.warning("no SPY price; skipping hedge reconciliation")
        return
    target_spx = sum(pos.hedge_shares for pos in portfolio.open_positions.values())
    target_spy = round(target_spx * spot / spy_price)
    current_spy = sum(
        p.position for p in ib.positions() if p.contract.symbol == HEDGE_SYMBOL
    )
    delta_shares = target_spy - current_spy
    if delta_shares == 0:
        return
    action = ACTION_BUY if delta_shares > 0 else ACTION_SELL
    ib.placeOrder(hedge, MarketOrder(action, abs(delta_shares)))
    logger.info("hedge: %s %d SPY (target=%d, current=%d)", action, abs(delta_shares), target_spy, current_spy)


# --- Orchestration ----------------------------------------------------------------


def _mark_open_positions(portfolio: Portfolio, quotes: pd.DataFrame) -> None:
    """Mark every open position that is quoted today to its mid and index level."""
    for optionid, position in portfolio.open_positions.items():
        if optionid in quotes.index:
            row = quotes.loc[optionid]
            portfolio.mark_position(optionid, float(row["mid"]), float(row["spot"]))


def _process_exits(
    portfolio: Portfolio,
    quotes: pd.DataFrame,
    today: pd.Timestamp,
    ib: Any | None,
) -> list[dict[str, Any]]:
    """Close positions that hit an exit rule; return the closed-trade records."""
    closed: list[dict[str, Any]] = []
    for optionid in list(portfolio.open_positions):
        position = portfolio.open_positions[optionid]
        if optionid not in quotes.index:
            exit_cost = trade_cost(
                position.units * position.multiplier * position.last_mid
            ) + trade_cost(position.hedge_shares * position.last_spot)
            closed.append(
                portfolio.close_position(optionid, today, EXIT_DELISTED, exit_cost)
            )
            continue
        row = quotes.loc[optionid]
        reason = classify_exit(
            position, float(row["mid"]), int(row["days_to_expiry"]), bool(row["exit_signal"])
        )
        if reason is None:
            continue
        close_action = ACTION_BUY if position.direction < 0 else ACTION_SELL
        fill = _fill_price(ib, optionid, close_action, position.units, float(row["mid"]))
        if fill is None:
            logger.warning("exit for %d (%s) did not fill; holding", optionid, reason)
            continue
        spot = float(row["spot"])
        portfolio.mark_position(optionid, fill, spot)  # capture fill vs mark
        exit_cost = trade_cost(
            position.units * position.multiplier * fill
        ) + trade_cost(position.hedge_shares * spot)
        closed.append(portfolio.close_position(optionid, today, reason, exit_cost))
    return closed


def _rehedge_survivors(
    portfolio: Portfolio, quotes: pd.DataFrame, deltas: dict[int, float]
) -> None:
    """Re-hedge every quoted open position to the model delta, charging cost."""
    for optionid, position in portfolio.open_positions.items():
        if optionid not in quotes.index:
            continue
        spot = float(quotes.loc[optionid, "spot"])
        target = hedge_target_shares(
            position.direction, position.units, position.multiplier,
            float(deltas.get(optionid, 0.0)),
        )
        cost = trade_cost((target - position.hedge_shares) * spot)
        portfolio.apply_cost(optionid, cost)
        portfolio.set_hedge(optionid, target)


def _open_entries(
    portfolio: Portfolio,
    plans: list[EntryPlan],
    today: pd.Timestamp,
    ib: Any | None,
) -> list[LivePosition]:
    """Route/book each planned entry and open it in the portfolio."""
    opened: list[LivePosition] = []
    for plan in plans:
        fill = _fill_price(ib, plan.optionid, ACTION_SELL, plan.units, plan.mid)
        if fill is None:
            logger.warning("entry for %d did not fill; skipping", plan.optionid)
            continue
        hedge_shares = hedge_target_shares(
            SHORT_DIRECTION, plan.units, CONTRACT_MULTIPLIER, plan.delta
        )
        entry_cost = trade_cost(plan.units * CONTRACT_MULTIPLIER * fill)
        hedge_cost = trade_cost(hedge_shares * plan.spot)
        position = LivePosition(
            optionid=plan.optionid,
            direction=SHORT_DIRECTION,
            units=plan.units,
            allocation=plan.allocation,
            entry_date=pd.Timestamp(today),
            entry_mid=fill,
            model_iv_entry=plan.model_iv,
            market_iv_entry=plan.market_iv,
            delta_entry=plan.delta,
            entry_deviation=plan.deviation,
            strike=plan.strike,
            cp_flag=plan.cp_flag,
            exdate=plan.exdate,
            last_mid=fill,
            last_spot=plan.spot,
            multiplier=CONTRACT_MULTIPLIER,
            hedge_shares=hedge_shares,
            cum_pnl=-(entry_cost + hedge_cost),
        )
        portfolio.open_position(position)
        opened.append(position)
    return opened


def execute_daily(
    portfolio: Portfolio,
    quotes: pd.DataFrame,
    params: HestonParams,
    is_high_vol: bool,
    today: pd.Timestamp,
    ib: Any | None = None,
) -> dict[str, Any]:
    """Run one trading day of execution against the portfolio.

    Marks and exits open positions, re-hedges survivors, opens new short-vol
    entries, and (when ``ib`` is supplied) reconciles the net SPY hedge.

    Args:
        portfolio: The book to mutate.
        quotes: Today's flagged, model-priced quote panel. Indexed by
            optionid, or carrying an ``optionid`` column.
        params: Heston parameters governing today (for hedge deltas).
        is_high_vol: Regime flag from ``high_vol_regime``.
        today: Trading date.
        ib: Connected IBKR instance for live routing, or None to book at mid.

    Returns:
        Summary dict with ``opened`` and ``closed`` counts and records.
    """
    quotes = quotes.copy()
    if quotes.index.name != "optionid":
        quotes = quotes.set_index("optionid")
    quotes = quotes[~quotes.index.duplicated(keep="first")]

    _mark_open_positions(portfolio, quotes)
    closed = _process_exits(portfolio, quotes, today, ib)

    deltas = compute_deltas(params, quotes, dict.fromkeys(quotes.index.tolist()))
    _rehedge_survivors(portfolio, quotes, deltas)

    plans = select_entries(portfolio, quotes, portfolio.equity, deltas, is_high_vol)
    opened = _open_entries(portfolio, plans, today, ib)

    if ib is not None and portfolio.open_positions:
        spot = float(quotes["spot"].iloc[0])
        reconcile_hedge(ib, portfolio, spot)

    logger.info(
        "%s: opened %d, closed %d, open now %d",
        pd.Timestamp(today).date(),
        len(opened),
        len(closed),
        len(portfolio.open_positions),
    )
    return {
        "opened": [pos.to_dict() for pos in opened],
        "closed": closed,
        "n_opened": len(opened),
        "n_closed": len(closed),
    }
