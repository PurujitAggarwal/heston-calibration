"""Performance reporting for the Heston IV mean-reversion backtest.

Produces, on every run:
    - annualised Sharpe and Sortino ratios, maximum drawdown, win rate,
      average holding period and trade count (printed and written to
      reports/performance.txt)
    - the equity curve from $100,000 with drawdown underneath
      (reports/equity_curve.png; log scale — the curve spans four orders
      of magnitude)
    - the model-vs-market IV surface plot for the sample date
      (reports/iv_surface.png, regenerated via the surface stage)

Sharpe uses simple daily returns against a zero benchmark; Sortino uses
downside deviation against a zero target.

Run:
    python -m src.heston.reporting
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.heston.backtest import (
    EQUITY_CURVE_PARQUET,
    STARTING_CAPITAL,
    TRADES_PARQUET,
)
from src.heston.calibration import PARAMS_PARQUET
from src.heston.data import DIV_YIELD_PARQUET, OPTIONS_PARQUET, ZERO_CURVE_PARQUET
from src.heston.surface import (
    DEFAULT_PLOT_DATE,
    GRID_COLOR,
    MODEL_COLOR,
    MUTED_TEXT_COLOR,
    REPORTS_DIR,
    SURFACE_COLOR,
    TEXT_COLOR,
    plot_model_vs_market,
)

TRADING_DAYS_PER_YEAR: float = 252.0
DRAWDOWN_COLOR: str = "#e34948"

EQUITY_PNG = REPORTS_DIR / "equity_curve.png"
PERFORMANCE_TXT = REPORTS_DIR / "performance.txt"


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns of an equity series.

    Args:
        equity: Daily equity values.

    Returns:
        Daily returns (first observation dropped).
    """
    return equity.pct_change().dropna()


def annualised_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio against a zero benchmark.

    Args:
        returns: Daily simple returns.

    Returns:
        sqrt(252) * mean / std (nan when the deviation is zero).
    """
    std = float(returns.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / std)


def annualised_sortino(returns: pd.Series) -> float:
    """Annualised Sortino ratio with a zero target.

    Downside deviation is sqrt(mean(min(r, 0)^2)) over all observations.

    Args:
        returns: Daily simple returns.

    Returns:
        sqrt(252) * mean / downside_deviation (nan when there is no
        downside).
    """
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    if downside == 0.0 or np.isnan(downside):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / downside)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown.

    Args:
        equity: Daily equity values.

    Returns:
        Most negative equity/rolling-peak - 1 (a negative number).
    """
    return float((equity / equity.cummax() - 1.0).min())


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    """All required performance metrics for one backtest run.

    Args:
        equity_curve: Daily frame with ``date`` and ``equity``.
        trades: One row per closed trade with ``pnl`` and ``holding_days``.

    Returns:
        Metric name -> value.
    """
    equity = equity_curve["equity"]
    returns = daily_returns(equity)
    return {
        "starting_capital": STARTING_CAPITAL,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / STARTING_CAPITAL - 1.0),
        "sharpe": annualised_sharpe(returns),
        "sortino": annualised_sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "n_trades": float(len(trades)),
    }


def format_report(metrics: dict[str, float]) -> str:
    """Human-readable performance summary.

    Args:
        metrics: Output of :func:`compute_metrics`.

    Returns:
        Formatted multi-line report string.
    """
    return "\n".join(
        [
            "Heston IV mean-reversion backtest — performance summary",
            "=" * 56,
            f"Starting capital     : ${metrics['starting_capital']:>12,.0f}",
            f"Final equity         : ${metrics['final_equity']:>12,.2f}",
            f"Total return         : {metrics['total_return']:>12.1%}",
            f"Annualised Sharpe    : {metrics['sharpe']:>12.2f}",
            f"Annualised Sortino   : {metrics['sortino']:>12.2f}",
            f"Maximum drawdown     : {metrics['max_drawdown']:>12.1%}",
            f"Win rate             : {metrics['win_rate']:>12.1%}",
            f"Average holding      : {metrics['avg_holding_days']:>9.1f} days",
            f"Number of trades     : {metrics['n_trades']:>12,.0f}",
        ]
    )


def write_equity_plot(equity_curve: pd.DataFrame, output_path: Path) -> None:
    """Equity curve (log scale) with drawdown panel underneath.

    Args:
        equity_curve: Daily frame with ``date`` and ``equity``.
        output_path: Destination PNG path.
    """
    dates = equity_curve["date"]
    equity = equity_curve["equity"]
    drawdown = equity / equity.cummax() - 1.0

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(12.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor(SURFACE_COLOR)
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(SURFACE_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7)
        ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)

    ax_eq.plot(dates, equity, color=MODEL_COLOR, linewidth=2.0)
    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity ($, log scale)", color=MUTED_TEXT_COLOR, fontsize=10)
    ax_eq.axhline(
        STARTING_CAPITAL, color=MUTED_TEXT_COLOR, linewidth=1.0, linestyle="--"
    )
    final = float(equity.iloc[-1])
    ax_eq.annotate(
        f"final ${final:,.0f}",
        xy=(dates.iloc[-1], final),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=TEXT_COLOR,
        fontsize=10,
    )
    ax_eq.set_title(
        "Strategy equity from $100,000", color=TEXT_COLOR, fontsize=12, loc="left"
    )

    ax_dd.fill_between(dates, drawdown, 0.0, color=DRAWDOWN_COLOR, alpha=0.35)
    ax_dd.plot(dates, drawdown, color=DRAWDOWN_COLOR, linewidth=1.2)
    ax_dd.set_ylabel("Drawdown", color=MUTED_TEXT_COLOR, fontsize=10)
    ax_dd.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE_COLOR)
    plt.close(fig)


def main() -> None:
    """CLI entry point: ``python -m src.heston.reporting``."""
    equity_curve = pd.read_parquet(EQUITY_CURVE_PARQUET)
    trades = pd.read_parquet(TRADES_PARQUET)

    metrics = compute_metrics(equity_curve, trades)
    report = format_report(metrics)
    print(report)

    PERFORMANCE_TXT.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_TXT.write_text(report + "\n")
    write_equity_plot(equity_curve, EQUITY_PNG)
    print(f"\nwrote {PERFORMANCE_TXT}")
    print(f"wrote {EQUITY_PNG}")

    chain = pd.read_parquet(OPTIONS_PARQUET)
    zero_curve = pd.read_parquet(ZERO_CURVE_PARQUET)
    div_yield = pd.read_parquet(DIV_YIELD_PARQUET)
    params_table = pd.read_parquet(PARAMS_PARQUET)
    plotted = plot_model_vs_market(
        chain, zero_curve, div_yield, params_table, pd.Timestamp(DEFAULT_PLOT_DATE)
    )
    print(f"wrote model-vs-market IV surface for {plotted:%Y-%m-%d}")


if __name__ == "__main__":
    main()
