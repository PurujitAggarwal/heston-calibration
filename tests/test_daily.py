"""Offline tests for src.paper_trading.daily (no ib_insync, no TWS, no SMTP).

build_flagged_panel runs the real Carr-Madan pricing on a small synthetic
chain; run_day is exercised with everything injected (ib=None, send=False), so
no order is routed and no email is sent.
"""

from __future__ import annotations

import pandas as pd

from src.heston.fft import HestonParams
from src.paper_trading.daily import build_flagged_panel, classify_regime, run_day
from src.paper_trading.portfolio import Portfolio

PARAMS = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.03)
TODAY = pd.Timestamp("2026-06-30")
SPOT = 5000.0
DTE = 60


def make_chain() -> pd.DataFrame:
    """A one-day chain of three OTM puts and three OTM calls around spot."""
    exdate = TODAY + pd.Timedelta(days=DTE)
    rows = []
    strikes = [(4800.0, "P"), (4850.0, "P"), (4900.0, "P"),
               (5100.0, "C"), (5150.0, "C"), (5200.0, "C")]
    for i, (strike, cp) in enumerate(strikes):
        bid = 20.0
        rows.append(
            {
                "date": TODAY,
                "exdate": exdate,
                "cp_flag": cp,
                "strike": strike,
                "spot": SPOT,
                "moneyness": strike / SPOT,
                "days_to_expiry": DTE,
                "best_bid": bid,
                "best_offer": bid + 1.0,
                "mid": bid + 0.5,
                "spread": 1.0,
                "impl_volatility": 0.18,
                "optionid": 100 + i,
            }
        )
    return pd.DataFrame(rows)


def make_zero_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [TODAY, TODAY], "days": [30.0, 90.0], "rate": [4.5, 4.6]}
    )


def make_div_yield() -> pd.DataFrame:
    return pd.DataFrame({"date": [TODAY], "rate": [1.8]})


def make_panel() -> pd.DataFrame:
    """A flagged, model-priced short-vol panel indexed by optionid."""
    exdate = TODAY + pd.Timedelta(days=64)
    rows = []
    for i in range(2):
        rows.append(
            {
                "optionid": 200 + i,
                "cp_flag": "P",
                "strike": 4700.0 + 10.0 * i,
                "spot": SPOT,
                "mid": 5.0,
                "market_iv": 0.20,
                "model_iv": 0.17,
                "deviation": 0.03 + 0.0001 * i,
                "maturity_years": 64.0 / 365.0,
                "rate": 0.045,
                "div_yield": 0.018,
                "days_to_expiry": 64,
                "exit_signal": False,
                "entry_short": True,
                "exdate": exdate,
            }
        )
    return pd.DataFrame(rows).set_index("optionid")


def test_build_flagged_panel_has_signal_columns() -> None:
    panel = build_flagged_panel(
        make_chain(), PARAMS, make_zero_curve(), make_div_yield(), TODAY
    )
    assert len(panel) == 6  # all six OTM quotes survive
    for column in (
        "model_iv", "deviation", "entry_short", "exit_signal", "liquid",
        "mid", "days_to_expiry", "cp_flag", "maturity_years", "exdate",
    ):
        assert column in panel.columns
    assert panel["model_iv"].notna().any()  # Carr-Madan produced model IVs


def test_run_day_executes_and_records_offline(monkeypatch) -> None:
    monkeypatch.setattr(Portfolio, "save", lambda self, *args, **kwargs: None)
    book = Portfolio()
    chain = pd.DataFrame({"date": [TODAY], "spot": [SPOT]})

    result = run_day(
        today=TODAY,
        ib=None,
        chain=chain,
        panel=make_panel(),
        params=PARAMS,
        portfolio=book,
        is_high_vol=False,
        send=False,
    )

    assert result["n_opened"] >= 1
    assert len(book.open_positions) >= 1
    assert len(book.equity_curve) == 1  # today's snapshot recorded
    assert book.equity_curve[0]["spx_close"] == SPOT
    assert result["is_high_vol"] is False


def test_classify_regime_calm_without_history(tmp_path) -> None:
    chain = pd.DataFrame({"date": [TODAY], "spot": [SPOT]})
    assert classify_regime(TODAY, chain, directory=tmp_path) is False
