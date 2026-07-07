"""Offline tests for src.paper_trading.data_live (no ib_insync, no TWS)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.paper_trading.data_live import (
    LIVE_CHAIN_BASE_COLUMNS,
    MAX_MATURITY_MONTHS,
    MIN_MATURITY_MONTHS,
    filter_expirations,
    filter_strikes,
    imply_rates_and_dividends,
    load_stored_chains,
    select_option_chain,
    store_chain,
    tickers_to_frame,
)


def make_ticker(
    con_id: int,
    right: str,
    strike: float,
    expiry: str,
    bid: float,
    ask: float,
    iv: float,
    delta: float = -0.3,
    vega: float = 12.0,
    volume: float = 100.0,
) -> SimpleNamespace:
    """Build a duck-typed ib_insync Ticker fixture."""
    contract = SimpleNamespace(
        conId=con_id,
        right=right,
        strike=strike,
        lastTradeDateOrContractMonth=expiry,
    )
    greeks = (
        None
        if iv is None
        else SimpleNamespace(impliedVol=iv, delta=delta, vega=vega)
    )
    return SimpleNamespace(
        contract=contract, bid=bid, ask=ask, volume=volume, modelGreeks=greeks
    )


def test_filter_expirations_keeps_one_to_twelve_month_band() -> None:
    today = pd.Timestamp("2026-01-15")
    expirations = [
        (today + pd.Timedelta(days=3)).strftime("%Y%m%d"),  # too soon
        (today + pd.Timedelta(days=32)).strftime("%Y%m%d"),  # ~1M, keep
        (today + pd.Timedelta(days=182)).strftime("%Y%m%d"),  # ~6M, keep
        (today + pd.Timedelta(days=360)).strftime("%Y%m%d"),  # ~12M, keep
        (today + pd.Timedelta(days=500)).strftime("%Y%m%d"),  # too far
    ]
    kept = filter_expirations(expirations, today)
    assert kept == sorted(expirations[1:4])
    assert MIN_MATURITY_MONTHS == 1 and MAX_MATURITY_MONTHS == 12


def test_filter_strikes_moneyness_band() -> None:
    spot = 5000.0
    strikes = [3500.0, 4000.0, 5000.0, 6000.0, 6500.0]
    kept = filter_strikes(strikes, spot)
    # 80-120% of 5000 = [4000, 6000]; 3500 and 6500 excluded.
    assert kept == [4000.0, 5000.0, 6000.0]


def test_select_option_chain_picks_am_settled_spx() -> None:
    chains = [
        SimpleNamespace(tradingClass="SPXW", exchange="CBOE", strikes={1.0}),
        SimpleNamespace(tradingClass="SPX", exchange="CBOE", strikes={2.0}),
    ]
    chosen = select_option_chain(chains)
    assert chosen.tradingClass == "SPX" and chosen.strikes == {2.0}
    with pytest.raises(ValueError, match="no SPX chain"):
        select_option_chain([SimpleNamespace(tradingClass="SPXW", exchange="CBOE")])


def test_tickers_to_frame_schema_and_values() -> None:
    today = pd.Timestamp("2026-01-15")
    tickers = [
        make_ticker(1, "P", 4900.0, "20260220", bid=40.0, ask=42.0, iv=0.18),
        make_ticker(2, "C", 5100.0, "20260220", bid=30.0, ask=32.0, iv=0.15),
    ]
    frame = tickers_to_frame(tickers, spot=5000.0, today=today)
    for column in LIVE_CHAIN_BASE_COLUMNS:
        assert column in frame.columns
    for derived in ("mid", "spread", "days_to_expiry", "moneyness"):
        assert derived in frame.columns
    put = frame.loc[frame["optionid"] == 1].iloc[0]
    assert put["cp_flag"] == "P"
    assert put["mid"] == pytest.approx(41.0)
    assert put["impl_volatility"] == pytest.approx(0.18)
    assert put["exdate"] == pd.Timestamp("2026-02-20")
    assert (frame["date"] == today).all()


def test_tickers_to_frame_drops_missing_iv_and_quotes() -> None:
    today = pd.Timestamp("2026-01-15")
    tickers = [
        make_ticker(1, "P", 4900.0, "20260220", bid=40.0, ask=42.0, iv=0.18),
        make_ticker(2, "C", 5100.0, "20260220", bid=30.0, ask=32.0, iv=None),  # no greeks
        make_ticker(3, "C", 5050.0, "20260220", bid=-1.0, ask=-1.0, iv=0.16),  # no quote
    ]
    frame = tickers_to_frame(tickers, spot=5000.0, today=today)
    assert list(frame["optionid"]) == [1]


def _synthetic_parity_day(
    spot: float, rate: float, div: float, days: int
) -> pd.DataFrame:
    """Chain for one date/maturity whose C-P exactly satisfies parity."""
    maturity = days / 365.0
    strikes = np.array([4850.0, 4900.0, 4950.0, 5000.0, 5050.0, 5100.0, 5150.0])
    disc, fwd_pv = np.exp(-rate * maturity), spot * np.exp(-div * maturity)
    put_mid = 60.0  # arbitrary positive put price
    call_mid = put_mid + fwd_pv - strikes * disc  # enforce C - P = fwd_pv - K*disc
    rows = []
    for strike, call, put in zip(strikes, call_mid, put_mid * np.ones_like(strikes)):
        for cp_flag, mid in (("C", call), ("P", put)):
            rows.append(
                {
                    "date": pd.Timestamp("2026-01-15"),
                    "exdate": pd.Timestamp("2026-01-15") + pd.Timedelta(days=days),
                    "cp_flag": cp_flag,
                    "strike": strike,
                    "mid": mid,
                    "spot": spot,
                    "days_to_expiry": days,
                    "moneyness": strike / spot,
                }
            )
    return pd.DataFrame(rows)


def test_imply_rates_and_dividends_recovers_known_r_and_q() -> None:
    spot, rate, div, days = 5000.0, 0.045, 0.018, 90
    chain = _synthetic_parity_day(spot, rate, div, days)
    zero_curve, div_yield = imply_rates_and_dividends(chain)
    assert list(zero_curve.columns) == ["date", "days", "rate"]
    assert list(div_yield.columns) == ["date", "rate"]
    # rates are returned in percent
    assert float(zero_curve["rate"].iloc[0]) == pytest.approx(rate * 100.0, abs=1e-4)
    assert float(div_yield["rate"].iloc[0]) == pytest.approx(div * 100.0, abs=1e-4)
    assert float(zero_curve["days"].iloc[0]) == pytest.approx(days)


def test_imply_rates_skips_maturities_with_too_few_strikes() -> None:
    # Only two paired strikes (< PARITY_MIN_STRIKES) -> maturity dropped, no crash.
    chain = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-15")] * 4,
            "exdate": [pd.Timestamp("2026-04-15")] * 4,
            "cp_flag": ["C", "P", "C", "P"],
            "strike": [4950.0, 4950.0, 5050.0, 5050.0],
            "mid": [70.0, 50.0, 40.0, 60.0],
            "spot": [5000.0] * 4,
            "days_to_expiry": [90] * 4,
            "moneyness": [0.99, 0.99, 1.01, 1.01],
        }
    )
    zero_curve, div_yield = imply_rates_and_dividends(chain)
    assert zero_curve.empty and div_yield.empty


def test_store_and_load_stored_chains_roundtrip(tmp_path) -> None:
    today = pd.Timestamp("2026-01-15")
    tickers = [
        make_ticker(1, "P", 4900.0, "20260220", bid=40.0, ask=42.0, iv=0.18),
        make_ticker(2, "C", 5100.0, "20260220", bid=30.0, ask=32.0, iv=0.15),
    ]
    frame = tickers_to_frame(tickers, spot=5000.0, today=today)
    path = store_chain(frame, today, directory=tmp_path)
    assert path.name == "2026-01-15.parquet"

    loaded = load_stored_chains(today, today, directory=tmp_path)
    assert len(loaded) == len(frame)
    # window excluding the file returns empty
    empty = load_stored_chains(
        today + pd.Timedelta(days=1), today + pd.Timedelta(days=5), directory=tmp_path
    )
    assert empty.empty
