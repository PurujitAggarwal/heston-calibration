"""Tests for src.heston.fft — Carr-Madan FFT pricing under Heston."""

from __future__ import annotations

import numpy as np
import pytest

from src.heston.fft import (
    HestonParams,
    bs_call_price_array,
    bs_price,
    bs_vega_array,
    implied_vol_newton,
    heston_call_prices,
    heston_call_quad,
    heston_char_fn,
    heston_delta,
    heston_price,
    heston_price_and_iv,
    implied_vol,
)

# Standard literature test set (Albrecher et al., 2007).
PARAMS = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)
SPOT = 100.0
RATE = 0.025
DIV = 0.015
MATURITY = 1.0


def test_char_fn_at_zero_is_one() -> None:
    phi = heston_char_fn(np.array([0.0]), PARAMS, MATURITY, RATE, DIV)
    assert phi[0] == pytest.approx(1.0 + 0.0j)


def test_char_fn_martingale_condition() -> None:
    # phi(-i) = E[S_T/S_0] must equal the forward growth factor.
    phi = heston_char_fn(np.array([-1j]), PARAMS, MATURITY, RATE, DIV)
    assert np.real(phi[0]) == pytest.approx(np.exp((RATE - DIV) * MATURITY), rel=1e-10)
    assert np.imag(phi[0]) == pytest.approx(0.0, abs=1e-10)


def test_fft_matches_quadrature_benchmark() -> None:
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        fft_px = heston_price(PARAMS, SPOT, strike, MATURITY, RATE, DIV, "C")
        quad_px = heston_call_quad(PARAMS, SPOT, strike, MATURITY, RATE, DIV)
        assert fft_px == pytest.approx(quad_px, abs=1e-4)


def test_fft_matches_black_scholes_limit() -> None:
    # sigma -> 0 with v0 = theta collapses Heston to BS at vol sqrt(theta).
    limit = HestonParams(kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0, v0=0.04)
    for strike in (85.0, 100.0, 115.0):
        heston_px = heston_price(limit, SPOT, strike, MATURITY, RATE, DIV, "C")
        bs_px = bs_price(SPOT, strike, MATURITY, RATE, DIV, 0.2, "C")
        assert heston_px == pytest.approx(bs_px, abs=5e-3)


def test_put_call_parity() -> None:
    strike = 105.0
    call = heston_price(PARAMS, SPOT, strike, MATURITY, RATE, DIV, "C")
    put = heston_price(PARAMS, SPOT, strike, MATURITY, RATE, DIV, "P")
    parity = SPOT * np.exp(-DIV * MATURITY) - strike * np.exp(-RATE * MATURITY)
    assert call - put == pytest.approx(parity, abs=1e-8)


def test_call_prices_monotone_and_within_bounds() -> None:
    strikes = np.linspace(80.0, 120.0, 41)
    calls = heston_call_prices(PARAMS, SPOT, strikes, MATURITY, RATE, DIV)
    assert np.all(np.diff(calls) < 0.0)
    lower = np.maximum(
        SPOT * np.exp(-DIV * MATURITY) - strikes * np.exp(-RATE * MATURITY), 0.0
    )
    assert np.all(calls >= lower - 1e-8)
    assert np.all(calls <= SPOT * np.exp(-DIV * MATURITY) + 1e-8)


def test_implied_vol_round_trip() -> None:
    vol = 0.234
    price = bs_price(SPOT, 110.0, MATURITY, RATE, DIV, vol, "P")
    recovered = implied_vol(price, SPOT, 110.0, MATURITY, RATE, DIV, "P")
    assert recovered == pytest.approx(vol, abs=1e-8)


def test_implied_vol_out_of_bounds_is_nan() -> None:
    assert np.isnan(implied_vol(-1.0, SPOT, 100.0, MATURITY, RATE, DIV, "C"))
    assert np.isnan(implied_vol(2.0 * SPOT, SPOT, 100.0, MATURITY, RATE, DIV, "C"))


def test_price_and_iv_returns_consistent_pair() -> None:
    price, iv = heston_price_and_iv(PARAMS, SPOT, 100.0, MATURITY, RATE, DIV, "C")
    assert bs_price(SPOT, 100.0, MATURITY, RATE, DIV, iv, "C") == pytest.approx(
        price, abs=1e-8
    )
    # ATM implied vol should sit between spot vol and long-run vol.
    assert np.sqrt(PARAMS.v0) * 0.8 < iv < np.sqrt(PARAMS.theta) * 1.2


def test_heston_delta_bounds_and_bs_limit() -> None:
    call_delta = heston_delta(PARAMS, SPOT, 100.0, MATURITY, RATE, DIV, "C")
    put_delta = heston_delta(PARAMS, SPOT, 100.0, MATURITY, RATE, DIV, "P")
    assert 0.0 < call_delta < 1.0
    assert -1.0 < put_delta < 0.0
    # Parity: call delta - put delta = exp(-q T).
    assert call_delta - put_delta == pytest.approx(
        np.exp(-DIV * MATURITY), abs=1e-4
    )


def test_invalid_cp_flag_raises() -> None:
    with pytest.raises(ValueError):
        heston_price(PARAMS, SPOT, 100.0, MATURITY, RATE, DIV, "X")
    with pytest.raises(ValueError):
        bs_price(SPOT, 100.0, MATURITY, RATE, DIV, 0.2, "straddle")


def test_newton_iv_matches_brentq() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.array([0.32, 0.25, 0.21, 0.19, 0.18])
    prices = bs_call_price_array(
        np.full(5, SPOT), strikes, np.full(5, MATURITY),
        np.full(5, RATE), np.full(5, DIV), vols,
    )
    seeds = np.full(5, 0.2)
    recovered = implied_vol_newton(
        prices, np.full(5, SPOT), strikes, np.full(5, MATURITY),
        np.full(5, RATE), np.full(5, DIV), seeds,
    )
    for k, price, iv in zip(strikes, prices, recovered):
        brent = implied_vol(float(price), SPOT, float(k), MATURITY, RATE, DIV, "C")
        assert iv == pytest.approx(brent, abs=1e-7)


def test_newton_iv_invalid_prices_are_nan() -> None:
    bad = np.array([-1.0, 200.0])  # below intrinsic / above forward bound
    strikes = np.array([100.0, 100.0])
    out = implied_vol_newton(
        bad, np.full(2, SPOT), strikes, np.full(2, MATURITY),
        np.full(2, RATE), np.full(2, DIV), np.full(2, 0.2),
    )
    assert np.isnan(out).all()


def test_bs_vega_array_matches_finite_difference() -> None:
    eps = 1e-6
    vega = bs_vega_array(
        np.array([SPOT]), np.array([105.0]), np.array([MATURITY]),
        np.array([RATE]), np.array([DIV]), np.array([0.2]),
    )[0]
    up = bs_price(SPOT, 105.0, MATURITY, RATE, DIV, 0.2 + eps, "C")
    down = bs_price(SPOT, 105.0, MATURITY, RATE, DIV, 0.2 - eps, "C")
    assert vega == pytest.approx((up - down) / (2 * eps), rel=1e-5)


def test_deep_smile_negative_rho_skew() -> None:
    # Negative rho must produce a downward-sloping IV skew in strike.
    _, iv_low = heston_price_and_iv(PARAMS, SPOT, 85.0, MATURITY, RATE, DIV, "P")
    _, iv_high = heston_price_and_iv(PARAMS, SPOT, 115.0, MATURITY, RATE, DIV, "C")
    assert iv_low > iv_high
