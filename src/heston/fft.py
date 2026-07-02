"""Carr-Madan FFT option pricing under the Heston (1993) model.

Implements:
    - the Heston characteristic function in the trap-free formulation of
      Albrecher, Mayer, Schoutens & Tistaert (2007), which is continuous in
      the principal complex branch,
    - Carr & Madan (1999) FFT call pricing with Simpson-rule weights,
    - puts via put-call parity,
    - Black-Scholes pricing and implied-volatility inversion (Brent),
    - a slow Gil-Pelaez / Heston (1993) quadrature pricer used only as an
      independent validation benchmark for the FFT,
    - Heston delta by central finite difference (used by the backtest).

All rates are continuously compounded decimals; maturities are in years.

Run a self-validation against benchmarks:
    python -m src.heston.fft
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm

# --- Carr-Madan grid ----------------------------------------------------------
FFT_N: int = 4096
DAMPENING_ALPHA: float = 1.5
GRID_SPACING_ETA: float = 0.25  # frequency-domain spacing
# log-strike spacing follows from the FFT constraint eta * lambda = 2 pi / N
LOG_STRIKE_SPACING: float = 2.0 * np.pi / (FFT_N * GRID_SPACING_ETA)

# --- Implied-vol inversion ------------------------------------------------------
IV_LOWER_BOUND: float = 1e-6
IV_UPPER_BOUND: float = 5.0

# --- Finite-difference delta ----------------------------------------------------
DELTA_REL_BUMP: float = 1e-4

# --- Vectorised Newton implied-vol inversion --------------------------------------
NEWTON_MAX_ITER: int = 50
NEWTON_TOL: float = 1e-10
NEWTON_MAX_STEP: float = 0.5
NEWTON_VEGA_FLOOR: float = 1e-8
NEWTON_SEED_MIN: float = 0.01
NEWTON_SEED_MAX: float = 3.0
NEWTON_DEFAULT_SEED: float = 0.2  # used when no seed vol is supplied

# --- Quadrature benchmark -------------------------------------------------------
QUAD_UPPER_LIMIT: float = 500.0
QUAD_MAX_SUBDIVISIONS: int = 200


@dataclass(frozen=True)
class HestonParams:
    """Heston model parameters.

    Attributes:
        kappa: Mean-reversion speed of variance.
        theta: Long-run variance.
        sigma: Volatility of variance ("vol of vol").
        rho: Correlation between spot and variance Brownian motions.
        v0: Initial instantaneous variance.
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def as_array(self) -> np.ndarray:
        """Return parameters as ``[kappa, theta, sigma, rho, v0]``."""
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])

    @staticmethod
    def from_array(values: np.ndarray) -> "HestonParams":
        """Build parameters from ``[kappa, theta, sigma, rho, v0]``."""
        return HestonParams(*(float(v) for v in values))

    def feller_satisfied(self) -> bool:
        """Return True if the Feller condition 2*kappa*theta >= sigma^2 holds."""
        return 2.0 * self.kappa * self.theta >= self.sigma**2


def heston_char_fn(
    u: np.ndarray,
    params: HestonParams,
    maturity: float,
    rate: float,
    div_yield: float,
) -> np.ndarray:
    """Characteristic function of log(S_T / S_0) under Heston.

    Trap-free formulation: choosing the root ``kappa - rho*sigma*i*u - d``
    keeps the complex logarithm on its principal branch for all maturities.

    Args:
        u: Real or complex evaluation points.
        params: Heston parameters.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.

    Returns:
        Complex array phi(u) = E[exp(i*u*log(S_T/S_0))].
    """
    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta**2 + sigma**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * maturity)
    log_term = np.log((1.0 - g * exp_dt) / (1.0 - g))
    a_term = (kappa * theta / sigma**2) * (
        (beta - d) * maturity - 2.0 * log_term
    )
    b_term = (v0 / sigma**2) * (beta - d) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    drift = iu * (rate - div_yield) * maturity
    return np.exp(drift + a_term + b_term)


def carr_madan_grid(
    params: HestonParams,
    maturity: float,
    rate: float,
    div_yield: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Price normalised calls on the full Carr-Madan log-strike grid.

    Args:
        params: Heston parameters.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.

    Returns:
        Tuple ``(log_strikes, call_over_spot)`` where ``log_strikes`` is
        k = log(K/S_0) on a grid of size FFT_N and ``call_over_spot`` the
        corresponding call prices divided by spot.
    """
    alpha = DAMPENING_ALPHA
    eta = GRID_SPACING_ETA
    lam = LOG_STRIKE_SPACING
    j = np.arange(FFT_N)
    v = eta * j
    b = 0.5 * FFT_N * lam
    k = -b + lam * j

    with np.errstate(over="ignore", invalid="ignore"):
        phi = heston_char_fn(
            v - (alpha + 1.0) * 1j, params, maturity, rate, div_yield
        )
        psi = (
            np.exp(-rate * maturity)
            * phi
            / (alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v)
        )
        simpson = (3.0 + (-1.0) ** (j + 1)) / 3.0
        simpson[0] = 1.0 / 3.0
        integrand = np.exp(1j * b * v) * psi * eta * simpson
        calls = np.exp(-alpha * k) / np.pi * np.real(np.fft.fft(integrand))
    return k, calls


def heston_call_prices(
    params: HestonParams,
    spot: float,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    div_yield: float,
) -> np.ndarray:
    """Price European calls for arbitrary strikes via the Carr-Madan FFT.

    Cubic-spline interpolation of the FFT log-strike grid onto the
    requested strikes.

    Args:
        params: Heston parameters.
        spot: Current index level S_0.
        strikes: Strike prices (array-like).
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.

    Returns:
        Call prices, same shape as ``strikes``.
    """
    k_grid, calls_over_spot = carr_madan_grid(params, maturity, rate, div_yield)
    k = np.log(np.asarray(strikes, dtype=float) / spot)
    if not np.all(np.isfinite(calls_over_spot)):
        # extreme parameters overflowed the characteristic function
        return np.full(k.shape, np.nan)
    spline = CubicSpline(k_grid, calls_over_spot)
    return spot * spline(k)


def heston_price(
    params: HestonParams,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    cp_flag: str,
) -> float:
    """Price a European call or put under Heston (puts via put-call parity).

    Args:
        params: Heston parameters.
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        cp_flag: ``"C"`` or ``"P"``.

    Returns:
        Option price.

    Raises:
        ValueError: If ``cp_flag`` is not "C" or "P".
    """
    call = float(
        heston_call_prices(
            params, spot, np.array([strike]), maturity, rate, div_yield
        )[0]
    )
    if cp_flag == "C":
        return call
    if cp_flag == "P":
        forward_leg = spot * np.exp(-div_yield * maturity)
        strike_leg = strike * np.exp(-rate * maturity)
        return call - forward_leg + strike_leg
    raise ValueError(f"cp_flag must be 'C' or 'P', got {cp_flag!r}")


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    vol: float,
    cp_flag: str,
) -> float:
    """Black-Scholes price of a European option with continuous dividends.

    Args:
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        vol: Annualised volatility.
        cp_flag: ``"C"`` or ``"P"``.

    Returns:
        Option price.

    Raises:
        ValueError: If ``cp_flag`` is not "C" or "P".
    """
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * maturity
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc_spot = spot * np.exp(-div_yield * maturity)
    disc_strike = strike * np.exp(-rate * maturity)
    if cp_flag == "C":
        return float(disc_spot * norm.cdf(d1) - disc_strike * norm.cdf(d2))
    if cp_flag == "P":
        return float(disc_strike * norm.cdf(-d2) - disc_spot * norm.cdf(-d1))
    raise ValueError(f"cp_flag must be 'C' or 'P', got {cp_flag!r}")


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    cp_flag: str,
) -> float:
    """Invert Black-Scholes for implied volatility via Brent's method.

    Args:
        price: Observed (or model) option price.
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        cp_flag: ``"C"`` or ``"P"``.

    Returns:
        Implied volatility, or ``nan`` when ``price`` lies outside the
        no-arbitrage bounds spanned by the volatility bracket.
    """
    low = bs_price(spot, strike, maturity, rate, div_yield, IV_LOWER_BOUND, cp_flag)
    high = bs_price(spot, strike, maturity, rate, div_yield, IV_UPPER_BOUND, cp_flag)
    if not low <= price <= high:
        return float("nan")
    return float(
        brentq(
            lambda vol: bs_price(
                spot, strike, maturity, rate, div_yield, vol, cp_flag
            )
            - price,
            IV_LOWER_BOUND,
            IV_UPPER_BOUND,
        )
    )


def bs_call_price_array(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    div_yield: np.ndarray,
    vol: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes call prices.

    Args:
        spot: Index levels.
        strike: Strikes.
        maturity: Times to expiry in years.
        rate: Continuously compounded risk-free rates.
        div_yield: Continuously compounded dividend yields.
        vol: Annualised volatilities.

    Returns:
        Call prices, broadcast over the inputs.
    """
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * maturity
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * np.exp(-div_yield * maturity) * norm.cdf(d1) - strike * np.exp(
        -rate * maturity
    ) * norm.cdf(d2)


def bs_vega_array(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    div_yield: np.ndarray,
    vol: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes vega (identical for calls and puts).

    Args:
        spot: Index levels.
        strike: Strikes.
        maturity: Times to expiry in years.
        rate: Continuously compounded risk-free rates.
        div_yield: Continuously compounded dividend yields.
        vol: Annualised volatilities.

    Returns:
        dPrice/dVol, broadcast over the inputs.
    """
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * maturity
    ) / (vol * sqrt_t)
    return spot * np.exp(-div_yield * maturity) * norm.pdf(d1) * sqrt_t


def implied_vol_newton(
    call_price: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    div_yield: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """Vectorised implied vol from CALL prices via damped Newton iteration.

    By put-call parity a call and put at the same strike share one implied
    vol, so put quotes should be converted to call prices before inversion.

    Args:
        call_price: Target call prices.
        spot: Index levels.
        strike: Strikes.
        maturity: Times to expiry in years.
        rate: Continuously compounded risk-free rates.
        div_yield: Continuously compounded dividend yields.
        seed: Starting vols (e.g. market implied vols).

    Returns:
        Implied vols; ``nan`` where the target price violates no-arbitrage
        bounds or the iteration fails to converge.
    """
    price = np.asarray(call_price, dtype=float)
    lower = np.maximum(
        spot * np.exp(-div_yield * maturity) - strike * np.exp(-rate * maturity),
        0.0,
    )
    upper = spot * np.exp(-div_yield * maturity)
    valid = (price > lower) & (price < upper)

    seed = np.asarray(seed, dtype=float)
    seed = np.where(np.isnan(seed), NEWTON_DEFAULT_SEED, seed)
    iv = np.clip(seed, NEWTON_SEED_MIN, NEWTON_SEED_MAX)
    iv = np.where(valid, iv, np.nan)
    for _ in range(NEWTON_MAX_ITER):
        with np.errstate(all="ignore"):
            model = bs_call_price_array(spot, strike, maturity, rate, div_yield, iv)
            vega = bs_vega_array(spot, strike, maturity, rate, div_yield, iv)
            error = model - price
            if np.nanmax(np.abs(error), initial=0.0) < NEWTON_TOL:
                break
            step = np.clip(
                error / np.maximum(vega, NEWTON_VEGA_FLOOR),
                -NEWTON_MAX_STEP,
                NEWTON_MAX_STEP,
            )
            iv = np.clip(iv - step, IV_LOWER_BOUND, IV_UPPER_BOUND)
    with np.errstate(all="ignore"):
        model = bs_call_price_array(spot, strike, maturity, rate, div_yield, iv)
        converged = np.abs(model - price) < 1e-6
    return np.where(valid & converged, iv, np.nan)


def heston_price_and_iv(
    params: HestonParams,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    cp_flag: str,
) -> tuple[float, float]:
    """Return the Heston model price and its Black-Scholes implied vol.

    Args:
        params: Heston parameters.
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        cp_flag: ``"C"`` or ``"P"``.

    Returns:
        Tuple ``(price, implied_vol)``.
    """
    price = heston_price(params, spot, strike, maturity, rate, div_yield, cp_flag)
    iv = implied_vol(price, spot, strike, maturity, rate, div_yield, cp_flag)
    return price, iv


def heston_delta(
    params: HestonParams,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    cp_flag: str,
) -> float:
    """Heston delta by central finite difference in spot.

    Args:
        params: Heston parameters.
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        cp_flag: ``"C"`` or ``"P"``.

    Returns:
        dPrice/dSpot.
    """
    bump = spot * DELTA_REL_BUMP
    up = heston_price(params, spot + bump, strike, maturity, rate, div_yield, cp_flag)
    down = heston_price(params, spot - bump, strike, maturity, rate, div_yield, cp_flag)
    return (up - down) / (2.0 * bump)


def heston_deltas_bulk(
    params: HestonParams,
    spot: float,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    div_yield: float,
    cp_flags: np.ndarray,
) -> np.ndarray:
    """Heston deltas for many options at one (date, expiry) in two FFTs.

    Central finite difference in spot; put deltas via parity
    (delta_put = delta_call - exp(-q*T)).

    Args:
        params: Heston parameters.
        spot: Current index level.
        strikes: Strike prices.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.
        cp_flags: Array of "C"/"P" flags aligned with ``strikes``.

    Returns:
        Deltas aligned with ``strikes``.
    """
    bump = spot * DELTA_REL_BUMP
    up = heston_call_prices(params, spot + bump, strikes, maturity, rate, div_yield)
    down = heston_call_prices(params, spot - bump, strikes, maturity, rate, div_yield)
    call_delta = (up - down) / (2.0 * bump)
    put_delta = call_delta - np.exp(-div_yield * maturity)
    return np.where(np.asarray(cp_flags) == "C", call_delta, put_delta)


def heston_call_quad(
    params: HestonParams,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
) -> float:
    """Reference Heston (1993) call price via Gil-Pelaez quadrature.

    Independent of the Carr-Madan machinery (different integrand and
    numerical scheme); used only to validate the FFT implementation.

    Args:
        params: Heston parameters.
        spot: Current index level.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        div_yield: Continuously compounded dividend yield.

    Returns:
        Call price.
    """
    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    x = np.log(spot)

    def probability(j: int) -> float:
        b = kappa - rho * sigma if j == 1 else kappa
        uj = 0.5 if j == 1 else -0.5

        def integrand(u: float) -> float:
            iu = 1j * u
            beta = b - rho * sigma * iu
            d = np.sqrt(beta**2 - sigma**2 * (2.0 * uj * iu - u**2))
            g = (beta - d) / (beta + d)
            exp_dt = np.exp(-d * maturity)
            c_term = (rate - div_yield) * iu * maturity + (
                kappa * theta / sigma**2
            ) * ((beta - d) * maturity - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
            d_term = ((beta - d) / sigma**2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
            f = np.exp(c_term + d_term * v0 + iu * x)
            return float(np.real(np.exp(-iu * np.log(strike)) * f / iu))

        integral, _ = quad(
            integrand, 0.0, QUAD_UPPER_LIMIT, limit=QUAD_MAX_SUBDIVISIONS
        )
        return 0.5 + integral / np.pi

    p1 = probability(1)
    p2 = probability(2)
    return float(
        spot * np.exp(-div_yield * maturity) * p1
        - strike * np.exp(-rate * maturity) * p2
    )


def _validate() -> None:
    """Print FFT vs quadrature and BS-limit benchmark comparisons."""
    params = HestonParams(
        kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175
    )
    spot, rate, div_yield, maturity = 100.0, 0.025, 0.0, 1.0
    print("strike |      FFT |     quad |     diff")
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        fft_px = heston_price(params, spot, strike, maturity, rate, div_yield, "C")
        quad_px = heston_call_quad(params, spot, strike, maturity, rate, div_yield)
        print(
            f"{strike:6.0f} | {fft_px:8.4f} | {quad_px:8.4f} | {fft_px - quad_px:+.2e}"
        )
    bs_limit = HestonParams(kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0, v0=0.04)
    heston_px = heston_price(bs_limit, spot, 100.0, maturity, rate, div_yield, "C")
    bs_px = bs_price(spot, 100.0, maturity, rate, div_yield, 0.2, "C")
    print(f"BS limit (sigma->0, vol=20%): heston={heston_px:.4f} bs={bs_px:.4f}")


if __name__ == "__main__":
    _validate()
