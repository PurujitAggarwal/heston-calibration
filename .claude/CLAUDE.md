# Heston Stochastic Volatility Calibration

## Project Vision
Full pipeline implementation of the Heston (1993) stochastic volatility model.
Calibrate model parameters to SPX option chains via Carr-Madan FFT pricing.
Reconstruct the full implied volatility surface and identify mispriced options.

## Model Specification
dS_t = μS_t dt + √v_t S_t dW_S
dv_t = κ(θ - v_t)dt + σ√v_t dW_v
corr(dW_S, dW_v) = ρ

Parameters: κ (mean reversion speed), θ (long-run variance), σ (vol of vol),
ρ (spot-vol correlation), v_0 (initial variance)

Pricing: Carr-Madan FFT method for European option prices
Calibration: Levenberg-Marquardt optimisation on implied vol surface

## Stack
- Language: Python 3.12
- Data: WRDS (OptionMetrics for SPX option chains)
- Key libraries: numpy, pandas, scipy, matplotlib
- FFT pricing: custom implementation in numpy

## Folder Structure
- src/heston/     — all production source code
- tests/          — pytest test suite
- data/raw/       — raw OptionMetrics data (read-only, never modify)
- data/processed/ — cleaned surfaces and calibrated parameters
- docs/           — write-ups and paper notes

## Pipeline Stages
1. data.py        — SPX option chain ingestion from OptionMetrics
2. fft.py         — Carr-Madan FFT option pricing
3. calibration.py — Levenberg-Marquardt parameter optimisation
4. surface.py     — implied volatility surface reconstruction
5. signals.py     — mispriced option identification
6. backtest.py    — delta-hedged P&L backtest
7. reporting.py   — Sharpe, Sortino, max drawdown, equity curve, vol surface plot

## Backtesting Rules
- Starting capital: $100,000
- Delta-hedged positions — hedge daily with underlying
- Transaction costs: 3bps per side
- Stop loss: 5% per position
- Entry: model IV deviates from market IV by more than 2 vol points
- Exit: deviation reverts within 0.5 vol points
- Quarterly recalibration of Heston parameters

## Performance Reporting (required on every backtest run)
- Equity curve from $100,000
- Annualised Sharpe ratio
- Annualised Sortino ratio
- Maximum drawdown
- Win rate
- Average holding period
- Number of trades
- Implied volatility surface plot (model vs market)

## No Lookahead Bias — Critical Rules
- All calibration uses only data available before each quarter start
- Rolling 1-year trailing windows for parameter estimation
- If lookahead bias detected anywhere, flag immediately

## Claude's Role
- Act as both autonomous coder and critical reviewer
- When building: write complete, production-quality code — not snippets
- When reviewing: flag lookahead bias and convergence issues first
- Never refactor code that isn't broken unless explicitly asked
- Never add dependencies without asking first
- All functions need type hints and docstrings
- No magic numbers — define all constants at top of each file
- Confirm with me after each pipeline stage before proceeding
- Always confirm before git commit or git push

## Key Papers
- Heston (1993): A Closed-Form Solution for Options with Stochastic Volatility
- Carr & Madan (1999): Option Valuation Using the Fast Fourier Transform
