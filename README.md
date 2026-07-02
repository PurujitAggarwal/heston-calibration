# Heston Stochastic Volatility Calibration

Full pipeline for calibrating the Heston (1993) stochastic volatility model to
SPX option chains, reconstructing the implied volatility surface, and
backtesting a delta-hedged, regime-adaptive short-volatility strategy on the
resulting mispricing signals.

## Model

The Heston model describes the joint dynamics of the index level and its
instantaneous variance:

```
dS_t = μ S_t dt + √v_t S_t dW_S
dv_t = κ(θ − v_t) dt + σ√v_t dW_v
corr(dW_S, dW_v) = ρ
```

| Parameter | Meaning |
|---|---|
| κ | mean-reversion speed of variance |
| θ | long-run variance |
| σ | volatility of variance ("vol of vol") |
| ρ | spot–variance correlation |
| v₀ | initial instantaneous variance |

## Pipeline

Run the stages in order:

```bash
python -m src.heston.data          # SPX chain ingestion from WRDS OptionMetrics
python -m src.heston.calibration   # rolling quarterly Levenberg-Marquardt calibration
python -m src.heston.surface       # model IV surface + per-quarter RMSE + plot
python -m src.heston.signals       # daily deviation panel + entry/exit flags
python -m src.heston.backtest      # delta-hedged simulation from $100,000
python -m src.heston.reporting     # Sharpe/Sortino/drawdown report + plots
```

### 1. Data (`data.py`)

Pulls standard AM-settled monthly SPX options (secid 108105) from WRDS
OptionMetrics (`optionm_all.opprcd`), 2010-01-04 → 2025-08-29: 7.36M quotes
filtered server-side to 80–120% moneyness and 7–400 days to expiry, plus
forward prices, the zero-coupon curve, implied dividend yields and the index
level. Requires a WRDS account with credentials in `~/.pgpass`.

### 2. Pricing (`fft.py`)

Carr & Madan (1999) FFT pricing under the trap-free Heston characteristic
function (Albrecher et al., 2007): N=4096 grid points, dampening α=1.5,
Simpson weights, cubic-spline interpolation in log-strike; puts via put-call
parity. Validated against an independent Gil-Pelaez quadrature pricer
(agreement ~2×10⁻⁷) and the Black-Scholes σ→0 limit. Also provides vectorised
Newton implied-vol inversion and bulk finite-difference deltas.

### 3. Calibration (`calibration.py`)

For each quarter start 2011Q1–2025Q3, calibrates (κ, θ, σ, ρ, v₀) to the
pooled month-end OTM surfaces of the trailing one-year window — strictly
before the quarter start, so parameters never see the data they trade on.
Levenberg-Marquardt (`scipy.optimize.least_squares`, `method="lm"`) runs in a
transformed space (log for positive parameters, tanh for ρ) with 5 seeded
random restarts. Residuals are implied-vol differences weighted by inverse
bid-ask spread expressed in vol points (spread/vega). All 59 quarters
converge; per-quarter parameters, fit statistics and convergence flags land in
`data/processed/heston_params.parquet`.

### 4. Surface (`surface.py`)

Reconstructs the model IV surface on a 25-point moneyness × 6-tenor grid
(1/2/3/6/9/12 months) and scores model vs market out of sample per quarter
(mean RMSE 3.87 vol pts; the worst quarters are the 2020 and 2011 crashes, as
a lookahead-free design should show). Saves `reports/iv_surface.png`.

### 5. Signals (`signals.py`)

Daily panel of every OTM quote with its model IV and deviation
(market − model). Entry: market IV more than 2 vol pts **above** model
(short vol). The symmetric long-vol leg is permanently disabled
(`LONG_VOL_ENABLED = False`): it lost $110.8k at a 15% win rate over
2011–2025, fighting vol clustering and the implied-vol risk premium.
Exit: deviation reverts inside 0.5 vol pts. Entries require bid-ask spread
below 0.5 vol pts.

### 6. Backtest (`backtest.py`)

Daily event loop from $100,000: marks at mid, stop loss at −5% of allocated
capital (all costs included), revert/expiry exits, daily rehedge to the
Heston model delta, 3 bps per side on every option and hedge notional.

Position sizing is **regime-adaptive**: 21-day realised vol of SPX closes,
lagged one day, compared with the 80th percentile of its trailing 252-day
distribution (no lookahead).

| Regime | Base allocation | Max concurrent positions |
|---|---|---|
| Calm (≤ 80th pct) | 1% × (20% / market IV), capped 5% | 50 |
| High vol (> 80th pct) | 0.5% × (20% / market IV), capped 5% | 20 |

Risk guards: minimum $2 option premium, hedge notional capped at 10× the
allocation, 5-trading-day re-entry cooldown after a stop, at most 10 new
positions per day.

### 7. Reporting (`reporting.py`)

Writes `reports/performance.txt`, the equity/drawdown chart and the model vs
market IV surface plot on every run.

## Production results (2011–2025, net of costs)

| Metric | Value |
|---|---|
| Final equity (from $100,000) | $722,916 |
| Total return | +622.9% |
| Annualised Sharpe | 0.93 |
| Annualised Sortino | 1.18 |
| Maximum drawdown | −33.5% |
| Win rate | 43.9% |
| Trades | 8,377 |
| Average holding period | 7.3 days |

Caveats to read alongside the numbers: the short-vol-only direction and the
regime thresholds were selected after observing the two-sided strategy's
results on the same sample, so these figures carry selection bias; executions
are at mid plus 3 bps; and the drawdown profile is the classic short-vol left
tail, dominated by a single 2020-scale event.

## Repository layout

```
src/heston/       pipeline modules (data, fft, calibration, surface,
                  signals, backtest, reporting)
tests/            pytest suite (77 tests, all offline — no WRDS required)
data/raw/         raw OptionMetrics pulls (gitignored, never modified)
data/processed/   calibrated parameters, signal panel, equity curve, trades
reports/          performance report and plots (gitignored)
docs/             notes and write-ups
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy matplotlib pyarrow pytest wrds
```

WRDS access: place a line for `wrds-pgdata.wharton.upenn.edu:9737` in
`~/.pgpass` (see WRDS documentation). Only `data.py` touches the network;
every other stage runs from the local parquet files.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## References

- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options.*
- Carr, P. & Madan, D. (1999). *Option Valuation Using the Fast Fourier
  Transform.*
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). *The Little
  Heston Trap.*
