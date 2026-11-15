# BTC Cent Forecast 01 - Blended ML Entry

An advanced Node.js TypeScript entry for the `btc-cent-forecast-01` competition on `make.fun`.

## Overview
This program forecasts Bitcoin's US-dollar price 240 minutes (4 hours) ahead of a given checkpoint using 1-minute BTC/USDC closes up to the checkpoint (4,320 prices).

## The Blended Model
An equal-weight blend of three independent forecasting components (all log-returns in natural logs):

1. **Component A — 24h Shape Ridge Regression**:
   - Features: 25 numbers ($k=0..24$), representing $\ln(\text{price}[t - 1440 + k \cdot 60]) - \ln(\text{price}[t - 1440])$.
   - Training samples: every $t$ from 1440 to $n-241$ inclusive.
   - Standardization: mean and std (std floored at 1e-12) over training samples.
   - Ridge regression: fit with $\lambda = 1000$ (bias term unpenalized) using Gaussian elimination with partial pivoting.
   - Predicts log-return at $t = n-1$.

2. **Component B — 480-min Demeaned-Shape k-NN ($K=10$, median)**:
   - Tail shape: last 480 closes, demeaned and RMS-normalized.
   - Sliding window shapes ($s$ from 0 to $n-721$): demeaned and RMS-normalized 480-minute windows.
   - Distance: sum of squared differences against tail shape.
   - Prediction: median of forward returns ($\ln(\text{price}[s+720] / \text{price}[s+480])$) of the 10 closest historical windows (breaking ties by smaller $s$).

3. **Component C — 60-Minute Momentum**:
   - $\ln(\text{last} / \text{price}[n-61])$.

## Blend & Fallbacks
- Predicted log-return = average of available component log-returns ($A, B, C$), clamped to $[-0.05, +0.05]$ as a safety guard against wild extrapolations on unseen data.
- `forecast = last * exp(predicted log-return)`, clamped to $[0.00001, 9999999.99999]$, rounded to 5 decimals.
- Robust guardrails and fallbacks for data availability and non-finite values.

## Running Locally
To test the entry against the kit's practice dataset:

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=/home/hatch/workspace/makefun/btc-cent-forecast-01-kit/btc-cent-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out \
node main.ts
```
