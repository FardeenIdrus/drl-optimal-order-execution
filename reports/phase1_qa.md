# Phase 1 Data Pipeline — QA Report (BTC, 2024-01-01 to 2025-12-31)

Real Hyperliquid L2 order-book data; deterministic, leakage-checked pipeline.

## Source & coverage (Stage 0)
- Hourly coverage: **17107/17232 (99.27%)**; download 14.72 GiB.
- Fully-covered days: 703; missing: 3; partial: 12.

## Pull integrity (Stage 1)
- Raw files on disk: **17107** vs expected 17107 (byte-exact: True).

## Minute book + HF stats (Stage 3)
- Minutes: **1020217**; duplicates: 0; coverage 99.4%; book ordering OK: True; median snapshots/min: 106.0.

## Features (Stage 4)
- Feature-valid rows: **995572 (94.58%)**; params {'imbalance_depth': 5, 'return_lookback': 5, 'vol_window': 30}.

| corr | spread_bps | imbalance | recent_return | rolling_vol | ask_depth |
|---|---|---|---|---|---|
| spread_bps | 1.00 | -0.01 | -0.06 | 0.48 | -0.11 |
| imbalance | -0.01 | 1.00 | 0.21 | 0.01 | -0.48 |
| recent_return | -0.06 | 0.21 | 1.00 | 0.00 | -0.08 |
| rolling_vol | 0.48 | 0.01 | 0.00 | 1.00 | -0.10 |
| ask_depth | -0.11 | -0.48 | -0.08 | -0.10 | 1.00 |

## Regimes + coverage gate (Stage 5)
- Episodes: **32758**; threshold: {'median': 0.00163781805766557}.

| split | calm | volatile |
|---|---|---|
| test | 4596 | 1955 |
| train | 13104 | 13103 |

- Coverage gate (min test-regime episodes): **1955** -> both regimes amply represented.

## Train/test datasets (Stage 6)
- train: **786210 rows / 26207 episodes**; test: **196530 rows**.
- Leakage checks: train_before_test=True, NaN in core fields=0, book ordering OK=True.

## Reproducibility & limitations
- Phase 1 has no randomness; same raw data + config => identical outputs.
- Limitation: Minute resolution; one book snapshot per minute for fills (no intra-minute replenishment).
- Limitation: Hyperliquid BTC perpetual futures (not spot/equities); single asset for now.
- Limitation: Fixed train-median regime threshold => test regime mix reflects the test period (here ~70/30 calm/volatile).
