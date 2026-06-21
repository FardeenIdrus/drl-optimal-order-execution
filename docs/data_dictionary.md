# Data Dictionary — train/test execution datasets

The Stage 6 outputs (`train.parquet`, `test.parquet`) are the data the RL environment
consumes. One row = one decision step within an execution episode. All prices in USD,
sizes in BTC, timestamps in milliseconds since the Unix epoch (UTC).

**Decision-time alignment (no look-ahead).** The row at decision minute `t` contains
observation features computed from bars `<= t-1` (Stage 4 shift) and the order book
*as-of* `t`, which is the previous minute's end-of-minute book (`minute[t-1]`). Both sides
carry a uniform one-bar lag, so nothing in a row depends on minute `t` itself or any
future minute.

## Keys
| Column | Type | Meaning |
|---|---|---|
| `episode_id` | int | Episode identifier (a 30-minute, clock-aligned, non-overlapping execution window). |
| `step` | int | Decision index within the episode, 0–29. |
| `ts` | int64 | Decision time (minute boundary), ms epoch UTC. |
| `regime` | str | `calm` or `volatile` — labelled by the episode's ex-post realised vol vs the train-median threshold. |

(`split` is implicit: train rows are in `train.parquet`, test rows in `test.parquet`.)

## Observation features (what the agent sees; raw, un-normalised)
| Column | Units | Definition |
|---|---|---|
| `spread_bps` | basis points | Mean intra-minute spread / mid × 1e4. |
| `imbalance` | [-1, 1] | (bid_depth − ask_depth)/(bid_depth + ask_depth) over top-5 (HF mean). |
| `recent_return` | log-return | log(mid_t / mid_{t−5}) over the prior 5 minutes. |
| `rolling_vol` | vol | sqrt(Σ intra-minute realised variance over the prior 30 minutes). |
| `ask_depth` | BTC | Mean available top-5 ask size (absolute liquidity the buy-only agent consumes). |

The agent's full state also includes **execution state — inventory remaining and
time-remaining — added by the environment at runtime**, not stored here.

## Fill book (as-of the decision; used by the env's fill engine, not part of the observation)
| Column | Units | Meaning |
|---|---|---|
| `mid` | USD | Mid-price as-of the decision (= minute[t−1] mid). |
| `bid_px_1`, `bid_sz_1` | USD, BTC | Best bid price and size. |
| `ask_px_1` … `ask_px_20` | USD | Ask ladder prices, level 1 (best) to 20, ascending. |
| `ask_sz_1` … `ask_sz_20` | BTC | Ask ladder sizes per level (the depth a buy order walks down). |

## Notes
- Episodes are kept only if all 30 minutes are present and feature-valid (no fabricated or
  gap-spanning data).
- `realized_vol` (the episode regime label) is computed over the episode window and is
  deliberately distinct from the `rolling_vol` feature (which is trailing and pre-decision),
  to avoid circularity between the regime definition and a feature.
- Single-snapshot minutes (no intra-minute return) contribute zero to realised variance.
