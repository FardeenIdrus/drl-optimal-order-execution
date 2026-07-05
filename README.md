# Deep RL for Optimal Order Execution on Real Hyperliquid Data

MSc dissertation project. Reinforcement-learning agents (DQN, PPO) execute a large order on
**real Hyperliquid BTC order-book data** and are benchmarked against TWAP and Almgren-Chriss
on implementation shortfall. The research question: **can an RL agent beat standard execution
schedules on real data, and, if so, which microstructure signal drives that edge and does the
driver change between calm and volatile regimes** (per-regime SHAP attribution + ablation)?
The completed replay study answers the first part in the negative (a robust null, mechanism
quantified); the active track tests whether modelling the book's reaction to the agent
changes that answer. No outperformance is assumed or claimed.

Built on the Queue-Reactive Model execution codebase (Huang, Lehalle & Rosenbaum 2015;
implementation arXiv 2511.15262, MIT-licensed, vendored **unmodified** under
`qrm_optimal_execution/`). All project code is in `src/execution/`.

## The two experimental tracks (read this first)

The repository contains two distinct experiments with **separate data sources and separate
code paths**. Understanding the split is essential to navigating the code.

### Track 1: L2 replay study (complete; result = a robust null)

- **Data:** Hyperliquid L2 order-book *snapshots* (20 levels/side, ~0.5 s cadence, BTC,
  Jan 2024 - Dec 2025, ~15 GB) from the public S3 archive.
- **Design:** the agent trades against replayed historical books. Fills walk the real ask
  ladder (direct impact is real), but the book never reacts to the agent: the next step is
  the next historical snapshot regardless of what the agent did.
- **Result:** neither DQN nor PPO beats TWAP, and the null is robust across **two decision
  resolutions** (60 s, 10 s), **three order sizes** (0.5%, 1%, 2% of daily volume), and
  **two horizons** (30 min, 10 min), 5 seeds each, with an anti-bias audit (fair action
  space, planted-edge detection power, correct reward sign). The measured mechanism: on
  BTC's deep book the cost lever a schedule controls (~0.6 bps) sits ~25x below per-episode
  price-drift noise (~16 bps), and the frozen book removes the market reaction an RL agent
  would exploit. The held-out L2 test set remains sealed.

### Track 2: L4 + Queue-Reactive Model (active)

- **Data:** Oxford "Open Book" Hyperliquid L4 dataset (Dec 2025, ~70 GB compressed): every
  individual order event (placements, cancellations, fills) plus a trades file.
- **Design (the fix for Track 1's structural flaw):** calibrate
  the Queue-Reactive Model on real per-order events, so the *simulated* book genuinely
  reacts to order flow, then train and evaluate the agents inside it on a disjoint data
  subset. The agent's fills then deplete queues, alter event rates, and can move the price:
  the endogenous impact channel that replay cannot represent.
- **Extension to the base model (documented, paper-anchored):** the base QRM moves price
  only when a queue empties, which provably freezes on BTC (price motion there is
  information-driven, ~7 ticks/s). Following Huang et al.'s own Model II/III direction, an
  **empirically measured exogenous reference-move process** (per-0.5 s jump distribution
  from the reconstructed book) drives the vendored engine's move machinery, while the
  calibrated queue dynamics supply the reaction. Event rates are measured on **quiet spells
  only** (reference price stable for >500 ms; guard length measured from the post-move
  re-quoting burst: 240 arrivals/s in the first 50 ms vs 31.5/s steady) so that no real
  behaviour is double-counted between the two components.

## Repository layout

```
src/execution/                project code (own package)
│
│  ── shared / Track-1 (L2 replay) ──────────────────────────────────────────
│  data/                      L2 pipeline, Stages 0-6 (raw S3 -> frozen datasets)
│    manifest.py  pull.py     S3 availability + raw download
│    schema.py  parse.py      canonical book contract; raw -> canonical
│    sources/                 Hyperliquid-specific listing/decoding (only source-aware code)
│    resample.py              -> bar-resolution book + intra-bar high-frequency stats
│    features.py              -> causal microstructure features (leakage-tested)
│    regimes.py  dataset.py   -> calm/volatile episode labels; frozen train/test tables
│    adv.py                   daily volume -> order sizing (% of ADV)
│  pipeline.py                config-driven orchestrator (idempotent, deterministic)
│  env/                       replay execution environment (Gymnasium)
│    fills.py                 ask-ladder fill engine (shared by agents AND benchmarks)
│    real_data_env.py         the replay env; reward = implementation shortfall vs arrival
│    benchmarks.py            TWAP, Almgren-Chriss (real-book-calibrated), fixed-liquidity
│    calibration.py           impact fits (linear eta for AC; sqrt deadline residual)
│    episode_store.py         chronological train/val/test split (leakage-free)
│    normalize.py             train-only feature normalisation
│  agents/                    DQN + PPO (Stable-Baselines3), shared architecture
│  eval/                      paired per-episode scoring; three-baseline decomposition
│                             (fixed-TWAP vs adaptive-TWAP vs agent), Wilcoxon + bootstrap
│
│  ── Track-2 (L4 + QRM) ────────────────────────────────────────────────────
│  data/l4/                   raw L4 events -> validated reconstructed book
│    book_diffs_reader.py     stream order-level diff events; inject removes/trades
│    orders_reader.py         binary order-status file -> timestamps + terminal removes
│    trades_reader.py         trades file -> authoritative market-order events
│    book_engine.py           order-by-order book reconstruction + evidence-based
│                             crossing guard (evicts the STALER side of a crossed pair)
│    snapshot_sampler.py      0.5 s snapshot grid over the event stream
│    reconstruct_month.py     month-scale driver (checkpoint/resume, memory-flat)
│    validate_vs_l2.py        cross-validation against the independent L2 archive
│  qrm/                       reconstructed book -> calibrated reactive simulator
│    event_labeler.py         classify events: limit arrival / cancel / market
│    ref_frame.py             sticky reference price; frame-consistent depth/queue state
│    quiet_spell.py           post-move burst measurement; quiet-spell-conditioned
│                             calibration + same-clock trade-volume anchor
│    calibrate_intensities.py earlier (unconditioned) calibrators, kept for comparison
│    assemble.py              package rates/book-shapes/AES into the engine's format
│    exo_ref_sim.py           the two-component simulator: endogenous queue dynamics
│                             + empirical exogenous reference moves
│    step3f.py                driver for the pre-registered validation gate (see below)
│    vendored.py              import path helper for the untouched vendored engine
│
configs/                      pipeline + experiment configs (single source of truth)
tests/                        209 pytest unit tests (leakage, fills, reconstruction,
                              guard regressions, quiet-spell conditioning; deterministic)
reports/                      QA evidence; qrm_3f_criteria.md = the FROZEN gate protocol
docs/                         data dictionary
qrm_optimal_execution/        vendored QRM scaffold (unmodified; cite the paper)
```

Data artifacts live **outside the repo** in a gitignored scratch directory
(`scratch_hyperliquid/`: L2 raw + datasets; `oxford_l4/`: L4 archives, reconstructed book,
calibration bundles).

The only deliberate crossover between tracks is `data/l4/validate_vs_l2.py` (the L4
reconstruction is checked against the independent L2 feed). At Step 4 the tracks reconnect
by design: the QRM becomes the market model behind the same `agents/` and `eval/` machinery,
so Track-2 results are scored identically to Track 1.

## What has been done

1. **L2 pipeline, environment, benchmarks, agents (Track 1)**: deterministic 7-stage
   pipeline with leakage tests; replay environment with one fill engine shared by agents
   and benchmarks; TWAP / Almgren-Chriss (calibrated from the real book, per regime) /
   fixed-liquidity-participation; DQN + PPO.
2. **The null result, made robust**: no RL edge over TWAP across resolution, size, and
   horizon; mechanism quantified (drift noise ~25x the schedule lever); degenerate DQN
   collapse identified and separated from PPO's valid null; evaluation via paired
   per-episode tests against both fixed-TWAP and adaptive-TWAP baselines.
3. **L4 book reconstruction (Track 2, Step 2)**: the full December 2025 BTC book rebuilt
   from ~1B individual order events; validated against the independent L2 archive
   (mid-price correlation 0.9999+, zero crossed books). A silent corruption on quiet days
   (frozen top-of-book caused by a directional flaw in the crossing guard interacting with
   orders whose removals are absent from the source data) was later detected by a
   month-wide scan, root-caused, fixed (evidence-based guard; end-of-hour retry for a
   cancel-vs-placement timing race), and the month re-reconstructed.
4. **QRM calibration (Step 3, iterations 1-2)**: event labelling, intensity measurement in
   the engine's own reference frame, data-driven resolution (K=5 levels, Q chosen from
   observed queue-size exposure), empirical book shapes, trades-file volume anchor; the
   frozen-price failure of the base model diagnosed and resolved with the empirical
   exogenous move process (price volatility 0 -> ~90% of real).
5. **Quiet-spell recalibration (Step 3f, current iteration)**: pre-registered protocol
   (`reports/qrm_3f_criteria.md`, frozen before running); the post-move re-quoting burst
   measured and confirmed (the evidence for the guard); event rates, book shapes, and the
   volume anchor re-measured on quiet spells only, on a single consistent clock; held-out
   gate-hour targets measured.

## What is in progress

- **The 3f validation gate**: 8-seed simulations of the calibrated model compared against a
  held-out hour the calibration never saw (within-day holdout: calibrate Dec-1 h00-02, gate
  h03), judged on price volatility, spread, inner-book emptiness, event mix, and traded
  volume, with tolerance bands derived from the real data's own calibration-window to
  gate-window drift across all 31 days. Pass/fail criteria and the decision rule were
  frozen in advance; blocked only on the month re-reconstruction finishing.

## What remains to be implemented

1. **Step 3g**: regime-conditional calibration: separate calm and volatile QRMs (queue
   rates AND the exogenous move process per regime; hourly realised-volatility median
   split, chronological calibrate/validate), feeding the dissertation's per-regime question.
2. **Step 4**: wire the calibrated QRM in as the RL market model (replacing frozen replay),
   re-run benchmarks inside it, and print the pre-training gates (does a realistic order
   visibly move the book; does the simulator reproduce held-out statistics; are both
   regimes represented).
3. **Step 5**: train DQN + PPO (5 seeds) inside the reactive simulator on the disjoint
   subset, scored through the same three-baseline decomposition as Track 1. The key
   question: does modelling the market's reaction change the Track-1 null?
4. **If an edge emerges**: the original contribution: per-regime SHAP + ablation
   attribution of the driver. **If the null persists**: a materially stronger,
   reaction-inclusive negative result, positioned against the simulator-based literature.
5. **Write-up** (UCL structure, positioned against arXiv 2511.15262 and the Huang et al.
   QRM paper).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Track 1 reads raw data from the Hyperliquid native archive (`s3://hyperliquid-archive`,
requester-pays) via the AWS CLI. Track 2 reads the Oxford "Open Book" archives (Zenodo,
DOI 10.5281/zenodo.18184441, CC BY 4.0), downloaded once into the scratch directory.

## Reproduce

```bash
# Track 1: the full L2 pipeline (idempotent; skips completed stages)
PYTHONPATH=src .venv/bin/python -m execution.pipeline \
    --config configs/pipeline.yaml --scratch-root <scratch>

# Track 2: month reconstruction (checkpointed; ~7 h)
PYTHONPATH=src .venv/bin/python -m execution.data.l4.reconstruct_month \
    --diffs-dir <scratch>/oxford_l4/diffs_extract \
    --orders-dir <scratch>/oxford_l4/orders_extract \
    --out <scratch>/oxford_l4/book_05s_v2 --cadence-ms 500

# Track 2: the pre-registered 3f protocol, in order
PYTHONPATH=src .venv/bin/python -m execution.qrm.step3f burst        --scratch <scratch>/oxford_l4
PYTHONPATH=src .venv/bin/python -m execution.qrm.step3f calibrate    --scratch <scratch>/oxford_l4
PYTHONPATH=src .venv/bin/python -m execution.qrm.step3f gate-targets --scratch <scratch>/oxford_l4
PYTHONPATH=src .venv/bin/python -m execution.qrm.step3f tolerances \
    --book-dir <scratch>/oxford_l4/book_05s_v2 --out <scratch>/oxford_l4/step3f/tolerances.json
PYTHONPATH=src .venv/bin/python -m execution.qrm.step3f gate \
    --scratch <scratch>/oxford_l4 --book-dir <scratch>/oxford_l4/book_05s_v2 \
    --tolerances <scratch>/oxford_l4/step3f/tolerances.json

# test suite (209 tests, deterministic)
PYTHONPATH=src .venv/bin/pytest tests -q
```

## Honest limitations (carried into the write-up)

- **Track 1 (replay)**: direct impact is real (fills walk the recorded ladder) but the
  market never reacts to the agent; this is the documented counterfactual-feedback
  limitation and the motivation for Track 2.
- **Track 2 (QRM)**: simulated price volatility matches reality partly by construction
  (the measured move distribution is injected), so validation weight rests on the
  statistics that are NOT injected (spread, book emptiness, event mix, volume) and on
  held-out comparison. ~95% of never-removed orders lack any removal record in the source
  data; the evidence-based crossing guard contains their effect, and the residual
  calibration bias direction (touch slightly too liquid) understates the agent's impact,
  i.e. is conservative for the research question. One calendar month (Dec 2025) bounds the
  regimes the simulator can reproduce; checked at the 3g gate.
- Single asset (BTC perpetual), single venue (Hyperliquid).
