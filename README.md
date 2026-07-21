# Deep Reinforcement Learning for Optimal Order Execution on Real Hyperliquid Order-Book Data

MSc dissertation project. Reinforcement-learning agents (PPO and DQN) execute a large BTC buy
order and are benchmarked against TWAP scheduling on implementation shortfall, using real
Hyperliquid BTC-USD order-book data (December 2025). The study runs on two complementary
experimental tracks. The reactive-simulator track is the main focus: agents trade inside a
queue-reactive market model calibrated on per-order (L4) data, so their own orders consume
liquidity and move the price. The frozen-replay track is the secondary strand: agents trade
against replayed historical snapshots that cannot react, and its chief role is to motivate
the move to a reacting market.

**Headline result to date:** a boundary null. On the reactive track, apparent advantages over
TWAP appeared repeatedly on the evaluation data used during development and failed every test
on fresh data: two one-shot sealed confirmations failed, and the two surviving cells of a
16-cell robustness grid died on a never-before-used replication block. Documenting how those
spurious edges arose, and the pre-registered evaluation machinery that caught them, is a
central contribution. A separately diagnosed finding: DQN systematically collapses into
inaction at realistic order sizes where PPO trains healthily.

## Project status (21 July 2026)

- Reactive-simulator track: experimentally complete. Primary campaign (20 agents), tuning
  and selection (98 runs), two sealed confirmations (both failed), 16-cell size-by-deadline
  robustness grid with a five-seed escalation and replication ladder (both triggered cells
  failed replication), DQN collapse diagnosis (cross-setting probe, update-rhythm variant,
  learned-value inspection), per-episode cost distributions.
- Frozen-replay track: all 70 agents trained and validated (14 arms: dataset by algorithm by
  order size, five seeds each). The one-shot sealed test-set evaluation has not yet run.
- Next experiment: the measured-signal extension (measure the order-flow to future-return
  relationship from the data and inject exactly that into the simulator, unifying the two
  tracks' mechanisms in one environment). Registered as a next step in the results document.
- The current results document lives at `reports/results_pack/results_pack.pdf`.

## Repository layout

```
src/execution/                All project code (one package).
  pipeline.py                 Config-driven orchestrator for the frozen-replay data
                              pipeline (idempotent, deterministic).

  data/                       Frozen-replay data pipeline: raw S3 archive to frozen datasets.
    manifest.py, pull.py      S3 availability listing and raw download.
    schema.py, parse.py       Canonical order-book contract; raw to canonical conversion.
    sources/                  Hyperliquid-specific listing and decoding.
    resample.py               Bar-resolution books plus intra-bar statistics.
    features.py               Causal microstructure features (leakage-tested).
    regimes.py, dataset.py    Calm/volatile episode labels; frozen train/val/test tables.
    adv.py                    Daily volume, used to size orders as a share of it.
    qa_report.py              Pipeline quality-assurance report.

  data/l4/                    Per-order (L4) data to a validated reconstructed book.
    book_diffs_reader.py      Streams order-level book-diff events.
    orders_reader.py          Order-status file: timestamps and terminal removals.
    trades_reader.py          Trades file: authoritative market-order events.
    book_engine.py            Order-by-order book reconstruction with a crossing guard.
    snapshot_sampler.py       0.5-second snapshot grid over the event stream.
    reconstruct_month.py      Month-scale driver (checkpointed, memory-flat).
    validate_vs_l2.py         Cross-validation of the reconstruction against the
                              independent snapshot archive.

  qrm/                        The reactive simulator and every reactive-track experiment.
    event_labeler.py          Classifies events: limit arrival, cancellation, market order.
    ref_frame.py              Reference price frame; depth and queue state.
    quiet_spell.py            Quiet-spell-conditioned calibration measurements.
    calibrate_intensities.py  Earlier unconditioned calibrators (kept for comparison).
    assemble.py               Packages calibrated rates, book shapes and sizes for the engine.
    exo_ref_sim.py            The two-component simulator: calibrated queue dynamics plus
                              an empirically measured, drift-neutralised background
                              price-move process.
    step3f.py, step3g.py      Calibration drivers: validation gate; per-regime calibration
                              (calm and volatile) with the drift-neutrality fairness gate.
    step4_gates.py            Pre-training environment validation gates (impact is real and
                              persistent, costs size-monotone, benchmark sanity).
    reactive_env.py           The reactive execution environment (Gymnasium).
    reactive_baselines.py     Fixed TWAP and adaptive TWAP benchmarks inside the simulator.
    train_reactive.py         Trains one agent (PPO or DQN) in the reactive environment;
                              writes model, configuration record and training curve.
    step5_judgement.py        Paired common-random-numbers evaluation of trained agents
                              against both TWAP benchmarks; behaviour audit before any cost
                              comparison; writes the judgement and audit records that are
                              the source of truth for every reported number.
    vendored.py               Import helper for the vendored engine below.

  env/                        Frozen-replay execution environment.
    fills.py                  Ask-ladder fill engine, shared by agents and benchmarks.
    real_data_env.py          The replay environment; reward is implementation shortfall.
    benchmarks.py             TWAP and related schedule benchmarks.
    calibration.py            Impact fits used by benchmark calibration.
    episode_store.py          Chronological train/val/test episode split (leakage-free).
    normalize.py              Train-only feature normalisation.

  agents/                     DQN and PPO (Stable-Baselines3) for the frozen-replay track:
                              model builders, policies, callbacks, training entry point.
  eval/                       Frozen-replay paired per-episode scoring and decomposition.

configs/                      Experiment and pipeline configuration files, one pair per
                              frozen-replay dataset (1-minute/30-min, 10-second/30-min,
                              10-second/10-min). Single source of truth for settings.

tests/                        219 deterministic pytest unit tests: leakage, fills, book
                              reconstruction, calibration, reactive environment, benchmark
                              parity, guard regressions.

reports/                      Protocols, analysis outputs and the results document.
  qrm_step4_criteria.md       The frozen, pre-registered decision rules for the reactive
                              track and every protocol amendment, dated before the runs
                              they govern.
  l2_test_protocol.md         Pre-registered protocol for the frozen-replay track's
                              one-shot sealed test-set evaluation (not yet executed).
  figures/                    Figure generation. qrm/make_figures.py and
                              l2/make_l2_figures.py rebuild every figure from the primary
                              result records.
  tables/make_tables.py       Rebuilds every LaTeX table from the primary result records.
  diagnostics/                DQN learned-value diagnostic and the per-episode
                              re-evaluation script, with their outputs.
  results_pack/               The results document. results_pack.tex/.pdf is the full
                              working draft of the results and discussion chapter;
                              *_overleaf.tex are self-contained single-file copies;
                              meeting_pack.* is a figure-by-figure walkthrough version;
                              figures/ and t*.tex are the embedded assets.

results_archive/              Version-controlled primary evidence for every reported
                              number: scored evaluations, behaviour audits, environment
                              gates, calibration bundles, per-episode cost arrays, and
                              every trained model with its configuration and training
                              curve, for both tracks (checksummed copies). See its
                              README.md for the layout. The provenance appendix of the
                              results document cites paths relative to this folder.

qrm_optimal_execution/        Vendored queue-reactive-model scaffold (Huang, Lehalle and
                              Rosenbaum 2015; implementation arXiv 2511.15262, MIT
                              licence), unmodified.

docs/data_dictionary.md       Data dictionary for the pipeline outputs.
```

Bulk data lives outside the repository in a gitignored scratch directory: raw archives,
reconstructed books and training datasets (about 160 GB). All of it is regenerable from
public sources with the pipeline code above; `results_archive/README.md` states the sources.

## Where to find what

- The results document: `reports/results_pack/results_pack.pdf`.
- The evidence behind any reported number: `results_archive/` (start from the results
  document's provenance appendix, which maps every figure and table to its source file).
- The pre-registered decision rules: `reports/qrm_step4_criteria.md` (reactive track) and
  `reports/l2_test_protocol.md` (frozen-replay sealed exam).

## Data sources

- Frozen-replay track: Hyperliquid public S3 snapshot archive (requester-pays), BTC.
- Reactive track: Oxford "Open Book" Hyperliquid L4 dataset (Zenodo,
  DOI 10.5281/zenodo.18184441, CC BY 4.0), December 2025 BTC.

## Setup and key commands

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# test suite (219 tests, deterministic)
PYTHONPATH=src .venv/bin/pytest tests -q

# frozen-replay data pipeline (idempotent; skips completed stages)
PYTHONPATH=src .venv/bin/python -m execution.pipeline \
    --config configs/pipeline.yaml --scratch-root <scratch>

# reactive track: train one agent / judge a set of runs (see --help for options)
PYTHONPATH=src .venv/bin/python -m execution.qrm.train_reactive --help
PYTHONPATH=src .venv/bin/python -m execution.qrm.step5_judgement --help

# rebuild every figure and table from the primary result records
# (scripts read the working record in the scratch directory; results_archive/ is its
#  checksummed snapshot inside the repository)
.venv/bin/python reports/figures/qrm/make_figures.py
.venv/bin/python reports/figures/l2/make_l2_figures.py
.venv/bin/python reports/tables/make_tables.py
```

## Limitations

Stated fully in the results document. In brief: execution is market-order-only; the reactive
simulator's background price process is deliberately unpredictable, so the null concerns
liquidity-timing and impact management rather than price prediction (the measured-signal
extension addresses this); one asset, one venue, one calendar month, two within-month
regimes.
