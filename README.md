# Deep RL for Optimal Order Execution — Regime-Adaptive Strategies

MSc dissertation project. Trains reinforcement-learning agents (DQN, PPO) to execute a large
order over a 30-minute horizon on **real Hyperliquid Level-2 order-book data**, benchmarks
them against TWAP/VWAP on implementation shortfall, and asks **which microstructure signal
drives the RL advantage — and whether that driver changes between calm and volatile regimes**
(per-regime SHAP attribution + ablation).

Built on the Queue-Reactive Model execution codebase (arXiv 2511.15262, MIT-licensed,
vendored under `qrm_optimal_execution/`) as a scaffold, fed real data instead of the
simulator. New code is in `src/execution/`.

## Status
**Phase 1 (data pipeline): complete.** A deterministic, leakage-checked pipeline turns ~15 GB
of raw Hyperliquid L2 (BTC, Jan 2024–Dec 2025) into frozen, regime-labelled train/test
execution datasets. See `reports/phase1_qa.md` for the evidence and `docs/data_dictionary.md`
for the schema. Phase 2 (RL environment + agents) is next.

## Repository layout
```
src/execution/            project code (own package)
  data/                   Stage 0-6 data pipeline
    manifest.py           Stage 0  S3 availability manifest
    pull.py               Stage 1  download raw L2 (.lz4) from the Hyperliquid archive
    schema.py             canonical order-book schema (source-agnostic contract)
    sources/              source adapters (Hyperliquid L2 listing + decode)
    parse.py              Stage 2  raw -> canonical book
    resample.py           Stage 3  -> per-minute book + high-frequency stats
    features.py           Stage 4  -> microstructure features (causal, no look-ahead)
    regimes.py            Stage 5  -> episodes + calm/volatile regime labels
    dataset.py            Stage 6  -> frozen train/test datasets
    qa_report.py          Stage 7  consolidated QA report
  pipeline.py             Stage 7  config-driven orchestrator (idempotent)
configs/pipeline.yaml     all pipeline parameters (single source of truth)
tests/                    pytest unit tests (incl. no-look-ahead / leakage tests)
docs/, reports/           data dictionary + Phase 1 QA evidence
qrm_optimal_execution/    vendored QRM scaffold (read with care; MIT, cite the paper)
```
Data artifacts live **outside the repo** in a scratch directory (gitignored).

## Setup
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
The pipeline reads raw data from the Hyperliquid native archive (`s3://hyperliquid-archive`,
requester-pays) via the AWS CLI; configure read-only AWS credentials before running Stage 1.

## Reproduce Phase 1
```bash
# one config + one command reproduces Stages 0-6 (idempotent: skips completed stages)
PYTHONPATH=src .venv/bin/python -m execution.pipeline \
    --config configs/pipeline.yaml --scratch-root <path-to-scratch>

# regenerate the consolidated QA evidence
PYTHONPATH=src .venv/bin/python -m execution.data.qa_report \
    --config configs/pipeline.yaml --scratch-root <path-to-scratch> --out-dir reports

# run the test suite
PYTHONPATH=src .venv/bin/pytest tests -q
```
The pipeline is deterministic: the same raw data and config produce identical outputs.

## Limitations (honest)
Minute-resolution execution (one book snapshot per minute for fills; no intra-minute
replenishment); Hyperliquid BTC perpetual futures, single asset; the fixed train-median
regime threshold means the test regime mix reflects the test period. See `reports/phase1_qa.md`.
