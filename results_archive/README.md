# Results archive — primary evidence for every reported number

This folder is the version-controlled archive of the study's primary result artifacts.
Every number, figure, and table in the report traces to a file in here; the provenance
appendix of the results document cites paths relative to this folder. Files are byte-exact
checksummed copies of the working record (verified at archive time).

## Layout

- `qrm/` — the reactive-simulator track (QRM on per-order BTC data):
  - `step5_*/` — scored evaluations: `judgement.json` (paired costs, per seed, vs both
    TWAP benchmarks, with significance) + `behaviour_audit.json` (validity flags), one
    folder per campaign/cell. These are the source of truth for every cited cost number.
  - `runs_*/` — trained agents: one folder per run with `model.zip` (trained policy),
    `meta.json` (full configuration), `curve.json` (training-time monitor evaluations).
  - `step3g/` — environment calibration: the calibrated model bundles per regime,
    calibration reports, and the fairness verdicts (drift + constant-pace checks).
  - `step4_gates_v3.json` — the environment validation gates of record.
  - `per_episode_v3/` — per-episode cost arrays for the 20 primary-campaign agents
    (2,000 episodes per policy, exact-match verified against the sealed records).
  - `RESULTS_MANIFEST.md` — the track's own manifest of current vs superseded folders.
  - `SUPERSEDED_step5_v2/` — the pre-drift-fix campaign scores, retained ONLY because the
    drift-confound figure cites them as its "before" condition. Never cite as results.
- `l2/` — the frozen-replay track (historical L2 data):
  - `runs/`, `runs_10s/`, `runs_10s_10min/` — all 70 agents (`model.zip`,
    `normalizer.json`, `meta.json`, `curve.csv` per seed folder), one folder per dataset.
  - `size_sweep_results.json` — the benchmark cost-versus-size sweep.
  - `SUPERSEDED_stubs_ppo193_1min/` — the four early-terminated runs found by the
    completeness census, retained for the audit trail of that disclosure. Never cite.

## What is deliberately NOT here

Raw exchange data (order-book diffs, orders, trades; ~150 GB) and the derived training
datasets (parquet). Both are regenerable: the raw data comes from Hyperliquid's public
S3 archive (`s3://hl-mainnet-node-data`, December 2025), and the repository's pipeline
scripts rebuild the datasets and the calibration from it.

## Relationship to the working record

The working record lives outside the repository (`scratch_hyperliquid/`) and remains the
operational source for scripts. This archive is the frozen, citable snapshot; it is
append-only (new campaign results are added the same way, e.g. the frozen-replay sealed
exam when it runs) and existing files in it are never edited.
