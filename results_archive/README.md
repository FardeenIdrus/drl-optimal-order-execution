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
datasets (parquet). Both are regenerable: the two-year snapshot record comes from
Hyperliquid's public archive (`s3://hyperliquid-archive`), the December 2025 per-order
record from the "Open Book" Level 4 dataset (Zenodo DOI 10.5281/zenodo.18184441), and the
repository's pipeline scripts rebuild the datasets and the calibration from them.

## Relationship to the working record

The working record lives outside the repository (`scratch_hyperliquid/`) and remains the
operational source for scripts. This archive is the frozen, citable snapshot; it is
append-only (new campaign results are added the same way, e.g. the frozen-replay sealed
exam when it runs) and existing files in it are never edited.

## Phase F additions (2026-07-31) — the measured-signal extension and after

Everything below was added when the archive was found to be missing every campaign after
mid-July, which made this README's opening claim untrue at the time it was checked.

- `qrm/step5_signal_dev/`, `.../_reserve/`, `.../_curveblock/`, `.../_sealed/` — the
  measured-signal extension scored on its development, reserve, curve-monitoring and
  one-shot sealed blocks. Includes `diagnostics_postnull/` (exploiter ceiling, base-env
  reader, learning diagnostics).
- `qrm/step5_signal_ceiling21e6/` — the attainable-edge ceiling, confirmed one-shot on a
  freshly minted block (0.313 bps calm, 0.625 bps volatile).
- `qrm/step5_comparators/` — Almgren-Chriss and oracle VWAP in both environments, plus the
  risk-return frontier summary and the per-episode agent arrays behind it.
- `qrm/step5_signal_obsfix{,_var,_dqn}/`, `qrm/step5_primary_v3_obsfix/` — the four
  observation-specification amendments (A4, A4.1, A4.2, A4.3).
- `qrm/signal/` — the injection instrument's certification record: measurement, kernel
  solution, drift-correction diagnostics, and the gate suite that certified it.
- `l2/sealed_exam/` — the frozen-replay one-shot sealed exam (70 agents, three datasets),
  the validation re-check used as its control, and the three-stage inversion analysis that
  attributed the apparent edge.

**Trained agents are archived as `meta.json` + `curve.json` only — the `model.zip` weights
are deliberately excluded.** They add ~19 MB across the extension campaigns, no number in the
report cites them, and training is bit-reproducible from the recorded configuration and seed
(verified under Amendment A4: the same seed trained twice produced identical weights across
all 24 policy tensors). The configuration and the training curve are cited, so both are kept.
The agents of the primary and confirmation campaigns retain their weights, as before.

**Integrity.** `CHECKSUMS.sha256` covers every file in this archive. Regenerate and verify:

    .venv/bin/python results_archive/archive_phase_f.py
    cd results_archive && shasum -a 256 -c CHECKSUMS.sha256 | grep -v ": OK$"

Last run 2026-07-31: 1,446 files, 58.4 MB, zero mismatches.
