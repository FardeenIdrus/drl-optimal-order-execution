# FIGURES + TABLES MANIFEST (started 2026-07-13)

The single tracker for every figure/table/visualisation: built, pending, and candidate.
Policy (user + Carlo, 2026-07-13): COMPREHENSIVE-FIRST — if a result is important or useful,
generate it now; curation down to the report's final set happens ONCE, at the end. Do not
pre-trim. Update this file whenever a figure/table is added, regenerated, or its status changes.

FOLDER STRUCTURE (refactored 2026-07-14):
  reports/figures/qrm/make_figures.py   -> QRM figures (fig1-8 built) + PDF/PNG
  reports/figures/l2/  make_l2_figures.py (to build) -> L2 figures
  reports/tables/      make_tables.py (to build) -> booktabs .tex tables
  reports/figures/FIGURES_TABLES_MANIFEST.md = this tracker (both tracks).
All numbers read from source-of-record JSONs — never hand-typed. Styling upgrade planned:
scienceplots style for uniformity; seaborn for distribution plots.

## FIGURES — QRM / reactive track (main focus)

| ID | Title / content | Status | Source data | Notes |
|---|---|---|---|---|
| F1 | Dev vs sealed collapse (both champions) | BUILT v4 | step5_selection_v3, step5_confirm_v3a, step5_confirm_v1b | headline figure |
| F2 | Size-response (edge does not scale with size) | BUILT v5 (2026-07-14: benchmark label outside right, centred on zero line) | step5_sweep_b5/b12/b50 + selection | |
| F3 | Tuning forest, volatile (12 variants) | BUILT v4 | step5_selection_v3 | whisker = seed range (mixed n) |
| F4 | Three-block sign-flip | BUILT v5 (2026-07-14: benchmark label outside right, centred on zero line) | curve.json finals + step5_v3/selection/confirms | block-luck exhibit |
| F5 | Learning curves, volatile (base + selected) | BUILT v5 (2026-07-14: benchmark label outside right, centred on zero line) | curve.json | monitor-block context in caption |
| F6 | Drift confound before/after | BUILT v1 | SUPERSEDED_step5_v2 + step5_v3 | "manufactured edge" exhibit |
| F7 | DQN collapse audit scatter | BUILT v3 (2026-07-14: zoom is now its own SIDE PANEL — the overlaid inset had covered the dqn point at (50,29); filled invalid markers get a white edge so the two near-coincident points at (87,86.5)/(87,89.5) read as distinct squares; true positions, nothing nudged) | step5_v3 behaviour_audit | |
| F8 | Regime comparison calm vs volatile (RQ2) | BUILT v1 | step5_v3 + step5_selection_v3 | |
| F9 | Grid heatmap 4 sizes x 4 horizons, per regime | PENDING (grid training 24/66) | step5_grid_* + step5_sweep_* + selection | |
| F10a | Absolute execution costs per policy/regime (descriptive anchor) | PENDING (needs per-episode re-eval) | re-eval dump | Carlo descriptive stats |
| F10b | Per-episode cost distributions (violin/density), agent vs both TWAPs | PENDING (same re-eval) | re-eval dump | seaborn |
| F11 | Intra-day / intra-week volume + liquidity profiles | PENDING (liquidity study) | L4 trades/book | Carlo request; feeds VWAP |
| F12 | Simulator realism: self-impact ratio, dump cost, book recovery; calibration fit | PENDING | gate records (step4, step3g) | methods chapter |
| F13 | Measured order-flow -> future-return curve (magnitude/horizon/decay, per regime) | PENDING (extension step 1) | L4 measurement | gold either way |
| F14 | Tuning forest, CALM (12 variants) | CANDIDATE (build next batch) | step5_selection_v3 | completes F3 |
| F15 | Behaviour-audit scatter for ALL 98 tuning runs (F7 style) | CANDIDATE | step5_selection_v3 audit | |
| F16 | Action-distribution profiles: what the agent actually does per regime (pace histogram) | CANDIDATE | behaviour_audit action_shares | feeds RQ3 |
| F17 | Fairness gate: pace-multiple cost gradient before/after drift fix | CANDIDATE | step3g fairness records | methods |
| F18 | Sealed-confirmation per-seed detail (both tests, both benchmarks) | CANDIDATE | step5_confirm_* | may fold into T3 |

## FIGURES — L2 / frozen-replay track (secondary chapter; full set per comprehensive-first)

| ID | Title / content | Status | Source data | Notes |
|---|---|---|---|---|
| L1 | Three-axis null summary (resolution x size x horizon in one panel) | PENDING | scratch_hyperliquid/runs, runs_10s, runs_10s_10min | the L2 finding |
| L2 | Size-sweep lever + defensibility boundary (benchmarks; agent flat as lever grows) | PENDING | size_sweep_results.json (+ existing size_sweep_defensibility.png as reference) | mechanism figure |
| L3 | Anti-bias validation: planted synthetic edge IS detected | PENDING | L2 audit artifacts | methodological credibility |
| L4 | L2 learning curves (PPO/DQN) | CANDIDATE | L2 run logs | |
| L5 | L2 DQN collapse evidence | CANDIDATE | L2 audit artifacts | mirrors F7 |

## TABLES (all emitted as booktabs .tex by make_tables.py — TO BUILD)

| ID | Table | Status | Source |
|---|---|---|---|
| T1 | Primary campaign: all 20 runs per-seed, both benchmarks, p, audit flag | BUILT 2026-07-14 (compiled OK; numbers verified vs source) | step5_v3 |
| T2 | Tuning/selection: all 28 groups (seeds, valid, pooled, across-seed p) | BUILT 2026-07-14 (compiled OK) | step5_selection_v3 |
| T3 | Both sealed confirmations: per-seed + pass/fail criteria checklist | BUILT 2026-07-14 (compiled OK; verified) | step5_confirm_v3a/v1b |
| T4 | Robustness grid: 4x4 pooled + p per regime | PENDING (grid) | step5_grid_* + sweeps |
| T5 | Environment validation: impact ratios, dump costs, drift/fairness numbers | PENDING (data ready) | gate records |
| T6 | Descriptive stats: cost distribution summaries per policy/regime | PENDING (re-eval) | re-eval dump |
| T7 | L2 track: three-axis null summary | PENDING (data ready) | L2 results |
| T8 | Hyperparameters: base config + every variant change | BUILT 2026-07-14 (compiled OK) | metas + criteria §5 |
| T9 | L2 per-run detail (per-seed, per-axis; validation columns now, test columns after the sealed L2 test) | PLANNED (promoted from candidate 2026-07-14, user request) | L2 runs + l2_test_protocol.md |
| T10 | L4 tuning campaign per-run appendix: all 98 runs per-seed, both benchmarks (T2 is the 28-group summary; this is the full run-level record) | PLANNED (added 2026-07-14, user request) | step5_selection_v3 |

## STATUS SNAPSHOT (update on change)
2026-07-14: F1-F8 all BUILT (F7 v2 fixed: label + inset zoom). Figures folder refactored to
qrm/ + l2/. Tables T1/T2/T3/T8 BUILT + pdflatex-compile-verified. L2 data mapped (see below).
Grid still training (24/66, 0 fail). NEXT: build L2 figures (L1 lever + three-axis null) with the
mapped data; then grid lands -> F9 heatmap + T4; then re-eval -> F10/T6; then extension.
Final curation happens at report assembly — nothing gets cut before then.


## L2-TRACK DATA MAP (investigated 2026-07-14 — read before building L1-L5)
Raw results are RECOVERABLE and structured, but in an OLDER heterogeneous layout and with
UNEVEN cell coverage (not every algo x size was run in every axis folder). Sources:
- Per-run final agent-vs-TWAP: `curve.csv` last row, column `val_vs_twap_mean` (also in each
  axis folder's `*_DONE.json` -> runs[].final.val_vs_twap_mean). Folders:
  `scratch_hyperliquid/runs` (minute res), `runs_10s` (10s/30min), `runs_10s_10min` (10s/10min).
- Size-sweep lever (CLEAN, complete): `scratch_hyperliquid/size_sweep_results.json` (list of
  {label, rows:[{pct, size_btc, twap_is, instant_is, sched_gap, twap_resid_freq}]}).
- COVERAGE GAP to handle honestly in L1: PPO at the primary size (96.57 BTC) was NOT run at
  minute resolution (runs/ has ppo_size193.13 only). So the three-axis null figure must present
  the actual cells run, not a filled grid. This is why L1-L5 need careful design, not a rushed
  uniform grid — deferred to a focused build, data path now mapped.
