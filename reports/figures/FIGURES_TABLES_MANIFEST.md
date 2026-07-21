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
| F9 | Grid heatmap 4 sizes x 4 horizons, per regime | BUILT 2026-07-15 (16 cells; solid outline = 2 live §7.5 triggers (calm); dashed = centre volatile, already failed 2 sealed tests) | step5_grid_* + step5_sweep_* + selection | |
| F10a | Absolute execution costs per policy/regime (violins) | BUILT 2026-07-17 (re-eval integrity exact 20/20) | per_episode_v3/*.npz | Carlo descriptive stats |
| F10b | Paired per-episode difference distributions vs adaptive TWAP | BUILT 2026-07-17 | per_episode_v3/*.npz | |
| F11 | Intra-day / intra-week volume + liquidity profiles | PENDING (liquidity study) | L4 trades/book | Carlo request; feeds VWAP |
| F12 | Simulator realism: self-impact ratio, dump cost, book recovery; calibration fit | PENDING | gate records (step4, step3g) | methods chapter |
| F13 | Measured order-flow -> future-return curve (magnitude/horizon/decay, per regime) | PENDING (extension step 1) | L4 measurement | gold either way |
| F14 | Tuning forest, CALM (12 variants) | BUILT 2026-07-17 | step5_selection_v3 | completes F3 |
| F15 | Behaviour-audit scatter for ALL 98 tuning runs (F7 style) | CANDIDATE | step5_selection_v3 audit | |
| F16 | Action-distribution profiles: what the agent actually does per regime (pace histogram) | CANDIDATE | behaviour_audit action_shares | feeds RQ3 |
| F17 | Fairness gate: pace-multiple cost gradient before/after drift fix | CANDIDATE | step3g fairness records | methods |
| F18 | Sealed-confirmation per-seed detail (both tests, both benchmarks) | CANDIDATE | step5_confirm_* | may fold into T3 |
| F19 | Ladder collapse: per-seed dev-block vs reserve-block (both escalated calm groups; the p=0.0001 group sign-flips) | BUILT 2026-07-17 (slope graph, reviewed) | step5_esc_* + step5_xblock_* | |
| F20 | DQN collapse rate by setting (4 settings incl. primary; base vs d3 rhythm) | BUILT 2026-07-17 | step5_dqnprobe_* + step5_d3_* + step5_v3 audit | companion to F7 |
| F21 | DQN Q-value tilt: per-agent do-nothing argmax share vs audit outcome | BUILT 2026-07-17 | reports/diagnostics/dqn_q_flatness.json | |
| F22 | Evaluation-architecture schematic (seed blocks + screen/replicate/confirm ladder + outcomes) | BUILT 2026-07-17 | diagram (no data) | methods chapter |

## FIGURES — L2 / frozen-replay track (secondary chapter; full set per comprehensive-first)

| ID | Title / content | Status | Source data | Notes |
|---|---|---|---|---|
| L1 | Three-lever null summary (resolution x size x deadline, all 70 agents, VALIDATION-labelled) | BUILT 2026-07-17 (reports/figures/l2/) | the 70 agents' meta.json | test-set panels follow the sealed exam |
| L2 | Size-sweep lever (benchmark cost grows; agent advantage does not appear; VALIDATION-labelled) | BUILT 2026-07-17 (reports/figures/l2/) | size_sweep_results.json + runs_10s metas | |
| L3 | Anti-bias validation: planted synthetic edge IS detected | **ARTIFACT NOT FOUND on disk (2026-07-17 search)** — claim quarantined; regenerate the small experiment or drop | (missing) | flagged honestly in the results pack |
| L4 | L2 learning curves (PPO/DQN) | CANDIDATE | L2 run logs | |
| L5 | L2 DQN collapse evidence | CANDIDATE | L2 audit artifacts | mirrors F7 |

## TABLES (all emitted as booktabs .tex by make_tables.py — TO BUILD)

| ID | Table | Status | Source |
|---|---|---|---|
| T1 | Primary campaign: all 20 runs per-seed, both benchmarks, p, audit flag | BUILT 2026-07-14 (compiled OK; numbers verified vs source) | step5_v3 |
| T2 | Tuning/selection: all 28 groups (seeds, valid, pooled, across-seed p) | BUILT 2026-07-14 (compiled OK) | step5_selection_v3 |
| T3 | Both sealed confirmations: per-seed + pass/fail criteria checklist | BUILT 2026-07-14 (compiled OK; verified) | step5_confirm_v3a/v1b |
| T4 | Robustness grid: 4x4 pooled + p per regime | BUILT 2026-07-15 (compiled OK; 2 live triggers flagged; centre volatile marked closed/§6-FAIL) | step5_grid_* + sweeps + selection |
| T5 | Environment validation: impact ratios, dump costs, drift/fairness numbers | BUILT 2026-07-17 (compiled OK) | step4_gates_v3.json + step3g fairness verdicts |
| T6 | Descriptive stats: cost distribution summaries per policy/regime | BUILT 2026-07-17 (compiled OK) | per_episode_v3/*.npz |
| T7 | L2 track: 14-arm validation summary (test columns follow the sealed exam) | BUILT 2026-07-17 (compiled OK) | 70 agents' meta.json |
| T8 | Hyperparameters: base config + every variant change | BUILT 2026-07-14 (compiled OK) | metas + criteria §5 |
| T9 | L2 per-run detail, all 70 agents (validation; test columns after the sealed exam) | BUILT 2026-07-17 (compiled OK) | 70 agents' meta.json |
| T10 | L4 tuning campaign per-run appendix: all 98 runs per-seed, both benchmarks | BUILT 2026-07-17 (compiled OK) | step5_selection_v3 |
| T11 | DQN probe per-run (18 runs: audit shares + costs) | BUILT 2026-07-17 (compiled OK) | step5_dqnprobe_* |
| T12 | Replication ladder per-seed (dev vs reserve, both trigger groups) | BUILT 2026-07-17 (compiled OK) | step5_esc_* + step5_xblock_* |

## STATUS SNAPSHOT (update on change)
2026-07-17 EOD: F1-F10b + F14 + F19-F22 + L1/L2fig ALL BUILT (18 figures); tables T1-T12 ALL
BUILT (T9/T10 as page-flowing longtables after user review); L3 artifact still MISSING
(quarantined). RESULTS PACK shipped: reports/results_pack/ (full draft + meeting pack + Overleaf
copies + talking points v3). Per-episode re-eval done (integrity 20/20). NEXT figure/table work:
L2 exam lands -> T7/T9 test columns + L1 test panels; extension -> F13; liquidity -> F11.

2026-07-16: F1-F9 BUILT (F2/F4/F5 v5 label placement; F7 v3 two-panel; F9 = 16-cell grid heatmap
with the two §7.5 triggers outlined and the centre sealed-fail marked). Tables T1/T2/T3/T4/T8
BUILT + compile-verified. F19 (ladder collapse) + F20 (DQN collapse by setting) PLANNED; F21
(Q-value tilt) CANDIDATE. Grid/ladder/DQN-probe campaigns all CLOSED (see live doc CURRENT
RESULTS (D)+(E)); d3 COMPLETE+JUDGED 2026-07-17 (rhythm objection closed, CURRENT RESULTS (F));
L2 fill-ins COMPLETE (4 ppo193@1-min stub-replacement retrains in flight). NEXT: F19/F20 builds; L2 exam
lands -> T7/T9 test columns + L1-L5; per-episode re-eval -> F10a/F10b + T6; extension -> F13.
Final curation happens at report assembly — nothing gets cut before then.

CAPTION + NAMING POLICY (user directive, post-18/07 meeting — CRUCIAL, applies to ALL
report-facing text): markers grade ONLY what is on the page; assume the reader has ZERO
background context. (1) Every figure/table caption must be SELF-CONTAINED — it must explain
what is shown AND why it looks the way it does (worked example: F4/three-block caption now
states WHY only the two selection winners have sealed points — sealed blocks are single-use,
the pre-registered budget allowed exactly two tests). (2) "Track 1"/"Track 2" are BANNED in
report-facing text — tracks are named by description (frozen-replay track / reactive-simulator
track); executed across all 4 pack .tex files 2026-07-18, zero mentions remain. (3) No internal
shorthand or doc names ("frozen rules files") in report prose. Internal docs keep L2/L4
shorthand — this policy is for anything a reader/marker sees.

CURATION STEER FOR FINAL ASSEMBLY (Carlo, 18/07 meeting; logged post-meeting): in the final
dissertation the L2 frozen-replay track is described briefly (mainly to justify the switch to
the QRM reactive track); only the most important L2 results appear in the main body, the rest
to appendix. Main focus = L4/QRM reactive track. THIS APPLIES AT ASSEMBLY ONLY — the
comprehensive-first policy above stands until then (user: "WE ARE NOT TRIMMING ANYTHING YET";
the results pack keeps every L2 figure/table for Carlo's review).

2026-07-14 (history): F1-F8 built; folder refactor; T1/T2/T3/T8 built; grid was training.


## L2-TRACK DATA MAP (investigated 2026-07-14 — read before building L1-L5)
Raw results are RECOVERABLE and structured, but in an OLDER heterogeneous layout and with
UNEVEN cell coverage (not every algo x size was run in every axis folder). Sources:
- Per-run final agent-vs-TWAP: `curve.csv` last row, column `val_vs_twap_mean` (also in each
  axis folder's `*_DONE.json` -> runs[].final.val_vs_twap_mean). Folders:
  `scratch_hyperliquid/runs` (minute res), `runs_10s` (10s/30min), `runs_10s_10min` (10s/10min).
- Size-sweep lever (CLEAN, complete): `scratch_hyperliquid/size_sweep_results.json` (list of
  {label, rows:[{pct, size_btc, twap_is, instant_is, sched_gap, twap_resid_freq}]}).
- COVERAGE GAP (2026-07-14 snapshot) to handle honestly in L1: PPO at the primary size (96.57
  BTC) was NOT run at minute resolution (runs/ had ppo_size193.13 only).
  UPDATE 2026-07-16: the gaps are BEING FILLED — 19 fill-in agents (user decision 2026-07-14,
  exact list + live status in `reports/l2_test_protocol.md` §3) are training; once done, every
  presented panel is complete and L1-L5 build from full panels plus the sealed-test columns.

## RESULTS PACK (2026-07-17)
`reports/results_pack/` = the living proto results chapter: `results_pack.tex` (source,
tables via \input), `results_pack_overleaf.tex` (single-file, all tables inlined - THE one
to paste into Overleaf), `figures/` (16 PDFs, drag into Overleaf), `results_pack.pdf`
(compiled, 24pp), `talking_points.md` (meeting script with probing Q&A). NOT frozen:
revised as pending experiments land; final curation at report assembly.
