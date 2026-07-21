# RAW RESULTS MANIFEST (self-describing map for this data folder)

Last updated 2026-07-10. This folder holds RAW results for the QRM reactive-execution project.
The analysis/write-ups live in the code repo (`drl-optimal-order-execution/reports/`); this file
exists so the DATA folder is self-describing on its own. `runs_*` = trained agents (model.zip +
meta.json + curve.json each). `step5*` = scored numbers (judgement.json + behaviour_audit.json).

WARNING: individual run folders do NOT record which simulator they used; a v2 and a v3 agent look
identical inside (identical inner names, identical meta.json keys). The folder name is the only
marker. For that reason ALL superseded folders were RENAMED with a `SUPERSEDED_` prefix on
2026-07-10 (contents untouched), so pointing a script at a stale folder can no longer happen
silently. Use this manifest as the source of truth.

## NAMING CONVENTION — how a folder name maps to an exact agent build
Every run folder is named `{algo}_{regime}_s{seed}{tag}`:
- `algo` = `ppo` or `dqn`; `regime` = `calm` or `volatile`; `s{seed}` = training seed (s0-s4).
- `tag` EMPTY = the base/primary configuration (e.g. `runs_primary_v3/ppo_volatile_s0`).
- `tag` present = a §5 tuning variant (e.g. `runs_tuning_v3/ppo_volatile_s0_v3b`).
The same names appear as the `run` field in every `judgement.json` / `behaviour_audit.json` row.
Each folder's `meta.json` also SELF-DESCRIBES the build (algo, regime, seed, tag, overrides such
as `lr: 0.0001`), so a build can always be verified from the artifact itself.

Variant tag key (change from the base config; full pre-registered table in repo
`reports/qrm_step4_criteria.md` §5):
| tag | change | example folder |
|---|---|---|
| `_v1a` | net_arch [64,64] | `runs_tuning_v3/ppo_volatile_s0_v1a` |
| `_v1b` | net_arch [128,128] | `runs_tuning_v3/ppo_volatile_s0_v1b` |
| `_v2`  | reward scale x100 | `runs_tuning_v3/ppo_volatile_s0_v2` |
| `_v3a` | learning rate 1e-3 | `runs_tuning_v3/ppo_volatile_s0_v3a` |
| `_v3b` | learning rate 1e-4 (tuning winner, volatile) | `runs_tuning_v3/ppo_volatile_s0_v3b` |
| `_v4a` | ent_coef 0.0 | `runs_tuning_v3/ppo_volatile_s0_v4a` |
| `_v4b` | ent_coef 0.05 | `runs_tuning_v3/ppo_volatile_s0_v4b` |
| `_v5`  | n_steps 8192 | `runs_tuning_v3/ppo_volatile_s0_v5` |
| `_d1`  | DQN exploration_final_eps 0.05 + anneal 0.5 | `runs_tuning_v3/dqn_volatile_s0_d1` |
| `_d2`  | DQN reward x100 + net [64,64] | `runs_tuning_v3/dqn_volatile_s0_d2` |

## CURRENT — cite these
- **`step5_confirm_v3a/` = R8b OUT-OF-SAMPLE CONFIRMATION VERDICT (one-shot, sealed 9e6 block,
  2026-07-11): PASS = FALSE — the volatile edge DID NOT REPLICATE** (pooled -0.0023 bps, across-seed
  p=0.38, 3/5 cheaper, all 5 seeds valid). PROJECT HEADLINE = BOUNDARY NULL. The dev-block -0.0628
  must never be cited as a confirmed edge. Agents: `runs_confirm_v3a/` (fresh seeds 5-9).
- `step5_confirm_v1b/` = REMEDIAL CONFIRMATION (bigger-network v1b, the literal-rule pick; one-shot,
  sealed 13e6 block, 2026-07-13): PASS=FALSE, pooled -0.0022, p=0.39, 2/5 cheaper. Agents:
  `runs_confirm_v1b/` (fresh seeds 10-14). BOTH sealed tests now FAIL -> null is robust to the
  selection-metric choice; confirmation family CLOSED at two tests.
- `step5_selection_v3/` = TUNING + SELECTION NUMBERS OF RECORD (sealed 2026-07-10; all 98 runs =
  60-run screen + 38-run selection batch). Selected config = ppo_volatile_v3a (lr 1e-3), dev-block
  pooled -0.0628 vs adaptive, across-seed p=0.0043, 5/5 seeds valid — UNCONFIRMED (see above); it
  OVERTOOK the 3-seed screen leader v3b once escalated to 5 seeds. V6/V6b (10M-step arms) rejected:
  longer training did not help.
- `step5_v3/` = PRIMARY-campaign numbers of record (clean, drift-fixed; sealed judgement).
- `runs_primary_v3/` = the 20 trained agents behind step5_v3 (DQN+PPO x calm+volatile x 5 seeds).
- `runs_tuning_v3/` = all 98 tuning/selection agents (10 variants x 3 seeds x 2 regimes + 14
  escalation seeds + 12 x 10M-step V6/V6b + 12 combo runs).
- `step5_tuning_v3/` = STAGE RECORD: the 60-run 3-seed screen scored 2026-07-09. Its numbers are
  reproduced identically inside step5_selection_v3 (determinism-checked); its 3-seed "winner v3b"
  is a stage result SUPERSEDED by the 5-seed selection above. `judgement_preAcrossSeed_backup.json`
  keeps the pre-backfill copy (across_seed block added after scoring; frozen verdicts byte-identical).
- `step3g/` = the calibrated simulator of record: `qrm_bundle_{calm,volatile}_b.npz` +
  `move_process_{calm,volatile}_centered.npz` (drift-neutralized; `..._DRIFTY_backup.npz` keeps the
  pre-fix version). `book_05s_v2/` = the reconstructed order book of record.

## SUPERSEDED — keep for the record, NEVER cite as results
All renamed 2026-07-10 with a `SUPERSEDED_` prefix (contents untouched; older docs may cite the
un-prefixed names — same folders):
- `SUPERSEDED_step5_v2/` + `SUPERSEDED_runs_primary_v2/` = drift-CONFOUNDED primary campaign
  (before the drift fix).
- `SUPERSEDED_step5/` + `SUPERSEDED_runs_reactive/` = original campaign, later found to have 10
  bugs (engineering shakedown).
- `SUPERSEDED_step5_wave1/` + `SUPERSEDED_runs_wave1/` + `SUPERSEDED_runs_wave2/` = old
  (pre-drift-fix) tuning runs. NOTE: their INNER agent names are identical to `runs_tuning_v3`'s
  (e.g. both contain a `dqn_calm_s0_d1`) — the parent folder is the only distinguisher.
- `SUPERSEDED_runs_reactive_smoke/` + `SUPERSEDED_runs_smoke_r4/` = throwaway smoke tests.

Not renamed (not results runs): `step3f/` (closed calibration step), `step4/` (realism-gate
artifacts), `book_05s/` (older book build; `book_05s_v2/` is the one of record), extraction dirs.

Provenance chain of the primary result: original buggy (`SUPERSEDED_step5`) -> corrected but
drift-confounded (`SUPERSEDED_step5_v2`) -> clean drift-fixed CURRENT (`step5_v3`).

## SWEEP RESULTS (added 2026-07-13, criteria §7 — CURRENT, cite these)
- `runs_sweep_b5/ b12/ b50/ h600/` = 24 sweep agents (selected config PPO lr 1e-3; tags
  `_v3aB5` = 5 BTC, `_v3aB12` = 12.5 BTC, `_v3aB50` = 50 BTC, `_v3aH600` = 25 BTC @ 600 s;
  meta.json records order_btc + env_steps).
- `step5_sweep_b5/ b12/ b50/ h600/` = scored sweep numbers (judgement.json records the cell's
  order_btc/env_steps). VERDICT: null holds in every cell; no §7.5 trigger; size-response
  non-monotone (edge absent at 5/12.5/50 BTC -> 25-BTC dev edge was selection luck).

## GRID RESULTS (added 2026-07-15, criteria §7.7 Part A+B — CURRENT, cite these)
- `runs_grid_{b5,b12,b25,b50}h150 / {b5,b12,b50}h600 / {b5,b12,b25,b50}h1200` = the 66 grid
  agents (11 new size x horizon cells x 2 regimes x 3 seeds; selected config PPO lr 1e-3; tags
  `_gS{5,12,25,50}H{150,600,1200}`; meta.json records order_btc + env_steps). Integrity-checked
  2026-07-15 (all 66 metas/configs/curves verified) before judging.
- `step5_grid_<cell>/` (11 dirs) = scored grid numbers (screen mode, dev block 5e6, n=2000,
  matching flags recorded in each judgement.json). VERDICT 2026-07-15: 64/66 valid (2 calm
  2.5-min runs failed the deadline-residual audit); null in 20/22 groups; strict per-seed
  ESCALATE false everywhere; §7.5 TRIGGER fires in TWO CALM groups — 50 BTC/10-min (pooled
  -0.0539, p=0.014, 3/3 cheaper) and 25 BTC/20-min (pooled -0.0629, p=0.0003, 3/3 cheaper).
  §7.5 ladder EXECUTED 2026-07-15 (criteria §7.5a/§7.5b) — outcome below.
- `step5_esc_b50h600/ step5_esc_b25h1200/` = LADDER STEP (i): both calm groups escalated to
  5 seeds (seeds 3,4 added to `runs_grid_*`), re-judged on dev 5e6. BOTH SURVIVED
  (-0.0432 p=0.0037; -0.0609 p=0.0001). Originals in step5_grid_* byte-identical (verified).
- `step5_xblock_b50h600/ step5_xblock_b25h1200/` = LADDER STEP (ii): same 10 agents on the
  RESERVE block 6,000,000 (FIRST USE, now spent as a cross-block instrument). **BOTH FAIL:
  -0.0095 (p=0.17) and +0.0177 (p=0.96, sign-flipped). NO sealed test spent. Grid verdict
  FINAL: null across the whole 4x4 design space; the two triggers were evaluation-block
  luck — third demonstrated block-luck illusion.** Full tables: live doc CURRENT RESULTS (D).

## D3 UPDATE-RHYTHM VARIANT (criteria §7.7 Part E — COMPLETE + JUDGED 2026-07-17, cite these)
- `runs_d3_{b25h300,b25h150,b5h150}/` = 18 agents: DQN base + library-default update rhythm
  (--dqn-train-freq 4 --dqn-batch-size 32; tags `_d3S{25,5}H{300,150}`), 3 cells x 2 regimes
  x 3 seeds. Integrity ALL PASS.
- `step5_d3_{b25h300,b25h150,b5h150}/` = judgements (dev block, matching flags). VERDICT:
  collapse NOT cured at the primary setting (25/5min: 2/6 valid vs base ~3/10 — unchanged);
  severity reduced only at the extreme deadline (25/2.5min: 4/6 vs 0/6); control healthy
  (5/2.5min: 4/6 vs 4/6). No cost trigger anywhere. Rhythm objection CLOSED (argument bank
  §N5; full table: live doc CURRENT RESULTS (F); outcome: criteria Part E).

## DQN PROBE (added 2026-07-16, criteria §7.7 Part D — CURRENT, cite these)
- `runs_dqnprobe_{b5h150,b25h150,b25h1200}/` = 18 base-config DQN agents (3 cells x 2 regimes
  x 3 seeds; tags `_dqS{5,25}H{150,1200}`).
- `step5_dqnprobe_{b5h150,b25h150,b25h1200}/` = judgements (dev block, matching flags).
  VERDICT: collapse SYSTEMATIC + SIZE-DRIVEN — valid 4/6 (5BTC/2.5min), 0/6 (25BTC/2.5min),
  2/6 (25BTC/20min) vs 3/10 at the primary cell; calm-concentrated (1/9 vs 5/9 volatile);
  no cost trigger anywhere. Part D CLOSED. Full table: live doc CURRENT RESULTS (E).
