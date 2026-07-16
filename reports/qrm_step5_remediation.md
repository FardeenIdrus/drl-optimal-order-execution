# QRM EXECUTION — LIVE WORKING DOC (results record + roadmap)

**Role (restructured 2026-07-11):** THE live document for the reactive-QRM execution study.
Layout: newest first — the run-directory manifest, then the two CURRENT RESULTS sections
(numbers of record), then the REMAINING ROADMAP, invariants, and write-up prose. Everything
below the HISTORICAL RECORD banner is completed/superseded work kept verbatim as the audit
trail — never cite numbers from down there. Documentation map: `reports/00_INDEX.md`.
Paths: repo = /Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution
(run everything with PYTHONPATH=src, venv .venv). Scratch =
/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4 (call it $S).
### RUN-DIRECTORY MANIFEST (current vs superseded) — the single source of truth for which results to cite
- **HEADLINE (2026-07-11, reinforced 2026-07-13): the volatile edge DID NOT REPLICATE. BOTH sealed
  tests FAIL.** (1) faster-learning v3a on the 9e6 block: pooled -0.0023, p=0.38, 3/5 cheaper
  (`$S/step5_confirm_v3a/`). (2) bigger-network v1b (the literal-rule pick) on the 13e6 block:
  pooled -0.0022, p=0.39, 2/5 cheaper (`$S/step5_confirm_v1b/`; criteria §6.10/§6.11). Both
  candidate champions return ~zero on fresh sealed data -> the §6.9 selection deviation is
  immaterial to the conclusion. PROJECT HEADLINE = BOUNDARY NULL. Do NOT cite any development-set
  number (-0.063 v3a, -0.056 v1b) as a confirmed edge. Confirmation family CLOSED at two tests.
- **CURRENT (cite these):** `$S/step5_confirm_v3a/{judgement,behaviour_audit}.json` = R8b
  CONFIRMATION numbers of record (one-shot, sealed 9e6 block; agents in `$S/runs_confirm_v3a/`).
  `$S/step5_selection_v3/{judgement,behaviour_audit}.json` = TUNING + SELECTION numbers of record
  (98 runs; agents in `$S/runs_tuning_v3/`); selected config was ppo_volatile_v3a (lr 1e-3),
  dev-block pooled -0.0628, across-seed p=0.0043, 5/5 valid — UNCONFIRMED, see headline.
  `$S/step5_v3/{judgement,behaviour_audit}.json` = PRIMARY-campaign numbers of record (20 base
  agents in `$S/runs_primary_v3/`). `$S/step5_tuning_v3/` = the sealed 60-run 3-seed screen STAGE
  record — its numbers are reproduced identically inside step5_selection_v3, and its 3-seed
  "winner v3b" line is a stage result SUPERSEDED by the 5-seed selection (v3a).
- **NAMING CONVENTION (variant -> folder):** run folders are `{algo}_{regime}_s{seed}{tag}`; tag
  empty = base config (primary campaign), tag = tuning variant (criteria §5), e.g. v3b volatile
  seed 0 = `$S/runs_tuning_v3/ppo_volatile_s0_v3b/`. The same names are the `run` field in every
  judgement/audit row; each folder's `meta.json` self-describes the build (tag + overrides, e.g.
  `lr: 0.0001`). Full tag key: `$S/RESULTS_MANIFEST.md`.
- **SUPERSEDED — keep for the record, NEVER cite as results.** All RENAMED 2026-07-10 with a
  `SUPERSEDED_` prefix (contents untouched) because inner run names collide with current ones:
  `SUPERSEDED_step5_v2` + `SUPERSEDED_runs_primary_v2` (drift-confounded); `SUPERSEDED_step5` +
  `SUPERSEDED_runs_reactive` (original buggy campaign); `SUPERSEDED_step5_wave1` +
  `SUPERSEDED_runs_wave1` + `SUPERSEDED_runs_wave2` (old pre-drift-fix tuning);
  `SUPERSEDED_runs_reactive_smoke` + `SUPERSEDED_runs_smoke_r4` (smoke tests). Older mentions of the
  un-prefixed names in this file refer to these same folders.

## CURRENT RESULTS (0) — R8b OUT-OF-SAMPLE CONFIRMATION: **DID NOT REPLICATE** (one-shot, sealed 9e6 block, 2026-07-11)

**Source of record (EXACT ABSOLUTE PATHS):**
- Verdict + costs + p-values, all 5 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_confirm_v3a/judgement.json`
- Behaviour audit:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_confirm_v3a/behaviour_audit.json`
- The 5 fresh-seed agents:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_confirm_v3a/`

**Design (pre-registered §6 + §6.4 + §6.7, fixed 2026-07-08/09, executed exactly):** selected config
ppo_volatile_v3a (lr 1e-3) retrained from scratch on FRESH seeds 5-9 (never used anywhere);
evaluated ONCE on the sealed 2,000-episode CRN block seed0=9,000,000 (verified untouched by any
prior run); audit before costs; binary pass rule fixed in advance; NO re-runs permitted.

**FULL per-seed table (sealed block, both benchmarks):**

| run (fresh seed) | vs_fixed | p_fixed | vs_adaptive | p_adapt | valid |
|---|---|---|---|---|---|
| ppo_volatile_s5_v3a | +0.0221 | 0.751 | +0.0222 | 0.749 | yes |
| ppo_volatile_s6_v3a | -0.0104 | 0.817 | -0.0103 | 0.813 | yes |
| ppo_volatile_s7_v3a | +0.0028 | 0.801 | +0.0028 | 0.804 | yes |
| ppo_volatile_s8_v3a | -0.0080 | 0.926 | -0.0079 | 0.936 | yes |
| ppo_volatile_s9_v3a | -0.0184 | 0.735 | -0.0184 | 0.758 | yes |

**§6.4 verdict (binary, pre-committed): PASS = false.**
- pooled vs adaptive = **-0.0023 bps** (needed < 0: met, but trivially), vs fixed = -0.0024;
- across-seed one-sided t p = **0.3785** (needed < 0.05: FAILED); Wilcoxon cross-check p = 0.41;
- cheaper in **3/5** valid seeds (needed >= 4: FAILED);
- 95% CI vs adaptive: [-0.0217, +0.0171] — tightly bracketing ZERO;
- all 5 seeds VALID (no behaviour collapse — the agents are healthy, they simply match TWAP).

**Pre-committed interpretation (criteria §6.5 + §6.7 M3 — written before the run, applied verbatim):
the project headline is a BOUNDARY NULL.** The development-block edge (pooled -0.063, across-seed
p=0.004, 5/5 cheaper, robust across 7 hyperparameter variants) did NOT replicate on fresh training
seeds + fresh sealed data (pooled -0.002 ≈ zero). Per the pre-registration, no re-runs, no
reinterpretation, no second confirmation attempt. The dev-block signal is reported honestly as
what it now is: a consistent in-development-data pattern that failed out-of-sample replication —
consistent with selection effects (winner's curse over 14 configs + block-specific fit) and/or
high seed-to-seed variance (already foreshadowed by the v3b->v3a re-ranking when seeds were added).

**What this does NOT change:** DQN's collapse findings, the calm-null, the drift-fix methodology,
and the regime CONTRAST question (why did the dev-block signal appear only in volatile?) all stand.
RQ3 attribution proceeds on the dev-block agents as EXPLORATORY analysis of an unconfirmed signal,
clearly labelled as such. The null-branch deliverable (§6.7 M3): a rigorous, pre-registered,
fully-documented null with the complete discovery-stage evidence — scientifically defensible as-is.

**POST-VERDICT PROCESS AUDIT (2026-07-11, user-requested; full record criteria §6.9):**
(1) Confirmation execution verified correct — right agents (folder list == judged list, all metas
exact v3a), right sealed block, rule applied as written; an independent from-scratch reproduction
of a recorded number (seed-5 vs adaptive, recomputed with library primitives, not the judge
script) returned +0.0222 vs recorded +0.0222 — EXACT MATCH, scorer verified. (2) A SELECTION-RULE DEVIATION is
disclosed: §6.1 says best pooled "across BOTH regimes"; the actual pick ranked volatile-only (plus
an undocumented escalated-only filter). Under the literal rule the target would have been v1b
(strict health) or v3b (lenient health), not v3a — v3a wins only volatile-only, by 0.0008 over w3a.
(3) Three-block diagnostic: EVERY config flips sign across blocks (1e6 monitor: +0.06..+0.15 worse;
5e6 dev: -0.03..-0.06 better; 9e6 sealed: ~0) -> the dev edge was common-mode block luck shared by
all variants; across-variant robustness was correlated evidence, not replication. Consequence of
the deviation: likely nil (the literal-rule picks carry the same inflation and read WORSE than TWAP
on the independent 1e6 block). Recommendation on record: disclose, do NOT run remedial sealed tests
of other variants (sequential-testing multiplicity + §6 terminality); decision rests with the user.

## CURRENT RESULTS (0b) — REMEDIAL CONFIRMATION (bigger-network agent, one-shot, sealed 13e6, 2026-07-13): ALSO FAILS

**Source of record (EXACT ABSOLUTE PATHS):**
- `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_confirm_v1b/judgement.json`
- `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_confirm_v1b/behaviour_audit.json`
- agents: `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_confirm_v1b/` (fresh seeds 10-14)

Why (criteria §6.9->§6.10): the WRITTEN selection rule (best pooled across BOTH regimes) picks
ppo_v1b (net 128x128), not the volatile-only-ranked v3a that was tested first. This is the one
remedial test of the literal-rule target, closing the "the pre-registered pick was never tested"
gap. Pre-registered §6.10 BEFORE the run; sealed block 13e6 (verified virgin); fresh seeds 10-14.

**FULL per-seed table (sealed 13e6 block, both benchmarks):**

| run (fresh seed) | vs_fixed | p_fixed | vs_adaptive | p_adapt | valid |
|---|---|---|---|---|---|
| ppo_volatile_s10_v1b | -0.0010 | 0.665 | -0.0019 | 0.650 | yes |
| ppo_volatile_s11_v1b | +0.0042 | 0.970 | +0.0033 | 0.953 | yes |
| ppo_volatile_s12_v1b | -0.0297 | 0.305 | -0.0306 | 0.294 | yes |
| ppo_volatile_s13_v1b | +0.0153 | 0.472 | +0.0144 | 0.491 | yes |
| ppo_volatile_s14_v1b | +0.0047 | 0.950 | +0.0038 | 0.961 | yes |

**§6.4 verdict: PASS = false.** pooled vs adaptive -0.0022 (dev was -0.056); across-seed p=0.39;
cheaper in 2/5; CI [-0.0232, +0.0188] bracketing zero; all 5 valid.

**What this settles:** BOTH candidate champions (volatile-only pick v3a on 9e6, across-both pick
v1b on 13e6) return ~zero out-of-sample. The §6.9 selection-metric deviation is therefore
IMMATERIAL to the conclusion - the null does not depend on which defensible selection rule you
use. Confirmation family CLOSED at two tests (§6.5/§6.7 terminality). Fifth independent
corroboration that the development edge was selection luck, not skill.

## CURRENT RESULTS (A) — TUNING + SELECTION, numbers of record (sealed 2026-07-10, `step5_selection_v3`)

**Source of record (EXACT ABSOLUTE PATHS):**
- Costs + p-values, all 98 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_selection_v3/judgement.json`
- Behaviour audit (valid/invalid), all 98 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_selection_v3/behaviour_audit.json`
- Trained agents + logs + learning curves (98 folders, named `{algo}_{regime}_s{seed}{tag}`):
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_tuning_v3/`
- Stage record (the earlier 60-run 3-seed screen, numbers reproduced in the above):
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_tuning_v3/judgement.json`

The JSON is authoritative if this transcription ever disagrees. Campaign design: 60-run screen
(10 variants x 3 seeds x 2 regimes) + 38-run selection batch per criteria §5c (escalation of all
7 §5-rule-3 cells to 5 seeds; V6/V6b 10M-step arms; Wave-3 combos w3a/w3b), all on the
drift-fixed env, all judged on the 5e6 dev block (2,000 CRN episodes/agent).

**STEP-1 SELECTION BATCH — COMPLETE (2026-07-10). 38 new runs, 0 failures, judged into
`$S/step5_selection_v3/judgement.json` (98 runs total: 60 old + 38 new).**

**Determinism check (PASSED):** all 60 previously-sealed screen numbers reproduced byte-for-byte
in the new 98-run scoring (0 mismatches on cost, p-value, or audit fields, checked programmatically).
The scorer is confirmed deterministic; re-scoring never perturbs old results.

**HEADLINE — THE WINNER CHANGED once escalated from 3 to 5 seeds, and this is the honest,
unmassaged mechanical result:**

| rank | config | seeds | pooled vs adaptive | across-seed p |
|---|---|---|---|---|
| **1 (WINNER)** | **ppo_volatile_v3a — lr 1e-3** | **5/5** | **-0.0628** | **0.0043** |
| 2 | ppo_volatile_w3a — net128+reward100 combo | 3/3 (not escalated; below the trigger) | -0.0620 | 0.0158 |
| 3 | ppo_volatile_v3b — lr 1e-4 (the 3-seed screen's leader) | 5/5 | -0.0602 | 0.0100 |
| 4 | ppo_volatile_v1b — net 128,128 | 5/5 | -0.0562 | 0.0051 |
| 5 | ppo_volatile_v5 — n_steps 8192 | 5/5 | -0.0485 | 0.0055 |

**v3b (learning rate 1e-4) led the 3-seed screen (-0.084) but WEAKENED on the 2 extra seeds**
(seed 3: -0.0095 n.s., seed 4: -0.0401) **while v3a (learning rate 1e-3) STRENGTHENED**
(seed 3: -0.0805, seed 4: -0.0433) and overtakes it on the full 5-seed pooled cost. This is
reported exactly as it came out, with no adjustment: the 3-seed screen's leader is NOT the final
selection. Both remain comfortably significant (across-seed p<0.01) and material (pooled
> -0.05 floor); this is a re-ranking among two real, tuning-robust variants, not a reversal of
the finding that a volatile edge exists.

**V6 / V6b (10M-step, double-length training) DID NOT HELP — the "needed longer" hypothesis is
REJECTED for both:**
- v6 (v2 @ 10M) volatile: -0.025, -0.078, -0.030 (2/3 valid, one seed collapsed) — WORSE than v2 @
  2M (-0.046, 5/5 valid).
- v6b (v3b @ 10M) volatile: -0.063, -0.008, -0.048 (3/3 valid) — no better than v3b @ 2M (-0.060).
- Both calm arms are ~null/positive. Longer training does not extend or strengthen the edge;
  2M steps (the primary budget) was already sufficient. This closes the "needed longer" arm of
  the pre-registered continuation rules with a clean negative result.

**Combos (w3a = net128+reward100, w3b = lr1e-4+reward100):** w3a is a strong #2 (-0.062,
p=0.016) but was NOT part of the 5-seed escalation set (it entered via the Wave-3 rule, not the
§5-rule-3 trigger, so stayed at 3 seeds per the batch design) — not directly comparable to the
5-seed configs and not eligible for selection under the "fully-healthy escalated" bar used here.
w3b is weaker (-0.036, 2/3 valid, survivorship-flagged).

**SELECTION (§5 rule 2, mechanical, no discretion): ppo_volatile_v3a (learning rate 1e-3).**
Best pooled cost vs adaptive-TWAP among configs with all seeds valid and audit-passing.
**This is the config that proceeds to R8b out-of-sample confirmation.**

Calm: best is ppo_calm_v4a (-0.021, p=0.0095, 5/5) — still small and not being carried to
confirmation (§6.7 restricts confirmation to volatile as the primary regime; calm has no
material signal across the whole campaign).

Full 98-run table: `$S/step5_selection_v3/judgement.json`. DQN: all collapse-prone arms (d1, d2)
remain invalid/null in both regimes; no change to the DQN conclusion.

### VARIANT CATALOG — every tuned group: what changed, exact folders, group result

Folder paths are under `$S/runs_tuning_v3/` (the `s{...}` braces list that group's seeds; each
folder self-describes its build in `meta.json`). `pooled`/`across-seed p` computed on VALID seeds
only. Sorted best-first within each regime.

| variant | tuned for | regime | agent folders (under $S/runs_tuning_v3/) | seeds | valid | pooled vs adaptive | across-seed p |
|---|---|---|---|---|---|---|---|
| _v3a | learning rate 1e-3 (faster) | volatile | `ppo_volatile_s{0,1,2,3,4}_v3a/` | 0,1,2,3,4 | 5/5 | -0.0628 | 0.0043 |
| _w3a | combo: net128 + reward x100 | volatile | `ppo_volatile_s{0,1,2}_w3a/` | 0,1,2 | 3/3 | -0.0620 | 0.0158 |
| _v3b | learning rate 1e-4 (slower) | volatile | `ppo_volatile_s{0,1,2,3,4}_v3b/` | 0,1,2,3,4 | 5/5 | -0.0602 | 0.0100 |
| _v1b | bigger network 128x128 | volatile | `ppo_volatile_s{0,1,2,3,4}_v1b/` | 0,1,2,3,4 | 5/5 | -0.0562 | 0.0051 |
| _v6 | V6: v2 config @ 10M steps | volatile | `ppo_volatile_s{0,1,2}_v6/` | 0,1,2 | 2/3 | -0.0515 | 0.1504 (survivorship: dropped seeds) |
| _v5 | n_steps 8192 (longer rollout) | volatile | `ppo_volatile_s{0,1,2,3,4}_v5/` | 0,1,2,3,4 | 5/5 | -0.0485 | 0.0055 |
| _v4a | ent_coef 0.0 (no exploration bonus) | volatile | `ppo_volatile_s{0,1,2}_v4a/` | 0,1,2 | 3/3 | -0.0472 | 0.0620 |
| _v2 | reward scale x100 | volatile | `ppo_volatile_s{0,1,2,3,4}_v2/` | 0,1,2,3,4 | 5/5 | -0.0462 | 0.0022 |
| _d1 | DQN eps_final 0.05 + anneal 0.5 | volatile | `dqn_volatile_s{0,1,2}_d1/` | 0,1,2 | 1/3 | -0.0403 | n/a (survivorship: dropped seeds) |
| _v6b | V6b: v3b config @ 10M steps | volatile | `ppo_volatile_s{0,1,2}_v6b/` | 0,1,2 | 3/3 | -0.0395 | 0.0685 |
| _w3b | combo: lr 1e-4 + reward x100 | volatile | `ppo_volatile_s{0,1,2}_w3b/` | 0,1,2 | 2/3 | -0.0363 | 0.0853 (survivorship: dropped seeds) |
| _v4b | ent_coef 0.05 (more exploration) | volatile | `ppo_volatile_s{0,1,2,3,4}_v4b/` | 0,1,2,3,4 | 5/5 | -0.0267 | 0.0476 |
| _d2 | DQN reward x100 + net 64x64 | volatile | `dqn_volatile_s{0,1,2}_d2/` | 0,1,2 | 3/3 | -0.0259 | 0.1985 |
| _v1a | bigger network 64x64 | volatile | `ppo_volatile_s{0,1,2}_v1a/` | 0,1,2 | 2/3 | -0.0196 | 0.1505 (survivorship: dropped seeds) |
| _v3b | learning rate 1e-4 (slower) | calm | `ppo_calm_s{0,1,2}_v3b/` | 0,1,2 | 2/3 | -0.0239 | 0.1108 (survivorship: dropped seeds) |
| _v4a | ent_coef 0.0 (no exploration bonus) | calm | `ppo_calm_s{0,1,2,3,4}_v4a/` | 0,1,2,3,4 | 5/5 | -0.0213 | 0.0095 |
| _v1b | bigger network 128x128 | calm | `ppo_calm_s{0,1,2}_v1b/` | 0,1,2 | 3/3 | -0.0211 | 0.1403 |
| _v5 | n_steps 8192 (longer rollout) | calm | `ppo_calm_s{0,1,2}_v5/` | 0,1,2 | 3/3 | -0.0159 | 0.1408 |
| _v1a | bigger network 64x64 | calm | `ppo_calm_s{0,1,2}_v1a/` | 0,1,2 | 3/3 | -0.0159 | 0.0395 |
| _w3b | combo: lr 1e-4 + reward x100 | calm | `ppo_calm_s{0,1,2}_w3b/` | 0,1,2 | 3/3 | -0.0116 | 0.0829 |
| _v6b | V6b: v3b config @ 10M steps | calm | `ppo_calm_s{0,1,2}_v6b/` | 0,1,2 | 3/3 | -0.0108 | 0.2141 |
| _d1 | DQN eps_final 0.05 + anneal 0.5 | calm | `dqn_calm_s{0,1,2}_d1/` | 0,1,2 | 1/3 | -0.0068 | n/a (survivorship: dropped seeds) |
| _w3a | combo: net128 + reward x100 | calm | `ppo_calm_s{0,1,2}_w3a/` | 0,1,2 | 3/3 | -0.0063 | 0.2002 |
| _v4b | ent_coef 0.05 (more exploration) | calm | `ppo_calm_s{0,1,2}_v4b/` | 0,1,2 | 3/3 | -0.0058 | 0.2003 |
| _v2 | reward scale x100 | calm | `ppo_calm_s{0,1,2}_v2/` | 0,1,2 | 3/3 | -0.0024 | 0.3218 |
| _v3a | learning rate 1e-3 (faster) | calm | `ppo_calm_s{0,1,2}_v3a/` | 0,1,2 | 3/3 | +0.0013 | 0.5279 |
| _v6 | V6: v2 config @ 10M steps | calm | `ppo_calm_s{0,1,2}_v6/` | 0,1,2 | 3/3 | +0.0046 | 0.7049 |
| _d2 | DQN reward x100 + net 64x64 | calm | `dqn_calm_s{0,1,2}_d2/` | 0,1,2 | 0/3 | n/a (no valid) | n/a (survivorship: dropped seeds) |

### FULL PER-SEED TABLE — all 98 runs, both benchmarks

**Sign convention:** cost = agent MINUS benchmark, in bps, over 2,000 CRN-paired episodes (eval
seed0 = 5,000,000). NEGATIVE = agent cheaper. p = per-seed Wilcoxon signed-rank. `valid` = passed
the behaviour audit. Sorted: volatile first, then variant, then seed.

| run | vs_fixed | p_fixed | vs_adaptive | p_adapt | valid |
|---|---|---|---|---|---|
| ppo_volatile_s0_v1a | -0.0089 | 0.803 | -0.0096 | 0.786 | yes |
| ppo_volatile_s1_v1a | -0.0290 | 0.374 | -0.0296 | 0.375 | yes |
| ppo_volatile_s2_v1a | +0.0232 | 0.546 | +0.0226 | 0.551 | NO |
| ppo_volatile_s0_v1b | -0.0150 | 0.573 | -0.0156 | 0.560 | yes |
| ppo_volatile_s1_v1b | -0.0436 | 0.196 | -0.0443 | 0.190 | yes |
| ppo_volatile_s2_v1b | -0.0640 | 0.071 | -0.0646 | 0.070 | yes |
| ppo_volatile_s3_v1b | -0.0870 | 0.020 | -0.0876 | 0.020 | yes |
| ppo_volatile_s4_v1b | -0.0684 | 0.051 | -0.0691 | 0.049 | yes |
| ppo_volatile_s0_v2 | -0.0501 | 0.168 | -0.0507 | 0.156 | yes |
| ppo_volatile_s1_v2 | -0.0615 | 0.119 | -0.0621 | 0.120 | yes |
| ppo_volatile_s2_v2 | -0.0519 | 0.186 | -0.0525 | 0.179 | yes |
| ppo_volatile_s3_v2 | -0.0495 | 0.156 | -0.0502 | 0.155 | yes |
| ppo_volatile_s4_v2 | -0.0150 | 0.965 | -0.0156 | 0.965 | yes |
| ppo_volatile_s0_v3a | -0.0494 | 0.243 | -0.0500 | 0.250 | yes |
| ppo_volatile_s1_v3a | -0.1044 | 0.012 | -0.1050 | 0.011 | yes |
| ppo_volatile_s2_v3a | -0.0345 | 0.369 | -0.0352 | 0.363 | yes |
| ppo_volatile_s3_v3a | -0.0798 | 0.023 | -0.0805 | 0.022 | yes |
| ppo_volatile_s4_v3a | -0.0427 | 0.236 | -0.0433 | 0.230 | yes |
| ppo_volatile_s0_v3b | -0.0717 | 0.047 | -0.0724 | 0.046 | yes |
| ppo_volatile_s1_v3b | -0.1019 | 0.006 | -0.1026 | 0.006 | yes |
| ppo_volatile_s2_v3b | -0.0759 | 0.038 | -0.0765 | 0.037 | yes |
| ppo_volatile_s3_v3b | -0.0089 | 0.866 | -0.0095 | 0.847 | yes |
| ppo_volatile_s4_v3b | -0.0394 | 0.344 | -0.0401 | 0.340 | yes |
| ppo_volatile_s0_v4a | -0.0501 | 0.188 | -0.0508 | 0.182 | yes |
| ppo_volatile_s1_v4a | -0.0764 | 0.018 | -0.0771 | 0.020 | yes |
| ppo_volatile_s2_v4a | -0.0131 | 0.944 | -0.0137 | 0.956 | yes |
| ppo_volatile_s0_v4b | -0.0402 | 0.266 | -0.0408 | 0.247 | yes |
| ppo_volatile_s1_v4b | -0.0380 | 0.334 | -0.0387 | 0.319 | yes |
| ppo_volatile_s2_v4b | -0.0231 | 0.422 | -0.0238 | 0.413 | yes |
| ppo_volatile_s3_v4b | -0.0492 | 0.154 | -0.0498 | 0.147 | yes |
| ppo_volatile_s4_v4b | +0.0201 | 0.682 | +0.0195 | 0.702 | yes |
| ppo_volatile_s0_v5 | -0.0506 | 0.156 | -0.0512 | 0.159 | yes |
| ppo_volatile_s1_v5 | -0.0211 | 0.698 | -0.0217 | 0.679 | yes |
| ppo_volatile_s2_v5 | -0.0261 | 0.454 | -0.0267 | 0.434 | yes |
| ppo_volatile_s3_v5 | -0.0779 | 0.056 | -0.0785 | 0.052 | yes |
| ppo_volatile_s4_v5 | -0.0637 | 0.124 | -0.0643 | 0.125 | yes |
| ppo_volatile_s0_v6 | -0.0245 | 0.829 | -0.0252 | 0.819 | yes |
| ppo_volatile_s1_v6 | -0.0772 | 0.083 | -0.0778 | 0.076 | yes |
| ppo_volatile_s2_v6 | -0.0295 | 0.608 | -0.0302 | 0.601 | NO |
| ppo_volatile_s0_v6b | -0.0620 | 0.093 | -0.0627 | 0.089 | yes |
| ppo_volatile_s1_v6b | -0.0073 | 0.976 | -0.0079 | 0.997 | yes |
| ppo_volatile_s2_v6b | -0.0474 | 0.151 | -0.0480 | 0.146 | yes |
| ppo_volatile_s0_w3a | -0.0602 | 0.100 | -0.0609 | 0.095 | yes |
| ppo_volatile_s1_w3a | -0.0814 | 0.029 | -0.0821 | 0.027 | yes |
| ppo_volatile_s2_w3a | -0.0423 | 0.470 | -0.0430 | 0.458 | yes |
| ppo_volatile_s0_w3b | -0.0456 | 0.153 | -0.0462 | 0.152 | yes |
| ppo_volatile_s1_w3b | +0.0471 | 0.467 | +0.0465 | 0.465 | NO |
| ppo_volatile_s2_w3b | -0.0257 | 0.647 | -0.0263 | 0.623 | yes |
| dqn_volatile_s0_d1 | +0.1398 | 0.157 | +0.1392 | 0.154 | NO |
| dqn_volatile_s1_d1 | +0.0324 | 0.307 | +0.0318 | 0.306 | NO |
| dqn_volatile_s2_d1 | -0.0396 | 0.334 | -0.0403 | 0.333 | yes |
| dqn_volatile_s0_d2 | +0.0063 | 0.675 | +0.0057 | 0.688 | yes |
| dqn_volatile_s1_d2 | -0.0729 | 0.032 | -0.0736 | 0.031 | yes |
| dqn_volatile_s2_d2 | -0.0092 | 0.709 | -0.0099 | 0.661 | yes |
| ppo_calm_s0_v1a | -0.0226 | 0.248 | -0.0225 | 0.251 | yes |
| ppo_calm_s1_v1a | -0.0068 | 0.770 | -0.0067 | 0.781 | yes |
| ppo_calm_s2_v1a | -0.0185 | 0.243 | -0.0184 | 0.241 | yes |
| ppo_calm_s0_v1b | +0.0053 | 0.960 | +0.0054 | 0.958 | yes |
| ppo_calm_s1_v1b | -0.0246 | 0.196 | -0.0245 | 0.195 | yes |
| ppo_calm_s2_v1b | -0.0441 | 0.016 | -0.0441 | 0.016 | yes |
| ppo_calm_s0_v2 | -0.0111 | 0.732 | -0.0110 | 0.734 | yes |
| ppo_calm_s1_v2 | +0.0001 | 0.787 | +0.0002 | 0.786 | yes |
| ppo_calm_s2_v2 | +0.0036 | 0.958 | +0.0036 | 0.944 | yes |
| ppo_calm_s0_v3a | +0.0337 | 0.070 | +0.0338 | 0.072 | yes |
| ppo_calm_s1_v3a | -0.0230 | 0.230 | -0.0229 | 0.229 | yes |
| ppo_calm_s2_v3a | -0.0069 | 0.607 | -0.0068 | 0.613 | yes |
| ppo_calm_s0_v3b | -0.0327 | 0.089 | -0.0326 | 0.091 | yes |
| ppo_calm_s1_v3b | -0.0153 | 0.623 | -0.0153 | 0.618 | yes |
| ppo_calm_s2_v3b | -0.0148 | 0.426 | -0.0147 | 0.427 | NO |
| ppo_calm_s0_v4a | -0.0189 | 0.416 | -0.0188 | 0.422 | yes |
| ppo_calm_s1_v4a | -0.0126 | 0.515 | -0.0125 | 0.519 | yes |
| ppo_calm_s2_v4a | -0.0298 | 0.370 | -0.0298 | 0.373 | yes |
| ppo_calm_s3_v4a | -0.0382 | 0.028 | -0.0381 | 0.029 | yes |
| ppo_calm_s4_v4a | -0.0076 | 0.872 | -0.0075 | 0.877 | yes |
| ppo_calm_s0_v4b | +0.0027 | 0.986 | +0.0028 | 0.980 | yes |
| ppo_calm_s1_v4b | -0.0161 | 0.444 | -0.0161 | 0.439 | yes |
| ppo_calm_s2_v4b | -0.0043 | 0.808 | -0.0042 | 0.814 | yes |
| ppo_calm_s0_v5 | -0.0068 | 0.808 | -0.0067 | 0.803 | yes |
| ppo_calm_s1_v5 | -0.0035 | 0.915 | -0.0034 | 0.929 | yes |
| ppo_calm_s2_v5 | -0.0377 | 0.047 | -0.0376 | 0.047 | yes |
| ppo_calm_s0_v6 | +0.0135 | 0.671 | +0.0136 | 0.667 | yes |
| ppo_calm_s1_v6 | +0.0100 | 0.996 | +0.0100 | 0.991 | yes |
| ppo_calm_s2_v6 | -0.0099 | 0.872 | -0.0098 | 0.873 | yes |
| ppo_calm_s0_v6b | -0.0319 | 0.066 | -0.0318 | 0.066 | yes |
| ppo_calm_s1_v6b | +0.0047 | 0.687 | +0.0047 | 0.683 | yes |
| ppo_calm_s2_v6b | -0.0053 | 0.573 | -0.0052 | 0.570 | yes |
| ppo_calm_s0_w3a | +0.0017 | 0.901 | +0.0018 | 0.892 | yes |
| ppo_calm_s1_w3a | -0.0178 | 0.287 | -0.0177 | 0.288 | yes |
| ppo_calm_s2_w3a | -0.0029 | 0.841 | -0.0029 | 0.852 | yes |
| ppo_calm_s0_w3b | -0.0218 | 0.235 | -0.0217 | 0.235 | yes |
| ppo_calm_s1_w3b | -0.0102 | 0.785 | -0.0101 | 0.802 | yes |
| ppo_calm_s2_w3b | -0.0032 | 0.648 | -0.0031 | 0.649 | yes |
| dqn_calm_s0_d1 | +0.0894 | 0.002 | +0.0895 | 0.002 | NO |
| dqn_calm_s1_d1 | -0.0069 | 0.780 | -0.0068 | 0.780 | yes |
| dqn_calm_s2_d1 | -0.0068 | 0.838 | -0.0067 | 0.844 | NO |
| dqn_calm_s0_d2 | +0.1442 | 0.002 | +0.1443 | 0.002 | NO |
| dqn_calm_s1_d2 | +0.1393 | 0.001 | +0.1393 | 0.001 | NO |
| dqn_calm_s2_d2 | +0.0099 | 0.604 | +0.0100 | 0.595 | NO |

## CURRENT RESULTS (B) — CLEAN v3 VERDICT — drift-fixed primary campaign (sealed judgement 2026-07-09)

**Source of record (EXACT ABSOLUTE PATHS — these are the CURRENT post-drift-fix numbers):**
- Costs + p-values, all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_v3/judgement.json`
- Behaviour audit (valid/invalid), all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_v3/behaviour_audit.json`
- Trained agents + logs + learning curves:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_primary_v3/`
- R8a tuning results (2026-07-09):
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_tuning_v3/judgement.json`

SAME 20-run
design, SAME training seeds 0-4, SAME eval block 5e6 as v2 — the ONLY change is the drift-neutralised
move process. So every v2->v3 difference is attributable to the drift fix.

**Frozen §3 verdict: BOUNDARY NULL in all 4 cells** (no cell clears per-seed p<0.01 in >=4/5 seeds AND
pooled <= -0.05). But the clean field DISENTANGLED artifact from signal:

| cell | valid | cheaper | pooled vs adaptive | across-seed t p | read |
|---|---|---|---|---|---|
| PPO volatile | 5/5 | 5/5 | **-0.047** | **0.006** | REAL drift-free signal (sub-threshold) |
| PPO calm     | 5/5 | 3/5 | -0.006 | 0.18  | tie / no edge |
| DQN calm     | 1/5 | 1/1 | -0.002 | n/a   | 4/5 collapse (do-nothing-then-dump) |
| DQN volatile | 2/5 | 2/2 | -0.035 | 0.042 (n=2, fragile) | 3/5 collapse |

**Across-seed test is now AUTO-COMPUTED (2026-07-09), not hand-run.** `step5_judgement.py --mode
screen` attaches an informational `across_seed` block to every cell (pooled edge, one-sided t p, 95%
CI, #cheaper) right next to the frozen EDGE/ESCALATE flag, so a small-but-consistent edge can never be
silently read as "null" (criteria §3a). It also carries `n_seeds_total` + a `trustworthy`/survivorship
flag, so a low p on few survivors (DQN-volatile: p=0.042 but only 2/5 valid) is auto-flagged, NOT an
edge. The sealed `step5_v3/judgement.json` was BACKFILLED from its saved per-seed numbers (no re-eval;
original kept as `judgement_preAcrossSeed_backup.json`); the frozen per-seed verdicts are byte-identical
before/after (asserted). Same statistic §6 already uses for confirmation, now surfaced at screening too.
The running R8a tuning `judgement.json` will be backfilled the same way once its judge finishes.

**FULL per-seed table (v3, both benchmarks).** cost = agent − benchmark, bps (negative = agent
cheaper); p = per-seed Wilcoxon. `vs_fixed` and `vs_adaptive` differ by <=0.0006 bps everywhere
(mean 0.0004): the two TWAP variants are near-identical in this env, so "beat fixed only" would give
the identical boundary null. Source of record: `$S/step5_v3/{judgement,behaviour_audit}.json`.

| run | vs_fixed | p_fixed | vs_adaptive | p_adapt | valid |
|---|---|---|---|---|---|
| ppo_volatile_s0 | -0.0317 | 0.350 | -0.0324 | 0.344 | yes |
| ppo_volatile_s1 | -0.0357 | 0.442 | -0.0364 | 0.433 | yes |
| ppo_volatile_s2 | -0.0892 | 0.012 | -0.0898 | 0.012 | yes |
| ppo_volatile_s3 | -0.0319 | 0.417 | -0.0326 | 0.413 | yes |
| ppo_volatile_s4 | -0.0433 | 0.444 | -0.0439 | 0.425 | yes |
| ppo_calm_s0 | -0.0108 | 0.958 | -0.0107 | 0.952 | yes |
| ppo_calm_s1 | +0.0110 | 0.576 | +0.0111 | 0.567 | yes |
| ppo_calm_s2 | -0.0194 | 0.324 | -0.0193 | 0.324 | yes |
| ppo_calm_s3 | +0.0020 | 0.597 | +0.0020 | 0.582 | yes |
| ppo_calm_s4 | -0.0108 | 0.632 | -0.0107 | 0.637 | yes |
| dqn_calm_s0 | +0.1124 | 0.015 | +0.1125 | 0.015 | NO |
| dqn_calm_s1 | +0.0596 | 0.072 | +0.0597 | 0.069 | NO |
| dqn_calm_s2 | +0.0263 | 0.326 | +0.0264 | 0.324 | NO |
| dqn_calm_s3 | +0.1533 | 0.000 | +0.1534 | 0.000 | NO |
| dqn_calm_s4 | -0.0016 | 0.829 | -0.0015 | 0.827 | yes |
| dqn_volatile_s0 | +0.4176 | 0.001 | +0.4169 | 0.001 | NO |
| dqn_volatile_s1 | -0.0296 | 0.632 | -0.0303 | 0.607 | yes |
| dqn_volatile_s2 | +0.3017 | 0.003 | +0.3011 | 0.003 | NO |
| dqn_volatile_s3 | -0.0389 | 0.383 | -0.0396 | 0.375 | yes |
| dqn_volatile_s4 | -0.0197 | 0.893 | -0.0203 | 0.902 | NO |

**Drifty (v2) -> clean (v3), only the drift removed:**
- **PPO CALM:** pooled -0.019 (p=0.025) -> -0.006 (p=0.18). The calm "edge" was MOSTLY DRIFT; it vanished.
- **PPO VOLATILE:** pooled -0.039 (p=0.066) -> -0.047 (p=0.006); 4/5 -> 5/5 seeds cheaper. The volatile
  edge is GENUINE: it survived drift removal and the across-seed signal FIRMED UP (the drift had been
  adding seed-to-seed noise that masked it). NOT an artifact.

**Interpretation:** the drift fix did exactly its job. In calm (low genuine volatility) the injected
drift dominated and was harvestable by front-loading -> fake edge, now gone. In volatile (rich real
dynamics) PPO has a genuine impact-management edge that the noisy drift had obscured -> now revealed.
DQN still collapses (the L2 do-nothing-then-dump pathology) in most seeds -> clean null/collapse.

**Status:** BOUNDARY NULL under the frozen rule. PPO volatile is a strong, UNCONFOUNDED, sub-threshold
candidate (5/5 seeds, across-seed p=0.006, pooled -0.047 just under the -0.05 floor; 0/5 per-seed at
p<0.01). It was NOTICED in this data, so per the pre-registration it CANNOT be claimed from it -> the
R8b out-of-sample confirmation is the honest test. RQ2/RQ3 now have a genuine regime-specific effect
to attribute. This CLEAN v3 supersedes the drift-confounded v2/R7 as the primary result.

## CURRENT RESULTS (C) — ROBUSTNESS SWEEPS (sealed 2026-07-13, criteria §7): NULL HOLDS ACROSS THE DESIGN SPACE

**Source of record (EXACT ABSOLUTE PATHS, one row per cell — scored files, then the 6 agent
folders for that cell; each agent folder holds model.zip + meta.json + curve.json and self-
describes its order_btc/env_steps):**

| cell | scored numbers (judgement.json + behaviour_audit.json) | trained agents (6 per cell) |
|---|---|---|
| 5 BTC / 300s | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_sweep_b5/` | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_sweep_b5/ppo_{calm,volatile}_s{0,1,2}_v3aB5/` |
| 12.5 BTC / 300s | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_sweep_b12/` | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_sweep_b12/ppo_{calm,volatile}_s{0,1,2}_v3aB12/` |
| 50 BTC / 300s | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_sweep_b50/` | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_sweep_b50/ppo_{calm,volatile}_s{0,1,2}_v3aB50/` |
| 25 BTC / 600s (10-min) | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_sweep_h600/` | `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_sweep_h600/ppo_{calm,volatile}_s{0,1,2}_v3aH600/` |

(The `{calm,volatile}` and `s{0,1,2}` braces enumerate the 6 folders in each cell: 2 regimes x 3
seeds.) 24 runs, 0 training failures; provenance verified in every judgement file (order_btc +
env_steps recorded). Config = the §5-selected PPO lr 1e-3; dev block eval_seed0=5,000,000, n=2000
CRN, audit-before-costs.

**FULL PER-SEED TABLE (vs adaptive-TWAP, bps; * = audit-INVALID; centre point 25 BTC/300s from
`step5_selection_v3` for comparison):**

| cell | volatile s0 s1 s2 | vol pooled | vol across-p | calm s0 s1 s2 | calm pooled | calm across-p |
|---|---|---|---|---|---|---|
| 5 BTC / 300s | +0.0273 +0.0709 +0.0296 | +0.0426 | 0.95 | -0.0219 -0.0250 +0.0045 | -0.0142 | 0.13 |
| 12.5 BTC / 300s | -0.0119* -0.0083 -0.0236 | -0.0159 (2/3 valid) | 0.14 | -0.0227 -0.0478 +0.0007 | -0.0233 | 0.12 |
| 25 BTC / 300s (centre, dev) | (5 seeds) | -0.0628 | 0.0043 | (3 seeds) | +0.0013 | 0.53 |
| 50 BTC / 300s | +0.0089 -0.0125 +0.0109 | +0.0024 | 0.61 | +0.0144 +0.0069 +0.0086 | +0.0100 | 0.98 |
| 25 BTC / 600s (10-min) | -0.0998 +0.0094 -0.0953 | -0.0619 | 0.11 | -0.0090 -0.0227 -0.0434 | -0.0250 | 0.065 |

**§7.5 trigger check (mechanical): NO cell meets the escalation criterion** (pooled <= -0.02 AND
across-seed p < 0.05 AND fully valid). The 10-min volatile cell has a negative pooled (-0.062) but
massive seed variance (two seeds ~-0.10, one +0.01 -> p=0.11): exactly the high-variance,
non-significant pattern §7.5 exists to keep from being over-read. No escalation, no claims.

**THE SIZE-RESPONSE PICTURE (volatile): +0.043 (5 BTC) -> -0.016 (12.5) -> -0.063 (25, dev) ->
+0.002 (50).** NOT monotone. If the 25-BTC dev "edge" had been genuine impact management it should
GROW with order size (more impact, more to manage); instead it is absent at 5, 12.5 AND 50 BTC,
existing only at exactly the size where all development and selection happened. This is a fourth
independent line of evidence (after the sealed confirmation, the three-block diagnostic, and the
winner's-curse re-rankings) that the dev-block edge was selection luck, not skill. At 5 BTC the
agent is actively WORSE than TWAP (+0.043) — with a tiny order there is nothing to manage and
deviation from even pacing only adds noise cost.

**Verdict: the boundary null HOLDS across order sizes 5-50 BTC and at the 10-minute horizon, in
both regimes.** The §1 multi-axis design is complete (cadence sweep pre-registered out).

## CURRENT RESULTS (D) — §7.7 GRID, Parts A+B (sealed judgements 2026-07-15, `step5_grid_*`): NULL IN 20/22 GROUPS; TWO CALM GROUPS TRIGGER THE §7.5 LADDER (escalation NOT yet launched — user decision pending)

Provenance: 66 runs (11 new cells x 2 regimes x 3 seeds), integrity-checked before judging
(all metas parsed: order size / deadline / config / seed / regime / gamma correct; curves
complete to 2M steps). Judged 2026-07-15, one judge per cell with matching
--order-btc/--env-steps (recorded in each judgement.json), screen mode, dev block
eval_seed0=5,000,000, n=2,000 CRN episodes, audit-before-costs. Parse-verified 11/11.

**SOURCE OF TRUTH = the JSON files at the paths below (2026-07-15, user rule: markdown tables
in this doc are a convenience COPY; every number cited in the report must trace to these files).**
Base path `$S` = `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4`.
Per cell: trained agents live in `$S/runs_grid_<cell>/<run>/` (one folder per run:
`model.zip` + `meta.json` + `curve.json`); scored results in `$S/step5_grid_<cell>/judgement.json`
(field `per_run`, one row per run, keyed by run name) + `$S/step5_grid_<cell>/behaviour_audit.json`.

| cell | trained agents (runs) | scored results (judgement + audit) |
|---|---|---|
| 5 BTC / 2.5 min | `$S/runs_grid_b5h150/` | `$S/step5_grid_b5h150/` |
| 12.5 BTC / 2.5 min | `$S/runs_grid_b12h150/` | `$S/step5_grid_b12h150/` |
| 25 BTC / 2.5 min | `$S/runs_grid_b25h150/` | `$S/step5_grid_b25h150/` |
| 50 BTC / 2.5 min | `$S/runs_grid_b50h150/` | `$S/step5_grid_b50h150/` |
| 5 BTC / 10 min | `$S/runs_grid_b5h600/` | `$S/step5_grid_b5h600/` |
| 12.5 BTC / 10 min | `$S/runs_grid_b12h600/` | `$S/step5_grid_b12h600/` |
| 50 BTC / 10 min (TRIGGER, calm) | `$S/runs_grid_b50h600/` | `$S/step5_grid_b50h600/` |
| 5 BTC / 20 min | `$S/runs_grid_b5h1200/` | `$S/step5_grid_b5h1200/` |
| 12.5 BTC / 20 min | `$S/runs_grid_b12h1200/` | `$S/step5_grid_b12h1200/` |
| 25 BTC / 20 min (TRIGGER, calm) | `$S/runs_grid_b25h1200/` | `$S/step5_grid_b25h1200/` |
| 50 BTC / 20 min | `$S/runs_grid_b50h1200/` | `$S/step5_grid_b50h1200/` |

Run naming inside each cell: `ppo_{calm|volatile}_s{0,1,2}_gS{5|12|25|50}H{150|600|1200}`
(the two trigger cells additionally gain `s3`,`s4` calm runs from the §7.5a escalation).
The already-tested 5 cells cited by F9/T4: `$S/step5_selection_v3/` (25 BTC/5-min),
`$S/step5_sweep_{b5,b12,b50}/` (5-min column), `$S/step5_sweep_h600/` (25 BTC/10-min).
Escalation/cross-block results will land in `$S/step5_esc_<cell>/` and `$S/step5_xblock_<cell>/`.

**Audit: 64/66 valid.** Two invalid, both calm at the 2.5-min deadline, both on the
deadline-residual criterion (forced-deadline buy finished >10% of episodes: 16% and 12.5%) —
mild deadline-leaning at the tightest horizon, not the DQN-style collapse:
`ppo_calm_s2_gS25H150`, `ppo_calm_s1_gS50H150`. Excluded from group verdicts per protocol.

**Strict per-seed ESCALATE flag: FALSE in all 22 groups** (no group has >=2 individually
significant seeds with pooled <= -0.05).

**§7.5 TRIGGER (pooled <= -0.02 AND across-seed p < 0.05, fully valid): fires in TWO groups,
both CALM:**
- **50 BTC / 10-min / calm**: pooled -0.0539 bps, across-seed p=0.014, 3/3 cheaper,
  95% CI [-0.0935, -0.0143].
- **25 BTC / 20-min / calm**: pooled -0.0629 bps, across-seed p=0.0003, 3/3 cheaper,
  95% CI [-0.0691, -0.0567] (strikingly tight).

**Interpretation context (logged before any escalation runs):**
1. These are DEV-BLOCK (5e6) numbers — the same block on which the 25-BTC volatile "edge"
   looked real (p=0.0043, 5 seeds) and then failed BOTH sealed tests. The §7.5 ladder
   (escalate to 5 seeds -> cross-block 6e6 -> at most ONE sealed test) exists precisely for
   this situation; step (ii) is the check that would have caught the earlier block-luck illusion.
2. Regime inversion: the original candidate edge was VOLATILE-only; these triggers are
   CALM-only (volatile in the same cells: 50/10min p=0.083; 25/20min pooled POSITIVE +0.016).
   A real mechanism appearing only in calm at two scattered cells, after calm showed nothing
   at the primary cell, has no pre-stated mechanism story.
3. Multiplicity: 22 groups screened; under a global null the §5-rule-3 condition is expected
   to fire roughly once by chance (p<0.05 alone: ~1.1 expected across 22 one-sided tests;
   the -0.02 floor tightens that, block-luck correlation across same-block cells loosens it).
   Two firings including one at p=0.0003 is worth the pre-registered follow-up, nothing more.
4. No edge claim can arise from these dev-block results directly (§7.5 interpretation cap).
   The §6.8 boundary-null headline STANDS unless a §7.5-triggered sealed confirmation passes.

**Next per §7.5 (pending user go-ahead): (i) escalate BOTH triggered groups to 5 seeds
(train seeds 3,4 calm for the two cells = 4 runs); (ii) cross-block replication of the
escalated groups on eval_seed0=6,000,000 (n=2,000, first use of the reserve block); (iii) only
if BOTH steps survive: at most ONE newly pre-registered sealed confirmation (§6 family
disclosure applies).**

### FULL PER-SEED TABLE — all 66 grid runs, both benchmarks (sealed judgements 2026-07-15, `step5_grid_*`)

| size (BTC) | horizon | regime | seed | vs fixed (bps) | p_fixed | vs adaptive (bps) | p_adaptive | valid |
|---|---|---|---|---|---|---|---|---|
| 5.0 | 2.5min | calm | 0 | +0.0036 | 0.731 | -0.0029 | 0.952 | yes |
| 5.0 | 2.5min | calm | 1 | -0.0252 | 0.026 | -0.0317 | 0.015 | yes |
| 5.0 | 2.5min | calm | 2 | +0.0130 | 0.383 | +0.0065 | 0.716 | yes |
| 5.0 | 2.5min | volatile | 0 | -0.0102 | 0.782 | -0.0103 | 0.753 | yes |
| 5.0 | 2.5min | volatile | 1 | +0.0017 | 0.856 | +0.0016 | 0.882 | yes |
| 5.0 | 2.5min | volatile | 2 | -0.0309 | 0.207 | -0.0310 | 0.207 | yes |
| 12.5 | 2.5min | calm | 0 | -0.0010 | 0.955 | -0.0012 | 0.950 | yes |
| 12.5 | 2.5min | calm | 1 | +0.0010 | 0.931 | +0.0009 | 0.961 | yes |
| 12.5 | 2.5min | calm | 2 | +0.0102 | 0.566 | +0.0101 | 0.574 | yes |
| 12.5 | 2.5min | volatile | 0 | +0.0177 | 0.426 | +0.0191 | 0.382 | yes |
| 12.5 | 2.5min | volatile | 1 | -0.0232 | 0.293 | -0.0219 | 0.337 | yes |
| 12.5 | 2.5min | volatile | 2 | +0.0392 | 0.198 | +0.0406 | 0.193 | yes |
| 25.0 | 2.5min | calm | 0 | -0.0069 | 0.562 | -0.0069 | 0.564 | yes |
| 25.0 | 2.5min | calm | 1 | +0.0127 | 0.308 | +0.0127 | 0.304 | yes |
| 25.0 | 2.5min | calm | 2 | -0.0195 | 0.263 | -0.0195 | 0.269 | **NO** |
| 25.0 | 2.5min | volatile | 0 | -0.0059 | 0.900 | -0.0069 | 0.870 | yes |
| 25.0 | 2.5min | volatile | 1 | -0.0159 | 0.505 | -0.0168 | 0.484 | yes |
| 25.0 | 2.5min | volatile | 2 | -0.0156 | 0.758 | -0.0166 | 0.719 | yes |
| 50.0 | 2.5min | calm | 0 | +0.0071 | 0.724 | +0.0069 | 0.733 | yes |
| 50.0 | 2.5min | calm | 1 | +0.0181 | 0.114 | +0.0179 | 0.116 | **NO** |
| 50.0 | 2.5min | calm | 2 | +0.0053 | 0.811 | +0.0051 | 0.816 | yes |
| 50.0 | 2.5min | volatile | 0 | +0.0243 | 0.474 | +0.0244 | 0.476 | yes |
| 50.0 | 2.5min | volatile | 1 | +0.0442 | 0.110 | +0.0443 | 0.112 | yes |
| 50.0 | 2.5min | volatile | 2 | +0.0214 | 0.582 | +0.0215 | 0.587 | yes |
| 5.0 | 10min | calm | 0 | +0.0027 | 0.891 | +0.0105 | 0.831 | yes |
| 5.0 | 10min | calm | 1 | -0.0081 | 0.799 | -0.0004 | 0.836 | yes |
| 5.0 | 10min | calm | 2 | +0.0136 | 0.538 | +0.0213 | 0.487 | yes |
| 5.0 | 10min | volatile | 0 | -0.0004 | 0.989 | -0.0448 | 0.471 | yes |
| 5.0 | 10min | volatile | 1 | -0.0383 | 0.658 | -0.0827 | 0.096 | yes |
| 5.0 | 10min | volatile | 2 | +0.0569 | 0.396 | +0.0125 | 0.948 | yes |
| 12.5 | 10min | calm | 0 | -0.0172 | 0.490 | -0.0217 | 0.385 | yes |
| 12.5 | 10min | calm | 1 | -0.0048 | 0.901 | -0.0093 | 0.998 | yes |
| 12.5 | 10min | calm | 2 | -0.0178 | 0.464 | -0.0223 | 0.434 | yes |
| 12.5 | 10min | volatile | 0 | -0.0163 | 0.637 | -0.0159 | 0.663 | yes |
| 12.5 | 10min | volatile | 1 | -0.0575 | 0.139 | -0.0570 | 0.158 | yes |
| 12.5 | 10min | volatile | 2 | -0.0063 | 0.777 | -0.0058 | 0.737 | yes |
| 50.0 | 10min | calm | 0 | -0.0453 | 0.066 | -0.0453 | 0.066 | yes |
| 50.0 | 10min | calm | 1 | -0.0723 | 0.017 | -0.0723 | 0.017 | yes |
| 50.0 | 10min | calm | 2 | -0.0441 | 0.138 | -0.0441 | 0.138 | yes |
| 50.0 | 10min | volatile | 0 | -0.0215 | 0.597 | -0.0223 | 0.585 | yes |
| 50.0 | 10min | volatile | 1 | -0.1532 | 0.018 | -0.1540 | 0.017 | yes |
| 50.0 | 10min | volatile | 2 | -0.0692 | 0.213 | -0.0700 | 0.206 | yes |
| 5.0 | 20min | calm | 0 | +0.0842 | 0.024 | +0.0362 | 0.283 | yes |
| 5.0 | 20min | calm | 1 | +0.0792 | 0.028 | +0.0312 | 0.495 | yes |
| 5.0 | 20min | calm | 2 | +0.0818 | 0.022 | +0.0339 | 0.268 | yes |
| 5.0 | 20min | volatile | 0 | +0.0545 | 0.304 | +0.1334 | 0.106 | yes |
| 5.0 | 20min | volatile | 1 | -0.0422 | 0.984 | +0.0367 | 0.478 | yes |
| 5.0 | 20min | volatile | 2 | -0.0545 | 0.368 | +0.0244 | 0.846 | yes |
| 12.5 | 20min | calm | 0 | +0.0313 | 0.413 | +0.0347 | 0.346 | yes |
| 12.5 | 20min | calm | 1 | -0.0056 | 0.918 | -0.0022 | 0.821 | yes |
| 12.5 | 20min | calm | 2 | +0.0208 | 0.632 | +0.0242 | 0.595 | yes |
| 12.5 | 20min | volatile | 0 | +0.1971 | 0.005 | +0.1976 | 0.005 | yes |
| 12.5 | 20min | volatile | 1 | +0.1652 | 0.007 | +0.1658 | 0.005 | yes |
| 12.5 | 20min | volatile | 2 | +0.1274 | 0.075 | +0.1279 | 0.065 | yes |
| 25.0 | 20min | calm | 0 | -0.0612 | 0.086 | -0.0601 | 0.090 | yes |
| 25.0 | 20min | calm | 1 | -0.0651 | 0.126 | -0.0640 | 0.134 | yes |
| 25.0 | 20min | calm | 2 | -0.0658 | 0.060 | -0.0647 | 0.061 | yes |
| 25.0 | 20min | volatile | 0 | -0.0024 | 0.979 | -0.0056 | 0.996 | yes |
| 25.0 | 20min | volatile | 1 | -0.0178 | 0.650 | -0.0210 | 0.625 | yes |
| 25.0 | 20min | volatile | 2 | +0.0764 | 0.213 | +0.0732 | 0.220 | yes |
| 50.0 | 20min | calm | 0 | -0.0472 | 0.160 | -0.0470 | 0.163 | yes |
| 50.0 | 20min | calm | 1 | +0.0567 | 0.083 | +0.0569 | 0.082 | yes |
| 50.0 | 20min | calm | 2 | -0.0610 | 0.086 | -0.0608 | 0.087 | yes |
| 50.0 | 20min | volatile | 0 | -0.0757 | 0.218 | -0.0766 | 0.217 | yes |
| 50.0 | 20min | volatile | 1 | +0.0434 | 0.428 | +0.0424 | 0.443 | yes |
| 50.0 | 20min | volatile | 2 | +0.0801 | 0.386 | +0.0791 | 0.393 | yes |

### GROUP VERDICTS — 22 groups (11 cells x 2 regimes)

| size (BTC) | horizon | regime | valid/total | cheaper vs adaptive | pooled vs adaptive (bps) | pooled vs fixed (bps) | across-seed p (adaptive) | 95% CI (adaptive) | ESCALATE | §7.5 trigger |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 10min | calm | 3/3 | 1/3 | +0.0105 | +0.0027 | 0.8819 | [-0.0165, +0.0374] | False | no |
| 5.0 | 10min | volatile | 3/3 | 2/3 | -0.0383 | +0.0060 | 0.1503 | [-0.1574, +0.0808] | False | no |
| 12.5 | 10min | calm | 3/3 | 3/3 | -0.0178 | -0.0133 | 0.0261 | [-0.0359, +0.0004] | False | no |
| 12.5 | 10min | volatile | 3/3 | 3/3 | -0.0262 | -0.0267 | 0.1180 | [-0.0937, +0.0412] | False | no |
| 50.0 | 10min | calm | 3/3 | 3/3 | -0.0539 | -0.0539 | 0.0140 | [-0.0935, -0.0143] | False | **YES** |
| 50.0 | 10min | volatile | 3/3 | 3/3 | -0.0821 | -0.0813 | 0.0833 | [-0.2477, +0.0835] | False | no |
| 5.0 | 2.5min | calm | 3/3 | 2/3 | -0.0094 | -0.0029 | 0.2498 | [-0.0588, +0.0400] | False | no |
| 5.0 | 2.5min | volatile | 3/3 | 2/3 | -0.0132 | -0.0131 | 0.1492 | [-0.0542, +0.0277] | False | no |
| 12.5 | 2.5min | calm | 3/3 | 1/3 | +0.0032 | +0.0034 | 0.7761 | [-0.0116, +0.0181] | False | no |
| 12.5 | 2.5min | volatile | 3/3 | 1/3 | +0.0126 | +0.0112 | 0.7186 | [-0.0662, +0.0914] | False | no |
| 25.0 | 2.5min | calm | 2/3 | 1/2 | +0.0029 | +0.0029 | 0.5919 | [-0.1214, +0.1272] | False | no |
| 25.0 | 2.5min | volatile | 3/3 | 3/3 | -0.0134 | -0.0125 | 0.0273 | [-0.0275, +0.0007] | False | no |
| 50.0 | 2.5min | calm | 2/3 | 0/2 | +0.0060 | +0.0062 | 0.9525 | [-0.0055, +0.0175] | False | no |
| 50.0 | 2.5min | volatile | 3/3 | 0/3 | +0.0301 | +0.0300 | 0.9740 | [-0.0007, +0.0608] | False | no |
| 5.0 | 20min | calm | 3/3 | 0/3 | +0.0338 | +0.0818 | 0.9991 | [+0.0276, +0.0400] | False | no |
| 5.0 | 20min | volatile | 3/3 | 0/3 | +0.0648 | -0.0141 | 0.8996 | [-0.0835, +0.2131] | False | no |
| 12.5 | 20min | calm | 3/3 | 1/3 | +0.0189 | +0.0155 | 0.8863 | [-0.0284, +0.0662] | False | no |
| 12.5 | 20min | volatile | 3/3 | 0/3 | +0.1638 | +0.1633 | 0.9926 | [+0.0771, +0.2504] | False | no |
| 25.0 | 20min | calm | 3/3 | 3/3 | -0.0629 | -0.0640 | 0.0003 | [-0.0691, -0.0567] | False | **YES** |
| 25.0 | 20min | volatile | 3/3 | 2/3 | +0.0156 | +0.0188 | 0.6764 | [-0.1100, +0.1411] | False | no |
| 50.0 | 20min | calm | 3/3 | 2/3 | -0.0169 | -0.0172 | 0.3466 | [-0.1768, +0.1429] | False | no |
| 50.0 | 20min | volatile | 3/3 | 1/3 | +0.0150 | +0.0160 | 0.6099 | [-0.1873, +0.2173] | False | no |


### §7.5a LADDER — STEP (i) ESCALATION RESULTS (official, 2026-07-15): BOTH GROUPS SURVIVE

Executed as pre-registered in criteria §7.5a: seeds 3,4 trained per group (integrity ALL PASS),
full dirs re-judged on the dev block (5e6, n=2000) into `$S/step5_esc_b50h600/` and
`$S/step5_esc_b25h1200/` (SOURCE OF TRUTH; the original `step5_grid_*` trigger records untouched).
DETERMINISM CHECK PASSED: every original seed's numbers byte-identical between the grid and
escalation judgements.

| group (calm) | seeds (vs adaptive) | pooled | across-seed p | cheaper | 95% CI | criterion (i) |
|---|---|---|---|---|---|---|
| 50 BTC / 10-min | -0.0453, -0.0723, -0.0441, **-0.0346**, **-0.0196** (bold = new) | -0.0432 | 0.0037 | 5/5 | [-0.0671, -0.0193] | **SURVIVES** |
| 25 BTC / 20-min | -0.0601, -0.0640, -0.0647, **-0.0721**, **-0.0435** | -0.0609 | 0.0001 | 5/5 | [-0.0741, -0.0477] | **SURVIVES** |

All 10 calm seeds valid; all 10 cheaper than BOTH benchmarks (vs-fixed within 0.001 of
vs-adaptive throughout). Honest note: both new-seed pairs came in WEAKER than their group's
original three (dilution consistent with mild winner's-curse on the trigger), but not weak
enough to break either group. Step (ii) followed (below).

### §7.5a LADDER — STEP (ii) CROSS-BLOCK RESULTS (official, 2026-07-15): **BOTH GROUPS FAIL.
LADDER CLOSED. NO SEALED TEST SPENT. THE GRID IS A NULL ACROSS THE ENTIRE DESIGN SPACE.**

Same 10 agents, NO retraining, judged at eval_seed0=6,000,000 (reserve block, FIRST USE —
verified in each judgement.json), n=2000, audit-before-costs. SOURCE OF TRUTH:
`$S/step5_xblock_b50h600/` + `$S/step5_xblock_b25h1200/` (judgement.json + behaviour_audit.json).

| group (calm) | dev block (5 seeds) | RESERVE block (same 5 seeds) | criterion (ii) |
|---|---|---|---|
| 50 BTC / 10-min | -0.0432, p=0.0037, 5/5 cheaper | **-0.0095, p=0.17, 4/5 cheaper, CI [-0.034, +0.015]** | **FAILS** |
| 25 BTC / 20-min | -0.0609, p=0.0001, 5/5 cheaper | **+0.0177, p=0.96, 1/5 cheaper, CI [-0.003, +0.038]** | **FAILS** |

Per-seed collapse (vs adaptive, dev -> reserve):
- 50/10min calm: s0 -0.045->-0.003; s1 -0.072->+0.012; s2 -0.044->-0.042; s3 -0.035->-0.006;
  s4 -0.020->-0.010. (Volatile, informational: all three flipped POSITIVE, +0.061..+0.092.)
- 25/20min calm: s0 -0.060->+0.012; s1 -0.064->+0.040; s2 -0.065->-0.006; s3 -0.072->+0.024;
  s4 -0.044->+0.018. All 10 runs valid on the reserve block.

**Interpretation (goes to writeup_arguments §L):** this is the THIRD and cleanest demonstrated
evaluation-block illusion of the project — a 5-seed group at p=0.0001 with CI [-0.074,-0.048]
on the dev block is WORSE than the benchmark on the very next untouched block. Seed agreement
cannot detect block luck; only fresh data can. The §7.5 ladder did exactly its job: caught it
for the cost of 4 training runs + 2 evaluations, without spending a third sealed test. The
§6.8 boundary-null headline STANDS, now robust across the full 4x4 size x deadline grid, both
regimes, and two evaluation blocks. Note the 10-min-column caveat logged before the outcome
(cells in a column share episode draws) proved out: the reserve block reversed the whole
pattern, calm and volatile alike.

## CURRENT RESULTS (E) — §7.7 PART D DQN CROSS-SETTING PROBE (sealed judgements 2026-07-16): THE COLLAPSE IS SYSTEMATIC AND SIZE-DRIVEN; NO COST TRIGGER ANYWHERE

Executed exactly as registered in criteria §7.7 Part D (finalised 2026-07-15 before the runs): 18 runs, DQN BASE config, 3 cells x 2 regimes x 3 seeds, integrity ALL PASS, judged on the dev block (5e6, n=2000), audit-before-costs.
**SOURCE OF TRUTH (user path rule):** `$S/runs_dqnprobe_{b5h150,b25h150,b25h1200}/` (agents; run naming `dqn_{regime}_s{seed}_dqS{5|25}H{150|1200}`) and `$S/step5_dqnprobe_{b5h150,b25h150,b25h1200}/judgement.json` + `behaviour_audit.json` (`$S` = `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4`).

### FULL TABLE — all 18 probe runs (audit = primary endpoint; costs secondary)

| cell | regime | seed | do-nothing % | forced-buy % | audit | vs fixed (bps) | p | vs adaptive (bps) | p |
|---|---|---|---|---|---|---|---|---|---|
| 5 BTC / 2.5-min | calm | 0 | 92.6 | 43.0 | **COLLAPSED** | -0.0221 | 0.583 | -0.0287 | 0.425 |
| 5 BTC / 2.5-min | calm | 1 | 35.9 | 21.5 | **COLLAPSED** | -0.0184 | 0.343 | -0.0250 | 0.159 |
| 5 BTC / 2.5-min | calm | 2 | 6.1 | 1.5 | valid | +0.0101 | 0.457 | +0.0036 | 0.824 |
| 5 BTC / 2.5-min | volatile | 0 | 7.0 | 0.0 | valid | +0.0318 | 0.351 | +0.0316 | 0.309 |
| 5 BTC / 2.5-min | volatile | 1 | 42.5 | 0.0 | valid | -0.0539 | 0.073 | -0.0540 | 0.176 |
| 5 BTC / 2.5-min | volatile | 2 | 0.7 | 0.0 | valid | -0.0405 | 0.165 | -0.0406 | 0.041 |
| 25 BTC / 2.5-min | calm | 0 | 82.2 | 100.0 | **COLLAPSED** | +0.0258 | 0.501 | +0.0258 | 0.500 |
| 25 BTC / 2.5-min | calm | 1 | 57.3 | 18.0 | **COLLAPSED** | -0.0090 | 0.651 | -0.0089 | 0.646 |
| 25 BTC / 2.5-min | calm | 2 | 12.6 | 67.0 | **COLLAPSED** | -0.0223 | 0.179 | -0.0223 | 0.179 |
| 25 BTC / 2.5-min | volatile | 0 | 17.4 | 48.5 | **COLLAPSED** | -0.0400 | 0.555 | -0.0410 | 0.529 |
| 25 BTC / 2.5-min | volatile | 1 | 46.5 | 12.5 | **COLLAPSED** | -0.0513 | 0.253 | -0.0523 | 0.247 |
| 25 BTC / 2.5-min | volatile | 2 | 36.6 | 71.5 | **COLLAPSED** | -0.0097 | 0.847 | -0.0106 | 0.828 |
| 25 BTC / 20-min | calm | 0 | 43.0 | 53.5 | **COLLAPSED** | -0.0529 | 0.083 | -0.0518 | 0.088 |
| 25 BTC / 20-min | calm | 1 | 43.0 | 87.0 | **COLLAPSED** | -0.0030 | 0.864 | -0.0019 | 0.866 |
| 25 BTC / 20-min | calm | 2 | 54.1 | 70.0 | **COLLAPSED** | -0.0473 | 0.341 | -0.0462 | 0.346 |
| 25 BTC / 20-min | volatile | 0 | 15.8 | 8.5 | valid | -0.2256 | 0.027 | -0.2288 | 0.025 |
| 25 BTC / 20-min | volatile | 1 | 30.0 | 0.0 | valid | -0.0507 | 0.436 | -0.0539 | 0.427 |
| 25 BTC / 20-min | volatile | 2 | 24.6 | 77.5 | **COLLAPSED** | -0.0161 | 0.605 | -0.0193 | 0.570 |

### VERDICTS + INTERPRETATION (argument bank §N)

Collapse rates: **5 BTC/2.5-min: 4/6 valid; 25 BTC/2.5-min: 0/6; 25 BTC/20-min: 2/6**
(reference: 25 BTC/5-min primary campaign = 3/10). Cost verdicts: no group triggers anything —
the only fully-valid group (5 BTC/2.5-min volatile) pools -0.021 at p=0.26 (null); every
25-BTC group lacks enough valid seeds for a verdict. Findings, stated before any write-up spin:
1. **Systematic, not setting-specific**: at the primary size DQN collapses at EVERY deadline
   tried (2.5/5/20 min). The "would DQN work elsewhere?" question now has a data answer.
2. **Order size is the driver, not idle room**: the shortest deadline at 25 BTC is the WORST
   cell (0/6), refuting the pre-stated idle-room hypothesis; at 5 BTC DQN mostly trades
   properly. Points to the harder credit-assignment problem when trades carry real impact.
3. **Calm-regime concentration**: 1/9 calm agents valid vs 5/9 volatile, consistent with the
   primary campaign.
No escalation, no trigger, nothing further owed; Part D CLOSED. Figure candidate F20 (collapse
rate by setting) added to the manifest.

## REMAINING ROADMAP (consolidated 2026-07-09) — SINGLE SOURCE OF TRUTH for what is left

Everything remaining, in order, with each step's pre-registration home. (Previously this was
scattered across criteria §1/§5/§6, the audit file, and R8/R9 here; consolidated now.)

> **AUTHORITATIVE STEP-BY-STEP ORDER (updated 2026-07-14). No time-gating: ordered by dependency
> and value only; every item is DONE regardless of schedule (see [[no-manufactured-timelines]]).**
>
> CURRENT STATE (2026-07-15): grid 66/66 trained AND JUDGED (integrity check all-pass first; see
> CURRENT RESULTS (D)): null in 20/22 groups, strict ESCALATE false everywhere, but TWO CALM
> groups fire the §7.5 trigger (50 BTC/10-min: pooled -0.0539, p=0.014; 25 BTC/20-min: pooled
> -0.0629, p=0.0003; both 3/3 cheaper). §7.5 LADDER EXECUTED + CLOSED (2026-07-15, §7.5a/§7.5b):
> both groups SURVIVED escalation to 5 seeds (p=0.0037 / p=0.0001) then BOTH FAILED the
> reserve block 6e6 first use (-0.0095 p=0.17; +0.0177 p=0.96 SIGN-FLIPPED). NO sealed test
> spent. GRID VERDICT FINAL: null across the whole 4x4 design space; the triggers were
> block luck (third demonstrated illusion; argument bank §L5; F19 planned).
> Argument bank updated (§K benchmark justification, §L+L5 grid findings). Figures F1-F8
> BUILT (2/4/5 at v5, F7 at v3 two-panel). Tables T1/T2/T3/T8 BUILT; T9/T10 planned. L2: fill-in
> plan + sealed-test protocol pre-registered (`l2_test_protocol.md`); L2 sealed test never run.
>
> NEXT, IN ORDER:
> 1. Grid finishes → JUDGE the 11 cells (one judge per `runs_grid_*` dir with that cell's
>    --order-btc/--env-steps, criteria §7.7) → verify all 11 judgement.json PARSE → log the FULL
>    66-row per-seed table here → apply §7.5 MECHANICALLY (no trigger → null across the design
>    space, grid closed; any trigger → escalate → reserve block 6e6 → at most ONE new sealed test;
>    apparent edges always reported with status either way) → F9 heatmap + T4 table.
> 1b. DQN cross-setting probe DECISION (criteria §7.7 Part D, added 2026-07-15): after the grid
>    verdict, user decides Option A (scope the DQN claim to the primary setting) vs Option B
>    (small pre-registered DQN probe; recommendation on record = B). Exact cells finalised in
>    Part D before any probe run.
> 2. L2 step (protocol pre-registered 2026-07-14 in `reports/l2_test_protocol.md`), in order:
>    (a) TRAIN THE 19 FILL-IN RUNS (user decision 2026-07-14: no half-complete panels in the
>    report) — ppo 96.57 @ 1-min (5 seeds), dqn 96.57 @ 1-min (4 seeds), dqn 193.13 @ 10-s
>    (5), dqn 386.27 @ 10-s (5); same configs as sibling runs. (b) Validate them. (c) ONE-SHOT
>    SEALED TEST of ALL 70 agents together, paired vs TWAP on each agent's own test split, all
>    arms reported with multiplicity stated. RAM caveat: L2 data loading has frozen this
>    machine before ([[ram-safe-chunked-processing]]) — single-agent footprint measured first,
>    sequential by default. → L2 test-set table (T9/T7 test columns) + figures L1-L5.
> 3. Per-episode cost re-eval (deterministic) → F10 distributions + T6 descriptive-stats table.
> 4. MEASURED-SIGNAL EXTENSION (writeup_arguments.md §J) — the only remaining experiment that can
>    change the headline; pre-register before running → F13.
> 5. Liquidity study (independent, parallel) → F11; feeds VWAP.
> 6. Almgren-Chriss + VWAP benchmarks → benchmark figure.
> 7. RQ3 attribution — AFTER the extension verdict (its labelling depends on the outcome).
> 8. Part C CADENCE check — a real pending step, NOT to be discarded. Needs the env change first
>    (make decision-frequency a parameter, re-derive gamma, re-check baselines/audit, tests), then
>    12 runs at the primary cell {0.5s, 2s} (criteria §7.7 Part C).
> Throughout: write-up; final figure/table curation + title/framing at report assembly.
>
> FIGURE/TABLE TRACKER: `reports/figures/FIGURES_TABLES_MANIFEST.md`. GRID SELF-CHECK if this reads
> stale: look for `$S/step5_grid_*/judgement.json`; else count meta.json in `$S/runs_grid_*` (6 per
> dir × 11 dirs = training done).

*(The numbered list below is the 2026-07-09 consolidation, kept as the record of each completed
step. Where its sequencing notes differ from the AUTHORITATIVE block above, the block above
governs — stamped 2026-07-15.)*

1. **R8a tuning + selection. COMPLETE (2026-07-10, sealed in `$S/step5_selection_v3/`).** 60-run
   screen + 38-run selection batch (escalations to 5 seeds, V6/V6b 10M arms, Wave-3 combos), 98
   runs total, 0 training failures. **WINNER (§5 rule 2, mechanical): ppo_volatile_v3a (lr 1e-3),
   pooled -0.0628, across-seed p=0.0043, 5/5 valid** — overtook the 3-seed screen leader v3b once
   escalated; V6/V6b rejected ('needed longer' is closed). Full tables + findings: CURRENT
   RESULTS (A) above. Stage records: 60-run screen table in the HISTORICAL RECORD below;
   protocol resolutions in criteria §5c. [criteria §5 + §5c]
2. **R8b out-of-sample confirmation — COMPLETE (2026-07-11). VERDICT: FAIL / DID NOT REPLICATE.**
   5 fresh-seed (5-9) retrains of ppo_volatile_v3a, judged ONCE on the sealed 9e6 block per §6.4:
   pooled -0.0023 bps, across-seed p=0.38, 3/5 cheaper, all seeds valid -> PASS=false. The project
   headline is now a BOUNDARY NULL (pre-committed §6.5/§6.7 M3 handling; no re-runs). Full table +
   interpretation: CURRENT RESULTS (0) above. Raw: `$S/step5_confirm_v3a/`. [criteria §6 + §6.7]
3. **Robustness sweeps — COMPLETE (2026-07-13, criteria §7). NULL HOLDS EVERYWHERE.** 24 runs, 0
   failures; no cell meets the §7.5 escalation trigger; size-response NOT monotone (edge absent at
   5/12.5/50 BTC, agent actively worse at 5 BTC) -> fourth independent line of evidence the 25-BTC
   dev edge was selection luck. Full table + interpretation: CURRENT RESULTS (C) above. Raw:
   `$S/step5_sweep_{b5,b12,b50,h600}/`. [criteria §1 + §7]:
   - **Order-size ladder: 5, 12.5, 25, 50 BTC** (primary 25). Tests whether the volatile edge GROWS
     with order size (bigger order -> more impact -> more room) or is size-specific. Valuable either
     way: if edge, shows scaling; if null, shows the null holds across sizes (the L2 three-axis
     pattern).
   - **10-minute horizon variant** (primary is 5-min / 300 s), primary size only.
   - Cadence stays 1 decision/second (NO cadence sweep planned for the QRM track; that was an L2-era
     lever).
4. **RQ3 per-regime attribution (THE CONTRIBUTION):** SHAP + ablation on the selected agent -> which
   order-book signal drives the volatile edge, and why calm has none. [criteria §1 features]
5. **Market liquidity + volume profile study (Carlo, 2026-07-09).** From the L4 trades, characterise
   intra-day and intra-week volume and liquidity. Doubles as market description AND supplies the
   volume curve the VWAP benchmark needs.
6. **Add Almgren-Chriss + VWAP benchmarks**, recalibrated to THIS env (AC: sigma/eta from the QRM,
   lambda swept to AC's best; VWAP: causal participation-rate driven by the step-5 profile); re-run
   the comparison. [audit C5]
7. **Write-up + finalise docs.** Include DESCRIPTIVE STATISTICS in the results section (full cost
   DISTRIBUTIONS per agent, benchmark, and regime, not just means; Carlo 2026-07-09). R9: fold final
   numbers into BUILD_PLAN/HANDOVER; sync web copies.

Sequencing: step 2 (confirmation) is the key result and precedes heavy investment in 3-4 on a
possibly-non-replicating edge; but step 3 (size ladder) is worth running regardless of 2's outcome.
Step 5 (AC/VWAP) is last because both must be calibrated to the final frozen env.

**Noted as possible future work / documented limitation (NOT scheduled): maker/taker + limit-order
execution.** Our agent is taker-only (market buys), the standard convention in the RL-execution
literature. Letting it POST limit orders (provide liquidity, earn the maker rebate, avoid crossing
the spread) is a genuine real-world lever but a substantial rebuild that changes the research question
(execution + liquidity provision). Decision 2026-07-09: keep as a documented limitation, do not build.

## INVARIANTS (do not violate while executing)

- The L2 test set stays sealed. Old run dirs are kept, never cited as results.
- Every criteria change is logged in reports/qrm_step4_criteria.md with date + rationale
  BEFORE the affected run executes. Frozen thresholds (§3) do NOT change.
- Audit before costs, always. CRN seed blocks unchanged (train base seed*1e7; curve 1e6;
  judgement 5e6). No new variants beyond the §5 table without a new documented mechanism.

## WRITE-UP LIMITATIONS (ready-to-use prose, no em dashes)

> **The full write-up ARGUMENT BANK lives in `reports/writeup_arguments.md`** (2026-07-11):
> environment signal structure + scope, why the null is not vacuous, testing-soundness defenses,
> the anatomy of the spurious edge, two-track complementarity, future work, contribution framing.
> This section keeps only the drafted limitation paragraphs; the reasoning behind them is there.

**Market-order-only execution (maker/taker limitation).** The agent in this study executes only
through market orders, which take liquidity by crossing the spread. This follows the standard
convention in the reinforcement-learning execution literature, where the problem is framed as
scheduling a sequence of market orders to minimise implementation shortfall against benchmarks such
as TWAP and Almgren-Chriss. A real execution desk can also provide liquidity by posting passive limit
orders, which avoid crossing the spread and earn the maker rebate that venues such as Hyperliquid
offer, at the cost of an uncertain fill. Modelling this would require the agent to choose not only how
much to trade but whether to take or to provide liquidity, to track its own resting orders in the
queue, and to face the risk that a passive order does not fill and must later be completed at a worse
price. This is a distinct and more complex problem that overlaps with market making, and it is left
for future work. The results here should therefore be read as the performance of an execution policy
restricted to market orders, which is the setting the benchmarks and the wider literature also assume.

---

# ===================== HISTORICAL RECORD (audit trail — kept verbatim) =====================

Everything below is COMPLETED work or SUPERSEDED results, preserved for the record. NEVER cite
numbers from below this banner as current; current numbers live in CURRENT RESULTS (A)/(B) above.

**Origin of this file** (its first role, 2026-07-07 — kept for context):
**Context for a fresh session:** a three-agent adversarial code review (2026-07-07) of the
reactive-QRM experiment found confirmed bugs + one design confound. ALL Step-5 results to
date (primary 20-run campaign, Wave-1/2/3 tuning runs, their judgements) are RECLASSIFIED
as engineering shakedown, NOT evidence. This file is the authoritative worklist: fix →
re-gate → re-train → re-judge. Update the STATUS lines in place as steps complete.

## THE CONFIRMED ISSUES (with locations)

- **I1 (design confound — drift):** move_process_{calm,volatile}.npz have positive mean
  (+0.00429 / +0.01493 ticks per 0.5s = +$2.6 / +$9.0 expected per 300s episode; verified
  directly). A buyer is systematically punished for waiting: always-2.0x beats adaptive
  TWAP by −0.068 (calm) / −0.213 (volatile) bps — past the ±0.05 materiality bar. The
  agents' "null" is confounded (they back-loaded into rising prices). USER DECISION
  (approved 2026-07-07): CENTER the move distributions per regime (zero mean, preserving
  everything else), disclose the measured drift in the write-up as a resampling artifact
  of i.i.d. replay (December's realized direction is not an execution signal).
- **I2 (env bug — deadline over-buy):** reactive_env.py step()/_finalize(): finalize buys
  remaining+carry, but carry is a sub-account of remaining (fills decrement both) →
  re-buys the banked fraction + ceil rounds up a phantom unit; paced policies over-execute
  (~+0.45 BTC on 25; 188% on a toy 0.6 BTC order); ALSO inflated deadline_residual_btc
  (~3x) which contaminated the behaviour audit. FIX: finalize buys remaining ONLY; residual
  metric = remaining only; carry increment capped by (remaining − carry) (root cause,
  reactive_env.py:220).
- **I3 (env bug — per-depth units):** _buy_units/_finalize convert every queue unit at
  aes1 (=aes[0]); engine semantics: a unit at depth d = aes[d] BTC (volatile aes[1]=0.378
  vs aes1=0.553 → 47% over-credit at depth 2). FIX: walk depths accumulating BTC with
  aes[d] per slot; target in BTC; stop when target reached (last unit may overshoot ≤1
  unit — unavoidable at unit resolution, keep + document).
- **I4 (env bug — flow feature):** reactive_env.py _run_interval: engine side coding is
  1=bid, 2=ask (verified); code tested sides==1 as "buy" and sides==0 (never occurs) as
  sell → feature = sell-print count mislabeled as buys. FIX:
  signed = (sides==2).sum() − (sides==1).sum() (ask-side market order = buy aggressor).
- **I5 (env bug — empty-side readout):** _mid_from_state/_spread_ticks use argmax on a
  possibly all-zero side → stale mid/1-tick spread reported off an empty book (reachable:
  agent sweeps ~11-13 BTC visible ask). FIX: if a side is empty, price it at the window
  edge (slot K) for mid/spread purposes and flag; document.
- **I6 (env/design bug — discarded endogenous ref moves):** _run_interval and
  exo_ref_sim.run_exo_qrm discard the engine's evolved p_ref (keep state only) → frame
  slip; agent permanent impact structurally zero; the module docstring's justification is
  stale (queues DO empty post-quiet-spell-fix; one reviewer measured 421 update_LOB
  firings/600s on calm_b, the other ~0.4% of intervals on volatile — reconcile during fix).
  FIX: capture p_ref from the engine (p_refs[-1]) and reconcile with the exogenous
  process: since exo moves were measured from ALL real mid changes (endo+exo combined),
  adding engine-generated moves on top double-counts background motion. RESOLUTION (with
  I1): set theta=0.0 for the BACKGROUND channel is wrong (theta gates ref-change-on-
  mid-move; mid moves when a queue empties)... implement instead: capture p_refs[-1] as
  truth (no discard), and REMOVE the endogenous component from the exogenous table by
  centring (I1) + re-measuring the exo distribution as (all real moves) — as-is, plus
  document that background endo+exo may double-count ~the endogenous rate; QUANTIFY
  endogenous move rate in the fixed env and, if it exceeds 5% of total move variance,
  subtract its measured distribution from the exo table (deconvolution by rate
  subtraction on the ±1 bins). Gates re-run will show if M1 overshoots.
- **I7 (env bug — rint banker's rounding):** exo_ref_sim.move_process_from_mids and
  step3g._move_process_from_deltas use np.rint → ±1.5→±2 always (re-form branch), never
  ±1 (shift branch); ±0.5→0. FIX: explicit rule: |delta| rounds half-AWAY-from-zero
  (np.sign(d)*np.floor(np.abs(d)+0.5)) EXCEPT ±0.5→0 kept deliberately (mid-flicker
  belongs to endogenous dynamics) — i.e. threshold at |d|>=0.75 for 1 tick; document.
- **I8 (training bug — DQN exploration):** train_reactive.py chunked model.learn(...,
  reset_num_timesteps=False) per 100k makes SB3 recompute the epsilon anneal against each
  chunk → eps hit floor within the first chunk in ALL 22 DQN runs (verified with minimal
  repro); D1 (the pre-registered exploration test) never ran as designed. FIX: single
  model.learn(total_timesteps=args.steps) with an SB3 callback (EveryNTimesteps →
  eval_paired_vs_adaptive) for the curve; verify eps trace in a smoke run.
- **I9 (judgement bugs):** (a) _v1aext tag forms an unjudgeable 2-run group → normalize
  tags (map _v1aext → _v1a) before grouping so the 5-seed cell judges under the frozen §3
  conditions; (b) audit action-share cap (>90% one action = INVALID) would invalidate
  always-2.0x while TWAP baselines are themselves single-action — REVISED RULE (log in
  criteria §4): the action-share cap applies only when the residual rule ALSO fires OR
  when the constant action is 0.0x (do-nothing collapse); a constant non-zero-pace policy
  is a legitimate schedule, judged on cost like any baseline; (c) executed_frac vacuous →
  report 1 − deadline_residual/order using the FIXED residual.
- **I10 (hygiene):** reactive_env.reset reseeds global np.random each episode and SB3 DQN
  epsilon coin-flips read that global stream → deterministic entanglement (no bias found;
  fix anyway): draw exo path + env randomness from a local Generator; leave global seeding
  OUT of the env except numba (needed by vendored njit) — numba seeding stays.
- **Known/accepted (no code change):** C5/C6 quantified further (market events ~5x rarer,
  ~7x chunkier at matched volume; M5a partly circular); beyond-window deadline pricing at
  deepest-visible is GENEROUS (favours dump policies) — counteracted by I2 fix +
  documented; G3 ordering checks are drift-free by design (document that this masks
  drift-driven orderings; the drift itself is removed by I1).

## EXECUTION SEQUENCE (update STATUS as you go)

- [x] **R0.** Kill obsolete training runs (done 2026-07-07).
- [x] **R1.** (DONE 2026-07-07: all fixes in, 219 tests pass; deadline-test premise updated for captured endo moves) Env fixes I2, I3, I4, I5, I10 in reactive_env.py; I6 capture-p_ref in
  reactive_env + exo_ref_sim; I7 rounding rule in both move-process builders. Unit tests
  for each (deadline exact-completion; per-depth BTC accounting; flow sign; empty-side
  readout; carry cap; p_ref continuity). Full pytest suite green.
- [x] **R2.** (DONE: centered via exponential tilt — means ~1e-18, std +0.1%; consumers repointed) I1 centering: new step3g subcommand `center-move-process` → writes
  move_process_{regime}_centered.npz (probs adjusted to zero mean by shifting mass
  between ±k bins minimally OR subtract mean via probability tilt — implement as:
  measured distribution, then p'[k] = p[k] adjusted by moving the minimal mass from
  positive bins to negative (documented method); verify mean ≈ 0, variance change < 2%).
  Env/gates/training point at the centered files.
- [x] **R3.** (DONE: endo share calm 11.5% -> deconvolved (+re-centered, robust tilt solver after a caught blow-up); volatile 3.7% documented; total move variance within 4% of real, drift ~0 within noise; fidelity gates = same accepted C5 picture, no regression; G1'/G2'/G3 ALL PASS after fixing the gates' own stale flat-aes1 normalization + G3 tolerance 0.015 with documented overshoot-tax mechanism) Quantify endogenous move rate in the fixed env (I6); if > 5% of move
  variance, deconvolve exo table; re-run per-regime fidelity gates (step3g gate-regime)
  on the corrected env stack + centered process; record vs frozen bands (criteria §3b
  Revision-1 form). Also re-run G1'/G2'/G3 (step4_gates with corrected env).
- [x] **R4.** (DONE: single learn() + CurveCallback; epsilon verified annealing correctly — 0.057 @100k of a 300k budget) Trainer fix I8 (single learn + callback curve): verify epsilon trace on a
  100k smoke run (expect eps ~1.0 early, floor only at ~25% of budget + warmup).
- [x] **R5.** (DONE: v1aext tag merge; audit share-cap revised per criteria 4b; executed_frac = voluntary completion) Judgement fixes I9 (tag normalization; audit rule revision; executed_frac);
  log the audit-rule revision in reports/qrm_step4_criteria.md §4 with rationale BEFORE
  re-judging.
- [x] **R6.** (DONE 2026-07-07: 20 runs to runs_primary_v2, no errors) RE-TRAIN PRIMARY:
  DQN + PPO x {calm, volatile} x 5 seeds, 25 BTC, 2M steps, corrected env + centered process.
- [x] **R7.** (DONE 2026-07-08) Sealed judgement on runs_primary_v2 → $S/step5_v2/judgement.json.
  **VERDICT = FORMAL NULL in all 4 cells (no cell met the frozen §3 edge conditions), BUT the
  corrected picture INVERTED vs the buggy runs:**
  - **PPO calm: all 5 seeds beat adaptive-TWAP** (−0.006 to −0.037 bps; across-seed t p=0.025;
    pooled −0.019). Individually significant seeds: 0/5. → sub-threshold favourable.
  - **PPO volatile: 4/5 beat it, 3 seeds individually clear the −0.05 materiality bar**
    (−0.059, −0.067, −0.067; across-seed t p=0.066; pooled −0.039). → near-miss.
  - **PPO overall: 9/10 seeds on the better side (sign-test p≈0.02).** The bug fixes removed a
    handicap that had been PENALISING paced policies; PPO flipped from losing to consistently
    (but sub-threshold) winning.
  - **DQN: NULL, clean.** With exploration genuinely working now, 8/10 seeds STILL collapse to
    do-nothing-then-dump (audit INVALID), the 2 valid lose (+0.09..+0.12). The collapse is now
    an un-confounded finding about the algorithm, not a wiring artifact.
  - **Sensitivity:** counting invalid seeds changes nothing (no seed negative-significant in DQN).
  - **Interpretation:** the pre-registered bar was NOT cleared → the honest headline is a
    BOUNDARY NULL. The consistent sub-threshold PPO signal is real-looking but was noticed IN
    this data, so it cannot be claimed without out-of-sample replication.
- [x] **R8. (CLOSED 2026-07-10: R8a ran + selection complete — winner ppo_volatile_v3a; R8b = the next step. See CURRENT RESULTS (A).)** Two tracks, kept distinct:
  - **(R8a) Pre-registered tuning table (§5), on the corrected env** — the FULL table (nets
    64/128, reward×100, lr, entropy, rollout; DQN D1/D2), 3-seed screening + escalation.
    Legitimate + always-planned: answers "could a better-tuned agent clear the bar?" Not
    "re-roll until it wins." DQN variants run for fairness even though DQN is far from the line.
  - **(R8b) Out-of-sample replication of the PPO signal** — new seeds, NEW reserved 2,000-episode
    block, frozen rules. ONLY needed IF we want to upgrade PPO from "suggestive boundary" to
    "claimed edge." Contentious if framed as chasing significance; clean if framed as
    replication-before-claim. Default if unsure: DO NOT run; report the boundary null.
- [~] **R9 (partial, 2026-07-09).** BUILD_PLAN top block + HANDOVER (14) CURRENT-STATUS block updated
  with the remediation + drift-fix story; root HANDOVER superseded-stamped; criteria §5 stale verdict
  stamped; this file's STATUS refreshed. REMAINING: fold the FINAL `step5_v3` numbers in once the
  clean campaign is judged; sync web copies.

> **SUPERSEDED (drift-confounded).** This v2 campaign predates the drift fix; its numbers must
> never be cited. The folders were RENAMED 2026-07-10: `step5_v2` -> `SUPERSEDED_step5_v2`,
> `runs_primary_v2` -> `SUPERSEDED_runs_primary_v2` (paths below updated to match). The current
> primary result is CURRENT RESULTS (B) above.

## FULL PER-SEED RESULTS TABLE — runs_primary_v2 (sealed judgement, 2026-07-08)

**Source of record (EXACT ABSOLUTE PATHS):**
- Costs + p-values, all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/SUPERSEDED_step5_v2/judgement.json`
- Behaviour audit (valid/invalid), all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/SUPERSEDED_step5_v2/behaviour_audit.json`
- Trained agents + logs + learning curves:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/SUPERSEDED_runs_primary_v2/`

This table is the human-readable transcription of those two JSON files; the JSON is
authoritative if they ever disagree.

**Sign convention:** `vs adaptive` / `vs fixed` = agent mean implementation shortfall MINUS
baseline, in bps, over 2,000 CRN-paired episodes (eval seed0 = 5,000,000). NEGATIVE = agent
CHEAPER (beat TWAP). POSITIVE = agent DEARER (lost). `p(adaptive)` = Wilcoxon signed-rank
p vs adaptive-TWAP (the frozen edge needs p < 0.01). `valid` = passed the Step-4.6 behaviour
audit (not a do-nothing-then-dump collapse).

| run | vs adaptive (bps) | vs fixed (bps) | p(adaptive) | valid |
|---|---|---|---|---|
| ppo_calm_s0     | -0.0094 | -0.0102 | 0.873 | yes |
| ppo_calm_s1     | -0.0064 | -0.0072 | 0.992 | yes |
| ppo_calm_s2     | -0.0372 | -0.0380 | 0.055 | yes |
| ppo_calm_s3     | -0.0244 | -0.0252 | 0.196 | yes |
| ppo_calm_s4     | -0.0194 | -0.0202 | 0.351 | yes |
| ppo_volatile_s0 | -0.0666 | -0.0660 | 0.041 | yes |
| ppo_volatile_s1 | -0.0670 | -0.0663 | 0.075 | yes |
| ppo_volatile_s2 | -0.0593 | -0.0587 | 0.123 | yes |
| ppo_volatile_s3 | -0.0026 | -0.0019 | 0.988 | yes |
| ppo_volatile_s4 | +0.0005 | +0.0011 | 0.954 | yes |
| dqn_calm_s0     | +0.3899 | +0.3891 | ~3e-24 | NO (0.0x share 76.6%) |
| dqn_calm_s1     | +0.1214 | +0.1206 | ~6e-07 | NO |
| dqn_calm_s2     | +0.5690 | +0.5682 | ~1e-26 | NO (0.0x share 90.6%) |
| dqn_calm_s3     | +0.1482 | +0.1474 | ~9e-09 | NO |
| dqn_calm_s4     | +0.2006 | +0.1998 | ~3e-12 | NO |
| dqn_volatile_s0 | +0.4703 | +0.4710 | ~5e-06 | NO |
| dqn_volatile_s1 | +0.1195 | +0.1201 | 0.050 | YES |
| dqn_volatile_s2 | +0.0873 | +0.0880 | 0.142 | YES |
| dqn_volatile_s3 | +0.1918 | +0.1925 | 0.008 | NO |
| dqn_volatile_s4 | +0.0341 | +0.0348 | 0.397 | NO |

**Cell verdicts (frozen §3 — EDGE needs all four: mean<0 vs both; p<0.01 vs both; sign in
>=4/5 seeds; pooled-vs-adaptive <= -0.05):**

| cell | valid seeds | seeds cheaper | seeds p<0.01 | pooled vs adaptive | EDGE |
|---|---|---|---|---|---|
| ppo_calm     | 5/5 | 5/5 | 0/5 | -0.0194 | NO |
| ppo_volatile | 5/5 | 4/5 | 0/5 | -0.0390 | NO |
| dqn_calm     | 0/5 | 0/5 | 0/5 (all sig WORSE) | n/a (no valid) | NO |
| dqn_volatile | 2/5 | 0/5 | 0/5 | +0.1034 (valid only) | NO |

**Which frozen rules PPO passed / failed (the exact answer):** PPO cleared Rule 1 (direction:
calm 5/5, volatile 4/5 cheaper) and Rule 3 (sign in >=4/5). PPO FAILED Rule 2 (p<0.01 — 0/10
seeds; best single seed p=0.041, a WIDE miss) AND Rule 4 (pooled <= -0.05 — calm -0.019,
volatile -0.039, a NARROW miss on volatile). So PPO missed TWO rules, not one. DQN: clean
loss — 8/10 seeds collapse to do-nothing-then-dump (INVALID); the 2 valid seeds both lose.

**Headline:** BOUNDARY NULL in all four cells. Real-looking but sub-threshold PPO tilt
(9/10 seeds on the cheaper side, sign-test p~0.02) — noticed IN this data, so unclaimable
without out-of-sample confirmation. See R8 below.

### IMPROVEMENTS OVER THE BENCHMARK — the positive evidence, stated plainly (logged for balance; NOT downplayed)

After the bug fixes, PPO's result INVERTED from the buggy runs (where it appeared to lose)
to consistently favourable. The full positive evidence, so it is on the record next to the
null label:
- **PPO was cheaper than the self-correcting adaptive-TWAP in 9 of its 10 seed-runs.** A sign
  test on 9/10 gives p ~ 0.02 — that consistency alone is unlikely to be chance at the
  ordinary 1-in-20 bar.
- **Calm:** all 5 seeds cheaper (−0.006 to −0.037 bps); the ACROSS-SEED test is significant at
  the ordinary 0.05 bar (t p = 0.025); pooled −0.019 bps.
- **Volatile:** 4 of 5 cheaper; 3 seeds individually cheaper by 0.059-0.067 bps; pooled
  −0.039 bps; across-seed t p = 0.066 (just misses 0.05).
- Best single seeds: volatile s0/s1 cheaper by ~0.067 bps at ~1-in-25 odds.

**What it does NOT (yet) meet** — the pre-registered §3 edge bar: p<0.01 per seed in >=4/5
seeds AND pooled <= -0.05. It misses the per-seed significance WIDELY and the materiality floor
NARROWLY (volatile -0.039 vs -0.05). Even at the lenient 0.05 per-seed bar, only 0-1 of 5 seeds
qualify per cell (needs 4) — so the effect is genuinely too weak at the single-run level, not
merely blocked by a strict threshold.

**Honest label:** a GENUINE, SMALL, CONSISTENT favourable signal — NOT yet a claimable edge.
The §6 out-of-sample confirmation is exactly what decides which of the two it is. Both the
positive evidence AND the unmet bar are reported; neither is buried, neither is oversold.

> **STAGE RECORD (sealed 2026-07-09).** The 60-run 3-seed screen. Its per-run numbers are
> reproduced identically inside `step5_selection_v3` (determinism-checked). Its 3-seed
> "SELECTION: v3b" line reflects 3-seed evidence only and is SUPERSEDED by the 5-seed selection
> (winner v3a) in CURRENT RESULTS (A). The PENDING list at the end was executed 2026-07-10.

## R8a TUNING SCREEN — FULL RESULTS (sealed judgement 2026-07-09, `step5_tuning_v3`)

**Source of record:** `$S/step5_tuning_v3/{judgement,behaviour_audit}.json` (runs_tuning_v3). 10
pre-registered §5 variants x 3 seeds x 2 regimes = 60 runs, 0 training failures, judged on the SAME
5e6 dev block as v3, drift-fixed env. Variant key: v1a net[64,64]; v1b net[128,128]; v2 reward x100;
v3a lr 1e-3; v3b lr 1e-4; v4a ent 0.0; v4b ent 0.05; v5 n_steps 8192; d1 DQN eps_final 0.05+anneal
0.5; d2 DQN reward x100+net[64,64]. `pooled` and `across-p` are computed on VALID seeds only; the
per-seed column lists all 3 seeds (bps vs adaptive, negative = agent cheaper).

**THE HEADLINE: the strict per-seed escalation trigger fires for ZERO variants — yet the across-seed
view shows PPO-volatile's edge is TUNING-ROBUST and IMPROVABLE.** If we had relied only on the frozen
per-seed trigger (needs >=2 of 3 seeds individually significant at p<0.01), NOTHING would escalate and
we would have wrongly declared the null "tuning-robust." The across-seed consistency view (pre-registered
§5 rule 3: pooled <= -0.02 AND across-seed p < 0.05) tells the opposite, correct story. This is the exact
mislabelling risk the across-seed safeguard exists to prevent, now demonstrated on real data.

**PPO VOLATILE — 7 of 8 variants negative-consistent; 6 clear across-seed p<0.05; best NEARLY DOUBLES the base edge:**

| variant | valid | seeds vs adaptive (bps) | pooled | across-p | note |
|---|---|---|---|---|---|
| v3b (lr 1e-4)   | 3/3 | -0.072 -0.103 -0.077 | **-0.084** | **0.006** | BEST — ~2x base (-0.047) |
| v3a (lr 1e-3)   | 3/3 | -0.050 -0.105 -0.035 | -0.063 | 0.048 | |
| v2 (reward x100)| 3/3 | -0.051 -0.062 -0.053 | -0.055 | 0.002 | pooled clears -0.05 floor; tightest spread |
| v4a (ent 0.0)   | 3/3 | -0.051 -0.077 -0.014 | -0.047 | 0.062 | |
| v1b (net128)    | 3/3 | -0.016 -0.044 -0.065 | -0.042 | 0.050 | |
| v4b (ent 0.05)  | 3/3 | -0.041 -0.039 -0.024 | -0.034 | 0.012 | |
| v5 (n_steps8192)| 3/3 | -0.051 -0.022 -0.027 | -0.033 | 0.034 | |
| v1a (net64)     | 2/3 | -0.010 -0.030 (+0.023 invalid) | -0.020 | 0.151 | weakest; 1 seed invalid |

**PPO CALM — mostly null, a couple of tiny consistent effects (consistent with v3: little/no calm signal):**

| variant | valid | pooled | across-p | note |
|---|---|---|---|---|
| v3b | 2/3 | -0.024 | 0.111 | |
| v1b | 3/3 | -0.021 | 0.140 | best calm pooled but n.s. |
| v4a | 3/3 | -0.020 | 0.028 | tiny but consistent |
| v1a | 3/3 | -0.016 | 0.040 | tiny but consistent |
| v5  | 3/3 | -0.016 | 0.141 | |
| v4b | 3/3 | -0.006 | 0.200 | |
| v2  | 3/3 | -0.002 | 0.322 | reward-scale did nothing in calm |
| v3a | 3/3 | +0.001 | 0.528 | |

**DQN — collapse persists in most collapse-fix variants; no significant edge anywhere:**
d1 calm 1/3 valid (pooled -0.007); d2 calm 0/3 valid (TOTAL collapse); d1 volatile 1/3 valid
(pooled -0.040 on 1 seed); d2 volatile 3/3 valid (pooled -0.026, across-p 0.199, n.s.). The
collapse-fix arms did not reliably fix the L2 do-nothing-then-dump pathology; d2 volatile survived but
shows no real edge. DQN remains the weak algorithm.

**SELECTION (§5 mechanical rule — best pooled cost vs adaptive among fully-healthy configs):**
- volatile: **ppo_volatile_v3b (lr 1e-4), pooled -0.084, across-seed p=0.006, 3/3 valid** — clear winner.
- calm: ppo_calm_v1b, pooled -0.021 (n.s.) — no real calm signal to carry.

**INTERPRETATION:** the tuning did its intended job in the STRONGEST possible way — it did not rescue a
null, it showed the volatile edge REPLICATES across nearly every one-factor hyperparameter change and can
be roughly DOUBLED by a smaller learning rate (v3b). This decisively closes the "you just trained it
badly" attack — in the opposite direction (the edge is robust, not fragile). Calm stays null; DQN stays
collapse-prone.

**PENDING per pre-registration BEFORE selecting the final confirmation target:** (1) V6 = 10M steps on the
best Wave-1 variant; (2) Wave-3 combinations (best-of-v1a/v1b x v2; and the two best-pooled variants
combined) — note v3b (lr) and v2 (reward) are the two strongest single factors and are mechanistically
complementary; (3) escalate the top consistent-edge volatile variants (v3b, v2, v3a) to the full 5 seeds.
Then select the single best healthy config and run R8b out-of-sample confirmation on the sealed 9e6 block.
NOTE: the strict ESCALATE flag in `judgement.json` reads False for every variant (per-seed trigger);
escalation candidates here were identified by the pre-registered §5-rule-3 across-seed criterion, applied
by hand (the flag is not wired to across-seed — verified faithful, kept informational per user).

> **CLOSED (2026-07-10).** Historical decision rationale; the recommended sequence (R8a full
> table -> select -> R8b confirm) was executed through selection. Kept for the record.

## R8 DECISION — REFINED (2026-07-08) — CLOSED 2026-07-10 (original header said "still OPEN pending go-ahead")

Two corrections to earlier framing in this file / chat, logged for honesty:

1. **The pre-registered near-miss escalation does NOT literally fire.** §5 escalation needs
   BOTH pooled <= -0.02 bps AND across-seed p < 0.05. Each cell satisfies exactly one and
   misses the other by a hair:
   - calm: pooled -0.0194 (MISSES -0.02 by 0.0006), across-seed t p=0.025 (clears 0.05).
   - volatile: pooled -0.0390 (clears -0.02), across-seed t p=0.066 (MISSES 0.05).
   Therefore running R8 is a documented JUDGMENT, not an automatic trigger. (Earlier wording
   "the machinery says to do this" overstated it — corrected here.)
2. **Tuning scope = the FULL §5 table, not a hand-picked single variant.** An earlier chat
   recommendation ("just the 64x64 variant at full 5-seed") was a quiet cherry-pick chosen
   after seeing the primary result. The disciplined, pre-committed move is the whole §5 table
   under its own screen-at-3-seeds-then-escalate rule, DQN variants included for fairness.

**Objective recommendation (the 90+ / highest-standard path), in this order:**
- **R8a — run the full pre-registered §5 tuning table** on the corrected env (screen 3 seeds,
  escalate promising ones per §5). Answers the examiner's question "was the null just an
  under-tuned agent?" Pre-committed, so it is disciplined exploration, not fishing.
- **R8b — take the single strongest candidate that emerges and confirm it out-of-sample:**
  brand-new seeds, a fresh DISJOINT 2,000-episode block never touched, a binary pass/fail rule
  written into criteria §6 BEFORE launch. Fresh data + pre-committed binary can only kill or
  survive the tilt; it cannot manufacture a false win. This is the ONLY move that upgrades
  "suggestive" to "claimed."
- **Sequence:** tune FIRST (may yield a stronger candidate than the primary PPO), THEN confirm
  the best candidate on fresh data. Do NOT skip tuning and confirm the primary PPO directly —
  that leaves the under-tuning question open and takes the weaker candidate into the test.
- **Headline stays BOUNDARY NULL** unless and until R8b passes. If it passes: a real, small,
  confirmed reactive-market execution edge, and RQ3 attribution switches on with a genuine
  effect to explain. If it fails: the null is boundary-documented and bulletproof.

**STATUS (updated 2026-07-09):** The R8b confirmation protocol IS written (criteria §6, done
2026-07-08). SUPERSEDED SINCE: a pre-launch audit (2026-07-08) found the corrected env STILL carried
a residual upward drift that confounded R7; the drift was neutralised, a permanent fairness gate
added, and the pre-training gates re-run (all pass). The primary campaign is being RE-RUN on the
drift-fixed env (`runs_primary_v3`). The R7 boundary-null above is now PROVISIONAL and will be
replaced by the `step5_v3` verdict. Full drift-fix record: `reports/qrm_prelaunch_audit_2026-07-08.md`.
R8a/R8b run AFTER the clean v3 verdict.

