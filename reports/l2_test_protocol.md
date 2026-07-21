# L2 TRACK — SEALED TEST-SET EVALUATION PROTOCOL (pre-registered 2026-07-14; NOT YET RUN)

Written BEFORE the evaluation executes, per the project's standing rule (rules first, results
after). Execution slot per the agreed order: AFTER the 66-run QRM grid is trained and judged,
and on explicit user go-ahead. ONE-SHOT: run once, no re-runs, full per-run table logged to
the live doc at the boundary regardless of outcome.

## 1. Why this exists

Every L2 number produced so far comes from the VALIDATION split, which was looked at
repeatedly during development (config comparisons, training monitoring). Those numbers carry
selection bias and cannot be the track's final verdict. The TEST split has never been touched:
verified 2026-07-14 — `dataset/test.parquet` (48M), `dataset_10s/test.parquet` (241M),
`dataset_10s_10min/test.parquet` (243M) exist; zero test-evaluation artifacts exist anywhere
under `runs/`, `runs_10s/`, `runs_10s_10min/`.

## 2. Exactly what has been trained (inventory verified on disk, 2026-07-14)

51 trained agents total, each folder holding `model.zip` + `normalizer.json` + `meta.json` +
`curve.csv` (validation history). Sizes in BTC: 96.57 (0.5% of average daily volume; ADV
19,313 BTC recorded in every meta.json), 193.13 (1%), 386.27 (2%). [CORRECTED 2026-07-17:
this line originally said ~1%/~2%/~4%; the sweep file and recorded ADV give 0.5%/1%/2%.]

| Dataset / cell | PPO 96.57 | PPO 193.13 | PPO 386.27 | DQN 96.57 | DQN 193.13 | DQN 386.27 |
|---|---|---|---|---|---|---|
| 1-minute data, 30-min deadline (`runs/`) | **0** | 5 seeds | **0** | **1 seed** | 5 seeds | **0** |
| 10-second data, 30-min deadline (`runs_10s/`) | 5 seeds | 5 seeds | 5 seeds | 5 seeds | **0** | **0** |
| 10-second data, 10-min deadline (`runs_10s_10min/`) | 5 seeds | 5 seeds | **0** | 5 seeds | 5 seeds | **0** |
| 1-minute data, 10-min deadline | — dataset never built; combination never run — | | | | | |

## 3. Missing coverage + FILL-IN TRAINING PLAN (user decision 2026-07-14: fill the holes,
##    do NOT present a half-complete table)

Holes as of 2026-07-14:
- PPO at the primary size (96.57) was never trained on 1-minute data; DQN at 96.57 on
  1-minute data has only 1 seed (not 5).
- DQN was never trained at 193.13 or 386.27 on 10-second/30-min data (only PPO got the
  size ladder there).
- The 1-minute/10-minute-deadline combination was never built. This one stays OUT by
  design: the track varies one lever at a time from the base cell, and that diagonal was
  never part of any comparison. Not a table hole.

**EXECUTION STATUS (2026-07-17): ALL 19 FILL-INS TRAINED.** Every panel is now complete
(70 agents total). Incident logged for the audit trail: the fill-in launcher's job-spec
word-splitting broke on the space in the "MSc Dissertation" path, so 18 runs trained with
CORRECT configs (verified field-by-field against sibling metas before recovery) but wrote to
a stray `~/Desktop/MSc/` folder; all 18 were verified complete + config-identical, then moved
to their proper `runs/`/`runs_10s/` locations (no destination collisions; stray folder
removed). Validation snapshot (vs TWAP, bps, group means): ppo96.57@1-min -0.089 (4/5 seeds
negative); dqn96.57@1-min +0.592 (seed 4 COLLAPSED: 97.8% deadline-leaning, +3.16 bps;
others mixed); dqn193.13@10-s +0.142; dqn386.27@10-s +0.205. Full 70-agent validation table =
step (b) gate deliverable. Sealed test still NOT run.

**STUB DISCOVERY + CORRECTION (2026-07-17, full-census verified).** A per-run census of all 70
nominal agents (model.zip + normalizer.json + meta.json) found 66/70 complete. The exception:
`runs/ppo_size193.13_seed{1..4}` were STUBS — curve.csv only, trained to 200,000 steps (10% of
budget), NO saved model — so they can never sit the sealed exam, and the historical
"ppo193@1-min: 5 seeds, mean -0.090" figure was stub-based and MUST NOT be cited (only seed 0
was a full run). Action (user-approved): stubs quarantined to
`scratch_hyperliquid/SUPERSEDED_stubs_ppo193_1min/` (never cited); seeds 1-4 RETRAINED properly
with the sibling config (2M steps; COMPLETED 2026-07-17, all 4 verified config-identical to
seed 0 with complete artifacts; values in the validation table at the end of this file). The §2 inventory's "51 agents each
holding model+normalizer+meta" claim was sample-checked, not census-checked, when written —
corrected by this census. ESCALATION NOTE (user question, answered for the record): promising
validation groups (e.g. ppo96.57@1-min, mean -0.089, 4/5 cheaper) are NOT escalated on this
track — the protocol has no selection stage; ALL 70 agents sit the same one-shot sealed exam,
which is every group's confirmation instrument (multiplicity over 14 arms disclosed). Pre-launch verifications (both PASSED, logged
here as the protocol's preconditions): (1) COMPARABILITY — the 1-min siblings pre-date the
2026-06-30 pipeline commit, so one sibling (dqn_size193.13_seed0, 1-min) was RETRAINED with
current code into a scratch dir: bit-exact reproduction (every config field identical, final
validation number identical to 17 d.p., all 21 curve points zero-diff). (2) RAM — measured
peaks: 1-min job 696 MB, 10-s job 441 MB; even 8-wide is ~5.6 GB worst case, so RAM is not
binding and the CPU cap (8) governs.

FILL-IN RUNS (19 total, planned; execute when the L2 step is reached, BEFORE the sealed
test so every agent sits the same one-shot exam):
| fill-in | runs |
|---|---|
| ppo_size96.57, 1-min data, 30-min deadline (`runs/`) | seeds 0-4 (5) |
| dqn_size96.57, 1-min data, 30-min deadline (`runs/`) | seeds 1-4 (4; seed 0 exists) |
| dqn_size193.13, 10-s data, 30-min deadline (`runs_10s/`) | seeds 0-4 (5) |
| dqn_size386.27, 10-s data, 30-min deadline (`runs_10s/`) | seeds 0-4 (5) |

Same configs/hyperparameters as the existing runs in each folder (read them from the
sibling runs' meta.json — no new variant is being introduced). After fill-ins: every
presented panel is complete — resolution panel 2 algos x 2 sizes, size-ladder panel
2 algos x 3 sizes, deadline panel 2 algos x 2 sizes.

## 4. The protocol (fixed in advance)

0. **Order within the L2 step: train the 19 fill-ins first, validate them, THEN run this
   sealed test once over ALL agents (51 existing + 19 fill-ins = 70).** The test is a
   single exam sat by everyone together; it does not run before the fill-ins are trained.
1. **Scope: ALL trained agents (70 after fill-ins). No selection.** Every run is evaluated on the test split
   of ITS OWN dataset, with the same episode construction, features, and order size as its
   validation, deterministic policy (no exploration), and the run's saved normaliser.
2. **Paired comparison:** agent vs the TWAP benchmark on identical test episodes; per-episode
   cost difference in basis points (negative = agent cheaper), same convention as validation.
3. **Recorded per run:** mean paired difference, per-episode count, standard deviation,
   Wilcoxon p. **Per arm (dataset x algo x size):** pooled mean over seeds, across-seed
   one-sided t p, number of seeds cheaper than TWAP.
4. **Multiplicity, stated up front:** 14 arms exist; all raw p-values are reported for every
   arm with the number of arms stated. No arm is cherry-picked; a single nominally significant
   arm among 14 is expected under the null and will be labelled as such.
5. **Expectation registered in advance:** validation showed ties/nulls and one DQN collapse
   arm. The test evaluation is confirmatory either way; whatever it shows is the chapter's
   final verdict.
6. **One-shot:** no re-runs, no second look, no post-hoc metric changes. Full per-run table
   goes into the live doc and into table T9/T7 (test columns) at the run boundary.

## 5. RAM safety (mandatory, before any evaluation starts)

L2 data loading has frozen this machine before. Procedure: run ONE agent's evaluation first
(smallest file: `dataset/test.parquet`, 48M), measure peak memory; then one on a 241M
10-second dataset, measure again. Only then decide sequential vs limited-parallel for the
remaining 49. Default: strictly sequential, one dataset loaded at a time, freed before the
next. Never load more than one test parquet simultaneously.

## 6. Outputs

- Full per-run test table appended to the live doc (all 51 rows).
- Table T9 (per-run detail: validation + test columns, clearly labelled) and T7 (track
  summary) updated; figures L1 (three-axis null, actual cells only) and L2 (size lever)
  gain their test-set counterparts.

## VALIDATION TABLE OF RECORD — all 70 agents (assembled 2026-07-17, step (b) gate)

All numbers = final validation result from each agent's own meta.json (vs TWAP, bps;
negative = agent cheaper). DL = leaned on the forced deadline buy in >10% of episodes.

| panel | algo | size | s0 | s1 | s2 | s3 | s4 | mean | cheaper |
|---|---|---|---|---|---|---|---|---|---|
| 1-minute data, 30-min deadline | PPO | 96.57 | -0.171 | -0.083 | -0.117 | +0.049 | -0.123 | -0.0890 | 4/5 |
| 1-minute data, 30-min deadline | PPO | 193.13 | -0.012 | +0.009 | +0.031 | -0.056 | -0.121 | -0.0297 | 3/5 |
| 1-minute data, 30-min deadline | DQN | 96.57 | -0.076 | +0.006 | -0.051 | -0.079 | +3.161 DL | +0.5920 | 3/5 |
| 1-minute data, 30-min deadline | DQN | 193.13 | -0.043 | +0.310 DL | -0.055 | -0.022 | +0.063 DL | +0.0505 | 3/5 |
| 10-second data, 30-min deadline | PPO | 96.57 | +0.206 | +0.159 | +0.185 | +0.192 | +0.149 | +0.1779 | 0/5 |
| 10-second data, 30-min deadline | PPO | 193.13 | +0.096 | +0.123 | +0.250 | +0.155 | -0.253 | +0.0740 | 1/5 |
| 10-second data, 30-min deadline | PPO | 386.27 | +0.229 | +0.174 | +0.156 | +0.217 | +0.227 | +0.2010 | 0/5 |
| 10-second data, 30-min deadline | DQN | 96.57 | +2.943 DL | +0.173 | +0.233 | +0.234 | +0.249 | +0.7663 | 0/5 |
| 10-second data, 30-min deadline | DQN | 193.13 | +0.244 | -0.102 | +0.057 | +0.254 | +0.256 | +0.1417 | 1/5 |
| 10-second data, 30-min deadline | DQN | 386.27 | -0.057 DL | +0.308 | +0.221 | +0.298 | +0.254 | +0.2049 | 1/5 |
| 10-second data, 10-min deadline | PPO | 96.57 | -0.026 | -0.095 | +0.137 | +0.045 | -0.028 | +0.0066 | 3/5 |
| 10-second data, 10-min deadline | PPO | 193.13 | -0.101 | +0.074 | +0.099 | +0.037 | +0.053 | +0.0324 | 1/5 |
| 10-second data, 10-min deadline | DQN | 96.57 | +0.207 DL | -0.026 | -0.028 | +0.055 | -0.086 | +0.0244 | 3/5 |
| 10-second data, 10-min deadline | DQN | 193.13 | +0.441 DL | -0.031 | +0.077 | -0.040 | +0.119 | +0.1133 | 2/5 |

TOTAL: 70/70 complete agents. The 4 ppo193@1-min retrains (replacing the quarantined stubs) came in at +0.009/+0.031/-0.056/-0.121 — NOTE: much weaker than the retracted 200k-step stub values (-0.086..-0.170), confirming the retraction mattered. This table is the validation record; the sealed exam (step d) remains NOT RUN, gated on the user.
