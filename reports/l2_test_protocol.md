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
`curve.csv` (validation history). Sizes in BTC: 96.57 (~1% of average daily volume),
193.13 (~2%), 386.27 (~4%).

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
