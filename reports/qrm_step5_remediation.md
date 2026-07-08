# Step-5 REMEDIATION PLAN (2026-07-07, user-approved) — self-contained execution record

**Context for a fresh session:** a three-agent adversarial code review (2026-07-07) of the
reactive-QRM experiment found confirmed bugs + one design confound. ALL Step-5 results to
date (primary 20-run campaign, Wave-1/2/3 tuning runs, their judgements) are RECLASSIFIED
as engineering shakedown, NOT evidence. This file is the authoritative worklist: fix →
re-gate → re-train → re-judge. Update the STATUS lines in place as steps complete.
Paths: repo = /Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution
(run everything with PYTHONPATH=src, venv .venv). Scratch =
/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4 (call it $S).
Old runs/judgements to keep for the record but never cite as results:
$S/runs_reactive, $S/runs_wave1, $S/runs_wave2, $S/step5, $S/step5_wave1.

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
- [ ] **R8. (OPEN — user deciding 2026-07-08).** Two tracks, kept distinct:
  - **(R8a) Pre-registered tuning table (§5), on the corrected env** — the FULL table (nets
    64/128, reward×100, lr, entropy, rollout; DQN D1/D2), 3-seed screening + escalation.
    Legitimate + always-planned: answers "could a better-tuned agent clear the bar?" Not
    "re-roll until it wins." DQN variants run for fairness even though DQN is far from the line.
  - **(R8b) Out-of-sample replication of the PPO signal** — new seeds, NEW reserved 2,000-episode
    block, frozen rules. ONLY needed IF we want to upgrade PPO from "suggestive boundary" to
    "claimed edge." Contentious if framed as chasing significance; clean if framed as
    replication-before-claim. Default if unsure: DO NOT run; report the boundary null.
- [ ] **R9.** Update BUILD_PLAN + HANDOVER with the full remediation story (the review, the
  issues, the reclassification of old results, the R7 verdict). Sync reminder for the web copies.

## INVARIANTS (do not violate while executing)

- The L2 test set stays sealed. Old run dirs are kept, never cited as results.
- Every criteria change is logged in reports/qrm_step4_criteria.md with date + rationale
  BEFORE the affected run executes. Frozen thresholds (§3) do NOT change.
- Audit before costs, always. CRN seed blocks unchanged (train base seed*1e7; curve 1e6;
  judgement 5e6). No new variants beyond the §5 table without a new documented mechanism.

## FULL PER-SEED RESULTS TABLE — runs_primary_v2 (sealed judgement, 2026-07-08)

**Source of record (EXACT ABSOLUTE PATHS):**
- Costs + p-values, all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_v2/judgement.json`
- Behaviour audit (valid/invalid), all 20 runs:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/step5_v2/behaviour_audit.json`
- Trained agents + logs + learning curves:
  `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_primary_v2/`

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

## R8 DECISION — REFINED (2026-07-08, still OPEN pending go-ahead to launch)

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
  written into criteria §5 BEFORE launch. Fresh data + pre-committed binary can only kill or
  survive the tilt; it cannot manufacture a false win. This is the ONLY move that upgrades
  "suggestive" to "claimed."
- **Sequence:** tune FIRST (may yield a stronger candidate than the primary PPO), THEN confirm
  the best candidate on fresh data. Do NOT skip tuning and confirm the primary PPO directly —
  that leaves the under-tuning question open and takes the weaker candidate into the test.
- **Headline stays BOUNDARY NULL** unless and until R8b passes. If it passes: a real, small,
  confirmed reactive-market execution edge, and RQ3 attribution switches on with a genuine
  effect to explain. If it fails: the null is boundary-documented and bulletproof.

**STATUS:** OPEN. Next physical action = write the R8b confirmation protocol into criteria §5,
then launch R8a. Nothing running.
