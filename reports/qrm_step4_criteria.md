# Step 4 — pre-registered criteria (FROZEN 2026-07-06, before any gate or training run)

Companion to BUILD_PLAN "STEP 4 — PLAN LOCKED (user-approved 06/07/2026)". The design
decisions are recorded there; this file freezes the NUMBERS: gate pass bands and the
definition of "beats TWAP". Revisions only with a stated mechanism, logged here (the 3f
discipline). Nothing below was computed from any simulation of the agent environment.

## 1. Locked design (summary; full rationale in BUILD_PLAN)

- Cadence 1 s; horizon 300 s (10-min robustness variant at primary size only, post-hoc).
- Sizes {5, 12.5, 25, 50} BTC (= the L2 {0.1,0.25,0.5,1.0}%-ADV ladder at matched trading
  intensity, Dec-2025 ADV 29,939 BTC/day from the trades file; candle API 30,109 agrees
  to 0.6 %). Primary 25 BTC. Same absolute sizes in both regimes.
- Action = TWAP-pace multiples {0, 0.5, 0.8, 1, 1.2, 1.5, 2} with fractional-unit carry.
- Observation = inventory, time, 20 queue sizes (AES units), spread, own fills (last +
  cumulative), trailing market trade volume + signed buy/sell balance (window 30 s).
- Reward = negative implementation shortfall vs arrival mid, bps. Deadline: force-buy
  through the current sim book; beyond-window remainder at the deepest visible level.
- Bundles: `step3g/qrm_bundle_{calm,volatile}_b.npz` (3g bundles of record).

## 2. Pre-training gates — pass bands

**G1 — reaction lever (matched pairs, identical seeds, with-order vs without).**
Protocol: N = 200 paired episodes per regime; the "with" arm executes the size via
fixed-TWAP; measure (a) mean |mid displacement| at t = 300 s, (b) mean absolute
difference in visible ask-side depth over the episode.
- PASS at the primary size (25 BTC): mean |mid displacement| >= 1 tick ($1) in each
  regime, AND displacement increases monotonically across the size ladder (Spearman
  rho > 0, 4 points). Rationale: a reacting market must move at least one price step
  under ~44 consumed units; monotonicity separates reaction from noise.
- FAIL -> stop; report to the supervisor before any training (the experiment's premise
  is then not satisfied at defensible sizes).

**G2 — effective cost-vs-size (the C5 tripwire).**
Protocol: cost (bps of mid) of executing {1, 2, 5, 10} AES units instantly, in-sim
(mean over 200 sampled sim book states per regime) vs walking real v2 books (mean over
all 0.5 s snapshots of the same regime's CALIBRATION blocks).
- Band (formula, mirrors 3f): for each size, the real per-block dispersion = median
  absolute relative deviation of per-block mean cost from the pooled mean, across the
  5 calibration blocks; PASS iff |sim − real_pooled| / real_pooled <= max(that
  dispersion, 25 %) at every size, per regime. The 25 % floor guards against the
  degenerate case of near-identical blocks making the band unmeetably tight (the 3f
  T_spr lesson, fixed in advance this time).
- FAIL -> stop; the fill mechanics misprice the very quantity the study measures.

**G3 — benchmark sanity (hard binary).**
Fixed-TWAP executes 100 % of the order in >= 99 % of 500 episodes per regime;
instant-dump mean cost >= fixed-TWAP mean cost; mean cost non-decreasing in order size
(4-point ladder, both regimes). Any violation -> wiring bug; fix before proceeding.

## 3. "Beats TWAP" — the pre-committed definition (Step 5 judgement)

- Evaluation: 2,000 CRN-paired episodes per regime; every policy evaluated on the SAME
  2,000 seeds (common random numbers). Metrics: per-episode implementation-shortfall
  difference (agent − baseline), in bps.
- **Edge** = ALL of: (i) mean paired difference < 0 vs BOTH fixed-TWAP and adaptive-TWAP
  (the always-1.0x policy); (ii) Wilcoxon signed-rank p < 0.01 vs both baselines;
  (iii) the sign holds in >= 4 of 5 training seeds (per algorithm, per regime);
  (iv) the pooled-across-seeds mean difference vs adaptive-TWAP <= −0.05 bps (economic
  materiality floor: CRN pairing can make trivially small differences significant).
- **Null** = anything else. Reported per regime, per algorithm, with the full paired
  distributions. A policy failing the behaviour audit (Step 4.6: action-distribution
  collapse) is reported as INVALID, not null, and excluded from headline claims —
  the L2-DQN-collapse rule.
- No post-hoc threshold changes; if a result sits at a boundary, it is reported at the
  boundary.

## 3b. REVISION 1 (2026-07-06, user-approved) — mechanism-gated changes to G1/G2/G3
## after the first gate run; ORIGINAL results preserved verbatim below

**First run (original criteria): G1 FAIL, G2 FAIL, G3 FAIL** (`step4_gates.json`, kept).

**G3 — two defects in the GATE IMPLEMENTATION (not the env), fixed with evidence:**
(i) the "instant dump" baseline could not dump (the 2.0x action cap turns it into a
front-loaded TWAP — traced fill-by-fill); (ii) ordering comparisons of ~0.1 bps were
run under +/- several bps of background drift entering policies at different fill
times (non-cancelling even on shared seeds). Fix: a true `force_buy_all` dump
(mechanics-only path, unreachable by the agent) and DRIFT-FREE ordering checks (jump
process disabled; legitimate because G3 validates mechanics, not realism).
**G3 re-run: PASS both regimes** — completion 100 %; true dump costs 5-7x TWAP
(calm 0.441 vs 0.059 bps; volatile 0.244 vs 0.052); size-monotone within tolerance.

**G1 — the METRIC was mis-specified for a temporary-impact environment.** Terminal
mid displacement after 300 s measures a footprint that the calibrated refill dynamics
and jump re-forms heal within seconds-to-minutes; it is flat in size (measured: $0.4-
0.7 at every size — preserved). The reaction lives in COSTS. **G1' (revised spec):**
drift-free, at the primary size, (a) SELF-IMPACT: executing the order doubles-ish the
cost of an identical immediate follow-on — pass bar: second-dump mean cost >= 1.25x
first-dump; (b) MONOTONE: second-dump cost increases with size (10/25/44 units,
Spearman rho > 0); (c) RECOVERY: a 5-unit probe's cost at +30 s is below its cost at
+1 s after the dump (the book demonstrably refills). Measured at revision time (so
G1' is verified, not aspirational): calm 2.4x / monotone 0.26->0.44->0.65 / recovery
0.347->0.124 bps; volatile 1.8x / monotone / 0.147->0.076. **SCOPE NOTE (write-up):**
permanent impact is ~zero in this simulator (footprints do not survive jump re-forms);
the modelled reaction is the TEMPORARY-impact channel — the dominant execution-cost
channel in the empirical literature, and the lever TWAP ignores.

**G2 — level vs slope split (mechanism = the pre-existing C5 resolution gap).**
Original G2 conflated two quantities. The LEVEL (cost of crossing the spread) is ~2.2x
too cheap in-sim — the C5 unit-resolution gap, now quantified in cost terms; it is
paid IDENTICALLY by the agent and every baseline, so it cancels in all paired
comparisons. The SLOPE (how cost GROWS with size — the impact component the study
measures) matches reality closely. **G2' (revised spec):** pass iff the sim's
1u->10u cost-growth ratio is within 25 % (relative) of the real ratio, per regime;
the level gap is carried as **caveat C6: "all absolute execution costs are understated
~2.2x; paired policy differences are unaffected; absolute cost levels are not
interpretable"**. Measured at revision time: growth sim 2.33x vs real 2.47x (calm),
sim 2.44x vs real 1.96x (volatile) — bands to be confirmed by the formal re-run.

## 4. Compute/protocol constants

Training: 2M env steps per run (revisit only upward, logged); 5 seeds; DQN + PPO;
per-regime training (agent never told the regime). Evaluation checkpoints every 100k
steps on 200 CRN episodes (curve), final judgement on the 2,000-episode set. Behaviour
audit: action histogram + deadline-residual frequency per seed, BEFORE unblinding
cost results (audit criteria: any single action > 90 % of steps across the final 200
episodes, or deadline residual fired in > 10 % of episodes -> flag INVALID).

**Audit-metric correction (2026-07-06, applied before any cost numbers were
computed):** the residual detector counts only MATERIAL leftovers (> 1 AES unit, the
smallest voluntarily-executable amount). The order (25 BTC = 44.22 units) leaves a
sub-unit fragment that reaches the deadline mechanism for EVERY policy including TWAP,
by construction of unit quantisation; the uncorrected detector flagged 20/20 agents at
a residual frequency of 100 % — mechanically impossible as a behaviour signal (the
first audit file is preserved). The action-share rule is unchanged.

## 4b. REMEDIATION REVISIONS (2026-07-07, logged before any re-run; full detail in
## reports/qrm_step5_remediation.md)

A three-agent adversarial code review found confirmed environment/training bugs and one
design confound (built-in drift). ALL results produced before this date are reclassified
as engineering shakedown. Changes affecting THIS protocol, logged before the re-run:
1. **Audit action-share rule revised:** the >90 %-one-action flag applies only when the
   constant action is 0.0x (do-nothing collapse) OR the material-residual rule also
   fires. Rationale: the old rule would invalidate a constant-pace policy (e.g. always
   2.0x) while the TWAP baselines are themselves single-action policies — an
   inconsistency that could suppress a legitimate schedule. Residual metric now = true
   unexecuted remainder (the over-buy bug that inflated it ~3x is fixed).
2. **executed_frac** now = 1 − deadline_residual/order (was vacuously 1.0).
3. **Variant grouping**: seed-extension runs (\_v1aext) merge into their parent variant
   (\_v1a) so the pre-registered 5-seed escalation is computable.
4. **Environment**: drift removed from the move processes (measured +$2.6/+$9.0 per
   episode — a resampling artifact, not an execution signal; user-approved); engine
   reference-price moves captured (agent-caused permanent impact restored); per-depth
   unit sizes in fills; deadline buys the true remainder only; flow-feature sign fixed;
   explicit jump-binning rule. Frozen §3 verdict conditions are UNCHANGED.

5. **G3 monotonicity tolerance 0.015 bps** (was implementation-detail 0.005): the
   <=1-unit deadline overshoot is a fixed cost, so per-BTC it taxes small orders more
   (~0.007 bps at 5 BTC — mechanism verified arithmetically); the check's purpose is
   catching ~0.2-bps-scale wiring inversions, which it retains.

## 5. PRIMARY VERDICT (2026-07-06) + pre-registered TUNING ROBUSTNESS CHECK

**Primary result (step5/judgement.json, 2,000 CRN episodes/agent): NULL in all four
cells.** Audit: 13/20 valid — ALL five volatile DQN seeds + one calm DQN seed collapsed
into deadline-dump reliance (residuals 18-100 % of the order; the L2 pathology
reproduced), one PPO volatile seed mildly flagged (13 %). Valid results: PPO calm =
exact tie (5 seeds within ±0.005 bps, n.s.); PPO volatile = slightly worse (+0.01 to
+0.06, mostly significant); DQN calm valid seeds = worse (+0.04 to +0.10). Zero seeds
met any edge condition. Sensitivity: counting INVALID seeds too changes nothing (no
seed anywhere was negative-significant) — the audit did not drive the null.

**Tuning robustness check — the variant table is CLOSED as of this section's commit.**
Purpose: close the "maybe you just trained it badly" attack. Every variant below is a
ONE-FACTOR change from the primary configuration, motivated by a named suspicion;
judged by §3 verbatim; 3 seeds x both regimes each; any variant meeting the edge
conditions at 3 seeds escalates to the full 5-seed protocol before any claim. No
variants may be added, dropped, or re-parametrised after Wave-1 results exist, absent
a new documented mechanism.

*Wave 1 — PPO, prime suspects (18 runs):*
| V1a | net_arch [64, 64] | capacity: the inherited [30x5] trunk was sized for 7 inputs; ours is 27 |
| V1b | net_arch [128, 128] | same suspicion, larger step |
| V2  | reward scale x100 | per-step rewards are ~0.001-0.05 bps — numerically tiny learning signal |

*Wave 2 — PPO secondary + DQN mechanism arm (run if Wave 1 null, or to bound a Wave-1 edge):*
| V3a/b | learning rate 1e-3 / 1e-4 (vs 3e-4) | step-size mismatch |
| V4a/b | ent_coef 0.0 / 0.05 (vs 0.01) | exploration tax vs premature settling |
| V5  | n_steps 8192 (vs 2048) | credit assignment across 300-step episodes |
| V6  | 10M steps on the best Wave-1 variant | "needed longer" (budget rule: upward only) |
| D1  | DQN exploration_final_eps 0.05 + anneal_frac 0.5 | the collapse mechanism: the deadline region is under-experienced when epsilon decays early |
| D2  | DQN reward x100 + net [64, 64] | joint scale+capacity variant of the collapse fix |

Judgement of the whole check: if EVERY variant is null under §3 → the primary null is
declared TUNING-ROBUST and reported with this full table. Compute is never a reason to
skip a listed variant.

**Pre-registered continuation rules (added 2026-07-06, BEFORE Wave 1 launched):**
1. *Wave 3 — combinations (runs regardless of Wave-1/2 outcomes):* (a) the better of
   V1a/V1b combined with V2 (capacity + signal strength are mechanistically
   complementary); (b) the two variants with the best pooled cost vs adaptive-TWAP
   combined, if different from (a). 3 seeds x 2 regimes each; same §3 judgement; same
   escalation rule.
2. *Sweep-design selection:* the design carried into the size ladder and the 10-min
   horizon variant = the variant (or combination) with the best pooled cost vs
   adaptive-TWAP across both regimes among all HEALTHY (audit-passing) configurations;
   ties break toward the simpler design. Selection is mechanical from this rule — no
   discretionary choice after seeing results.
3. *Near-miss escalation in the sweeps:* any sweep cell with pooled cost vs
   adaptive-TWAP <= −0.02 bps and p < 0.05 (but short of the §3 edge) triggers a
   focused follow-up at that cell: full 5-seed protocol plus the Wave-3 combination
   designs, before any conclusion about that cell is reported.

## 6. OUT-OF-SAMPLE CONFIRMATION PROTOCOL (R8b) — pre-registered 2026-07-08, BEFORE any confirmation run

**Why this exists.** The corrected primary campaign (`step5_v2`, boundary null in all 4
cells) showed a consistent but SUB-THRESHOLD PPO tilt: 9/10 seeds cheaper than
adaptive-TWAP; pooled volatile −0.039 bps, calm −0.019 bps; but 0/10 seeds significant at
p<0.01 and the −0.05 materiality floor unmet. That tilt was noticed IN the primary data, so
it cannot be claimed from the same data (circular). This protocol is the SINGLE, pre-committed
replication test on data that played no role in generating the hypothesis. It is committed
here BEFORE the confirmation runs execute. Its verdict is TERMINAL: no re-training-and-
re-judging, no threshold edits, no additional confirmation waves.

**6.1 Target configuration (mechanical, not discretionary).** The confirmation target = the
single configuration carried forward by §5 sweep-design selection (the healthy, audit-passing
config with the best pooled cost vs adaptive-TWAP across both regimes). If the §5 tuning table
produces NO healthy config that beats the primary PPO on pooled cost, the target = the primary
PPO configuration (inherited net_arch, reward unscaled, lr 3e-4, ent_coef 0.01, n_steps 2048).
Recorded here at selection time, before confirmation runs.

**6.2 Primary vs secondary regime (fixed now).** Primary confirmation regime = the regime in
which the target config shows the LARGER pooled discovery effect vs adaptive-TWAP (currently
VOLATILE, −0.039 vs calm −0.019). Only the PRIMARY regime carries the headline "confirmed
edge"; the secondary (currently calm) is reported as corroboration, NOT an independent claim
(this avoids regime-shopping across two tests). If tuning changes which regime is larger, the
primary is re-derived by this same rule and logged here before confirmation runs.

**6.3 Fresh, disjoint data (seeds fixed now).**
- Retrain the target config on FIVE new training seeds, indices 5,6,7,8,9 (training-episode
  bases 5e7..9e7 via the existing seed*1e7 rule) — disjoint from the primary's seeds 0-4.
- Evaluate on a fresh CRN block: `eval_seed0 = 9_000_000`, `n_eval = 2_000` (episodes
  9,000,000..9,001,999) — disjoint from all training-episode seeds, the curve-eval block
  (1e6), and the primary judgement block (5e6). Behaviour audit on the same block.
- Same env stack, same centered move process, same order/horizon (25 BTC, 300 s).

**6.4 Pass/fail rule (binary, pre-committed).** Behaviour audit (§4.6) FIRST; seeds that
collapse (do-nothing-then-dump) are INVALID and excluded. CONFIRMATION PASSES in a regime iff
ALL of:
(a) pooled mean paired difference vs BOTH fixed-TWAP and adaptive-TWAP < 0 (cheaper), pooled
    across the valid new seeds on the fresh block;
(b) across-seed significance p < 0.05 (one-sided, H1: mean difference < 0) vs adaptive-TWAP,
    computed on the valid new-seed mean differences (t-test; Wilcoxon signed-rank reported as
    a robustness cross-check);
(c) the cheaper direction holds in >= 4 of 5 valid new seeds (single-seed-artifact guard —
    the L2 lesson).
The bar is 1-in-20 (p<0.05), NOT the §3 screening bar of 1-in-100: this is ONE pre-registered
replication hypothesis, so it does not carry the multiple-agent burden that justified the
stricter screening bar. Effect size + 95% CI are reported regardless of pass/fail (ASA-style;
no bright-line-only reporting). Significance is tested vs adaptive-TWAP (the harder,
self-correcting benchmark); the vs-fixed-TWAP numbers are reported alongside.

**6.5 Verdict handling (terminal).**
- PASS in the primary regime → PPO has a confirmed, if small, reactive-market execution edge
  in that regime. Report effect size + CI. RQ3 per-regime attribution proceeds with a genuine
  effect to explain.
- FAIL → the boundary null is the FINAL, hardened headline: a null that survived pre-registered
  tuning AND an out-of-sample replication test. Report the full fresh-block distributions.
- Either way this is the END of the edge search. No new confirmation waves, no threshold edits,
  no re-rolling. Further work is attribution/robustness on the frozen verdict, not a re-test.

**6.6 Invariants.** The CONFIRMATION block (9e6) is the held-out set: it is NEVER used for
tuning screening or model selection — only for the one §6 confirmation. The 5e6 block is the
DEVELOPMENT/selection block (primary judgement + §5 tuning screening); a tuning winner selected
on 5e6 is expected to be optimistically biased THERE, which is exactly why §6 re-tests it on the
untouched 9e6. The L2 test set stays sealed. This §6 is committed before the confirmation runs;
if 6.1/6.2 re-derive the target/primary regime after tuning, those are logged here (with the
tuning-result reference) BEFORE confirmation executes.
