# Step 4 — pre-registered criteria (FROZEN 2026-07-06, before any gate or training run)

Companion to BUILD_PLAN "STEP 4 — PLAN LOCKED (user-approved 06/07/2026)". The design
decisions are recorded there; this file freezes the NUMBERS: gate pass bands and the
definition of "beats TWAP". Revisions only with a stated mechanism, logged here (the 3f
discipline). Nothing below was computed from any simulation of the agent environment.

Path convention (added 2026-07-15): every `$S/...` raw-result path in this file resolves with
`$S` = `/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4`. The JSON
files at those paths are the source of truth for every number; markdown is a convenience copy.

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

**3a. TRANSPARENCY ADDITION (2026-07-09) — auto-report the across-seed view alongside the
frozen per-seed screen. CHANGES NO RULE OR VERDICT above.** The §3 screen is deliberately
strict (per-seed Wilcoxon at p<0.01), which by design can stamp a small-but-consistent edge
as "null" even when every seed points the same way (the PPO-volatile v3 case: 0/5 seeds
individually significant, yet 5/5 cheaper, across-seed t p=0.006). To make sure such an edge
is never silently mislabelled, `step5_judgement.py --mode screen` now attaches an
INFORMATIONAL `across_seed` block to every cell's verdict: pooled edge vs both benchmarks,
the one-sided across-seed t-test p, the 95% CI, and #cheaper. It is reported ALONGSIDE, never
in place of, the frozen EDGE/ESCALATE flag — the frozen verdicts are byte-identical
before/after (asserted on the sealed v3 file; original kept as `judgement_preAcrossSeed_backup.json`).
The block also guards the OPPOSITE error: when the audit invalidated seeds it runs on
survivors only, so it carries `n_seeds_total`, `trustworthy=False`, and a survivorship
warning whenever any seed was dropped (e.g. DQN-volatile v3: across-seed p=0.042 but only
2/5 valid -> flagged, NOT an edge). This is the same across-seed statistic already
pre-registered for §6 confirmation, now merely surfaced at the screening stage too.

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

> **SUPERSEDED (2026-07-08).** The `step5/judgement.json` verdict below is the ORIGINAL buggy
> campaign (reclassified as engineering shakedown by §4b). It was replaced by the corrected
> `step5_v2` (remediation R7), which was ITSELF found to be drift-confounded by the pre-launch audit
> and is being replaced by the drift-fixed `step5_v3`. The pre-registered TUNING TABLE below (Waves
> 1-3 + continuation rules) REMAINS VALID and is what R8a runs on the drift-fixed env. Latest state:
> `reports/qrm_prelaunch_audit_2026-07-08.md`.

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

## 5c. SELECTION BATCH RESOLUTIONS (logged 2026-07-10, BEFORE the batch runs) — how the §5
## continuation rules apply to the actual R8a screen outcome (`step5_tuning_v3`, sealed 2026-07-09)

The screen's strict 3-seed ESCALATE trigger (>=2/3 per-seed significant) fired for ZERO variants.
The §5-rule-3 consistency criterion (pooled <= -0.02 AND across-seed p < 0.05) is met by SEVEN
cells. Resolutions, all fixed here before any new run:

1. **Escalation set = ALL SEVEN qualifying cells, no discretionary subset:** ppo_volatile
   {v3b, v3a, v2, v1b, v4b, v5} + ppo_calm {v4a}. Each gets seeds 3,4 added (config byte-identical
   to its 3-seed siblings, only the seed differs) -> full 5-seed evidence in the qualifying regime.
   14 runs.
2. **V6 target ambiguity + resolution.** "Best Wave-1 variant" is ambiguous: by the across-both-
   regimes metric it is V1b (-0.031 vs V2's -0.029); by the volatile/primary-regime metric (§6.7
   designates volatile primary) it is V2 (-0.055 vs -0.042). RESOLUTION: literal V6 = V2 @ 10M steps
   (tag `_v6`), resolving toward the primary-regime metric. ADDITION (new, mechanism-named): `_v6b`
   = V3b @ 10M steps — lr 1e-4 makes per-step updates ~3x smaller than base, so the "needed longer"
   suspicion applies MOST strongly to v3b, and v3b is the selection front-runner, so the 10M axis
   must be tested where it matters. V1b @ 10M is NOT run (last of the three contenders on the
   primary metric; its capacity dial enters the batch via combo `_w3a`; if `_w3a` wins selection,
   the 10M axis extends to it before confirmation). 6 + 6 runs, both regimes, seeds 0-2.
3. **Wave-3 combo (b) interpretation.** Rule (b) says "the two variants with the best pooled cost
   combined" — the literal top-2 (v3b, v3a) are the SAME dial (learning rate) set to two values and
   cannot coexist in one agent. Mechanical resolution: the best two COMBINABLE (different-dial)
   variants = v3b x v2. So: `_w3a` = v1b x v2 (rule (a), unambiguous: v1b beats v1a on both regimes)
   and `_w3b` = v3b x v2 (rule (b) as resolved). 3 seeds x 2 regimes each = 12 runs.
4. **Placement + judging:** new runs train into `runs_tuning_v3/` (same dir, so grouping pools the
   escalated seeds with their siblings). The selection judgement re-scores the WHOLE dir (60 + 38 =
   98 runs) into a NEW out dir `step5_selection_v3/` — `step5_tuning_v3/` stays sealed as the 60-run
   screen record. The scorer is deterministic on the same eval block (proven: --mode screen exactly
   reproduced the v3 verdicts), so re-scored old cells reproduce their sealed numbers.
5. **Selection then applies §5 continuation rule 2 verbatim:** best pooled cost vs adaptive-TWAP
   across both regimes among HEALTHY (audit-passing) configurations, ties toward simpler; mechanical,
   no discretion. The winner goes to §6/§6.7 confirmation unchanged.
6. **Seeds:** escalation uses training seeds 3,4; V6/combos use 0-2. Confirmation seeds 5-9 remain
   untouched; the sealed 9e6 eval block remains untouched.

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

## 6.7 REVISION (pre-registered 2026-07-09, informed by the clean v3 primary, BEFORE any R8 run)

Addresses the three methodology-audit items (M1 regime-selection, M2 disclosure+power, M3
null-branch), locked in now while the fresh 9e6 block is still untouched.

**M1 — the confirmation is now a SINGLE-regime test, so no regime-selection alpha inflation.** The
clean v3 primary shows the favourable PPO signal is VOLATILE-specific: volatile 5/5 seeds cheaper,
pooled −0.047, across-seed p=0.006; calm collapsed to a tie (pooled −0.006, p=0.18) once the drift
was removed. So the confirmation targets ONE regime, VOLATILE, chosen NOT by "larger of two" but
because it is the only regime with any signal AND is a-priori the expected one on mechanism (volatile
has larger market impact and larger book reaction, so a reactive-market edge should live there). Calm
is reported as a documented clean null, NOT tested for an edge. This supersedes §6.2's "larger of two"
rule and removes the max-of-2 concern.

**M2 — full disclosure of how §6.4 differs from the §3 screening test, with justification.** §6.4
relaxes §3 on three axes; all are deliberate and disclosed: (i) p 0.01→0.05 — a single pre-registered
replication carries no multiple-agent burden; (ii) two-sided→one-sided — the hypothesis has a
committed direction (cheaper), and the effect was directional in discovery; (iii) both-baselines→
significance-vs-adaptive-only — adaptive-TWAP is the harder, self-correcting benchmark, and §6.4(a)
still requires the mean to be cheaper vs BOTH. POWER NOTE (pre-registered): at the observed volatile
effect (~−0.047 bps, across-seed SD ~0.024 over 5 seeds), the one-sided n=5 t-test at α=0.05 has
roughly ~80% power to detect it — i.e. the bar is genuinely HARD to clear, not lowered to fit; a null
confirmation is a real possibility, which is the point.

**M3 — null-branch deliverable committed NOW (before the result).** If the confirmation FAILS (or is
not run), the dissertation's contribution is the reaction-inclusive, mechanistically-explained,
per-regime NEGATIVE result: (a) a reactive QRM execution environment built + calibrated + bug-hardened
+ drift-neutralised on real BTC L4; (b) the demonstration that a naive setup manufactures a spurious
edge (the drift confound) and that neutralising it dissolves the calm "edge" while leaving a genuine
but sub-threshold volatile signal; (c) per-regime SHAP/ablation attribution of WHY the effect is
volatile-specific and why it stays small. This is a valid, defensible contribution independent of
whether the confirmation passes; it is committed here so the pivot cannot read as post-hoc.

## 6.8 CONFIRMATION VERDICT (recorded 2026-07-11 — the one-shot §6 run happened; outcome logged verbatim)

Executed exactly as pre-registered (§6.4 rule, §6.7 single-regime volatile, sealed block
seed0=9,000,000 verified untouched beforehand, audit-before-costs, fresh seeds 5-9):

**PASS = FALSE — the volatile edge DID NOT REPLICATE.** pooled vs adaptive = -0.0023 bps
(vs -0.0628 on the development block); across-seed one-sided t p = 0.3785 (rule: <0.05);
cheaper in 3/5 valid seeds (rule: >=4/5); 95% CI [-0.0217, +0.0171]; all 5 seeds
behaviour-valid (no collapse). Raw record: `$S/step5_confirm_v3a/judgement.json`.

Per §6.5/§6.7 M3 (pre-committed): the project headline is a BOUNDARY NULL; the development-set
signal is reported as unreplicated; NO re-runs, NO second confirmation attempt, NO threshold
revisiting. The M3 null-branch deliverable is now the dissertation's contribution frame.

## 6.9 POST-VERDICT PROCESS AUDIT (2026-07-11, user-requested) — selection-rule deviation
## DISCLOSED + three-block diagnostic

**Audit trigger:** the edge trajectory (base -0.047 -> v3b@3seeds -0.084 -> v3a@5seeds -0.063 ->
confirmation -0.002) raised the question whether selection/testing was correct.

**Finding 1 — the confirmation EXECUTION was correct (verified):** the judged runs are exactly the
5 fresh-seed v3a builds (folder list == judgement run list; all metas exact: lr 1e-3, 2M steps,
tag _v3a); block = 9,000,000 as pre-registered; §6.4 rule applied as written. A from-scratch
reproduction of a recorded number using library primitives (not the judge script) was run as an
independent check (result recorded in the live doc when complete).

**Finding 2 — a SELECTION-RULE DEVIATION (execution error, disclosed):** §6.1/§5-rule-2 as written
select "best pooled cost vs adaptive-TWAP ACROSS BOTH REGIMES among HEALTHY configurations."
The actual selection (2026-07-10) ranked VOLATILE-ONLY and additionally restricted to
escalated-to-5-seed configs (a filter not in the written rule). Recomputed rankings from
`step5_selection_v3` under the literal rule: strict health (no invalid seeds anywhere), simple
average: v1b -0.0386 > v4a -0.0343 > w3a -0.0341 (v3a -0.0307); valid-run-weighted: v1b -0.0430 >
v3a -0.0387; lenient health: v3b -0.0421 > v1b -0.0386. Under NO across-both reading is v3a the
literal pick; it wins only volatile-only (-0.0628, and only by 0.0008 over w3a). The volatile-only
reading has a coherent rationale (§6.7 made volatile the sole confirmation regime, and calm shows
no signal anywhere) but it was NOT the rule as written and is recorded here as a deviation.

**Finding 3 — three-block diagnostic (why the trajectory happened):** the final-checkpoint score of
every relevant config on the 1e6 MONITOR block (n=200/run, never used for any decision) vs the two
judgement blocks:
| config (volatile) | 1e6 monitor block | 5e6 dev block | 9e6 sealed block |
|---|---|---|---|
| base PPO | +0.066 | -0.047 | (not run) |
| v3a | +0.147 | -0.063 | -0.002 |
| v3b | +0.114 | -0.060 | (not run) |
| v1b | +0.057 | -0.056 | (not run) |
| confirm agents (fresh seeds) | +0.126 | (not run) | -0.002 |
EVERY config flips sign with the block: worse-than-TWAP on 1e6, better on 5e6, ~zero on 9e6.
Interpretation: the dev-block edge was common-mode BLOCK LUCK shared by all variants (they were all
measured on the same 2,000 markets), not config skill; across-variant "robustness" was therefore
correlated evidence, not replication. Block-to-block variance of the paired mean dominates the
within-block CRN significance — a design lesson recorded for the write-up.

**Consequence assessment for the deviation:** likely NIL for the headline. The literal-rule picks
(v1b or v3b) carry the same shared-block inflation, and the 1e6 block provides direct independent-
block evidence AGAINST both (+0.057 / +0.114, i.e. worse than TWAP there). No config shows
cross-block consistency. Decision on any remedial test of the literal-rule target rests with the
user; the recorded recommendation is NOT to run one (sequential-testing multiplicity + §6
terminality + the diagnostic above), and instead to disclose this deviation in the write-up.

## 7. ROBUSTNESS SWEEP PROTOCOL (pre-registered 2026-07-12, BEFORE any sweep run; §1 design executed
## on the null branch)

**Purpose.** With the headline a §6.8 boundary null, the sweeps answer: does "agent matches TWAP,
does not beat it" HOLD across order size and horizon, or is the null specific to 25 BTC / 5 min?
Either answer completes the §1 design (the L2 track's multi-axis pattern).

**7.1 Config (fixed).** The §5-selected config: PPO, lr 1e-3, all else base (2M steps, net 30x5,
reward 1.0). The §6.9 selection-metric deviation is disclosed and does not alter the sweep design
(under a null, any top-cluster config tells the same story; the recorded selection is used).

**7.2 Cells + seeds.** New training, 3 seeds (0,1,2) x 2 regimes per cell:
- Size ladder: 5 BTC (tag `_v3aB5`), 12.5 BTC (`_v3aB12`), 50 BTC (`_v3aB50`), horizon 300 s.
- Horizon variant: 25 BTC at 600 s / 600 decisions (`_v3aH600`). Cadence stays 1/s; per-step gamma
  unchanged (the locked fixed-rate rule). NO cadence sweep (pre-registered out).
- Centre point 25 BTC/300 s = the EXISTING sealed v3a numbers in `step5_selection_v3` (volatile 5
  seeds, calm 3) — not re-run.
Total new runs: 24. Dirs: `runs_sweep_b5/ b12/ b50/ h600/` (6 runs each), judged per-dir into
`step5_sweep_b5/ b12/ b50/ h600/` with matching `--order-btc`/`--env-steps`.

**7.3 Evaluation.** Dev block eval_seed0=5,000,000, n=2,000 CRN episodes, audit BEFORE costs, both
TWAP baselines re-run per cell at the cell's size/horizon, screen-mode verdicts + the informational
across-seed block. Sealed blocks (9e6 and any future confirmation block) are NOT touched.

**7.4 Audit thresholds unchanged — transfer VERIFIED before this registration.** The residual
criterion (episodes with >1 unit left at deadline; cap 10%) was checked on TWAP itself at all four
sizes x both regimes (n=200, dev seeds): res_frac <= 0.01 everywhere (worst 0.010, fixed-TWAP 5 BTC
volatile). The 25-BTC-calibrated audit is fair across the ladder; no threshold change.

**7.5 Interpretation cap + if-edge procedure (pre-committed).** No edge claim can arise from a
sweep cell directly. Any cell meeting §5-rule-3 (pooled <= -0.02 bps AND across-seed p < 0.05,
fully valid) triggers, in order: (i) escalation to 5 seeds; (ii) CROSS-BLOCK replication on a
second dev block (eval_seed0 = 6,000,000, n=2,000) — the check that would have caught the 25-BTC
block-luck illusion; (iii) only if BOTH survive, at most ONE newly pre-registered sealed
confirmation on a fresh never-used block, disclosed as an additional test in the §6 family. The
§6.8 headline stands unless that confirmation passes.

**7.5a SURVIVAL CRITERIA + EXECUTION DETAIL (pinned 2026-07-15, BEFORE the first §7.5
execution — user-approved).** §7.5 named the steps but not the numeric pass bars; they are fixed
here before any escalation run. Triggering groups (grid judgements 2026-07-15): ppo-calm
50 BTC/600 s (`runs_grid_b50h600`, pooled -0.0539, p=0.014) and ppo-calm 25 BTC/1200 s
(`runs_grid_b25h1200`, pooled -0.0629, p=0.0003).
- **Step (i) — escalation.** Train seeds 3 and 4 for each group, identical config (PPO lr 1e-3,
  all else base, same order size / env steps / tag, same dirs). Re-judge each full dir into a NEW
  `step5_esc_<cell>/` (the original `step5_grid_<cell>/` stays the untouched record of the
  trigger). SURVIVES iff, for the calm group with 5 seeds on the dev block (5e6, n=2000):
  pooled vs adaptive <= -0.02 bps AND one-sided across-seed t p vs adaptive < 0.05 AND cheaper
  vs adaptive in >= 4/5 seeds AND >= 4/5 seeds audit-valid. (Basis = §5-rule-3's adaptive-TWAP
  form, matching the trigger itself; vs-fixed numbers are recorded alongside but do not gate.)
- **Step (ii) — cross-block.** Same 5 agents, NO retraining, judged at eval_seed0 = 6,000,000
  (verified 2026-07-15: no judgement on record has ever used this block; first use), n=2000,
  into `step5_xblock_<cell>/`. SURVIVES iff the identical condition holds on this block.
- **Step (iii) — sealed stage (only if a group survives BOTH).** Returns to the user BEFORE any
  registration, with two pre-stated options: (a) literal §7.5 — ONE test at alpha=0.05 on the
  group with the stronger cross-block result; or (b) if both groups survive, BOTH tested at
  alpha=0.025 each (same familywise error, more information; would be a disclosed, dated
  amendment). Either way: fresh never-used block, one shot, third test in the §6 family (the
  §6.11 closure explicitly carves out the §7.5 route), familywise multiplicity stated in the
  registration. Groups that fail (i) or (ii) are reported in full and closed.

**7.5b LADDER VERDICT (recorded 2026-07-15 — §7.5a executed exactly as written).**
Step (i): BOTH groups survived at 5 seeds (50/10min calm pooled -0.0432, p=0.0037, 5/5;
25/20min calm -0.0609, p=0.0001, 5/5; determinism of the original seeds verified).
Step (ii), reserve block 6,000,000 first use: **BOTH FAIL** — 50/10min calm -0.0095 (p=0.17,
4/5); 25/20min calm +0.0177 (p=0.96, 1/5, sign-FLIPPED). Ladder CLOSED per the fail rule;
step (iii) never reached; NO third sealed test spent; the 6e6 reserve block is now SPENT as
a cross-block instrument. The §6.8 boundary-null headline stands across the full §7.7 grid.
Raw: `$S/step5_esc_{b50h600,b25h1200}/` + `$S/step5_xblock_{b50h600,b25h1200}/`.

**7.6 Code surface (logged; defaults verified).** New flags: `train_reactive --env-steps`;
`step5_judgement --order-btc --env-steps` (env + audit + baselines threaded). At defaults the new
code EXACTLY reproduces sealed records (audit entry of `runs_confirm_v3a/ppo_volatile_s5_v3a`
reproduced field-for-field) and the 219-test suite passes. meta.json now records `env_steps`;
judgement.json records `order_btc` + `env_steps` (provenance).

## 6.10 REMEDIAL CONFIRMATION (pre-registered 2026-07-13, BEFORE the run) — the §6.9 deviation remedy

**Purpose.** §6.9 disclosed that the §6.1 selection rule as written ("best pooled cost across BOTH
regimes among healthy configs") selects ppo_v1b (net 128x128), not the volatile-only-ranked v3a
that was tested. This is the ONE remedial test of the literal-rule target, closing the "the
pre-registered pick was never tested" gap with data. User-approved 2026-07-12/13.

**Design (mirrors §6/§6.7 exactly):**
- Target config: PPO, net_arch [128,128], all else base (2M steps, reward 1.0, lr 3e-4 default) —
  verified from `runs_tuning_v3/ppo_volatile_s0_v1b/meta.json` (no other overrides).
- Regime: VOLATILE only (§6.7 M1, the regime of record; v1b's calm pooled -0.021 is below the
  materiality floor and calm is excluded a priori as before).
- Fresh training seeds 10,11,12,13,14 (verified unused anywhere; episode bases 100M-140M, verified
  disjoint from every eval block).
- Sealed eval block: eval_seed0 = 13,000,000, n = 2,000 (verified untouched by any prior result;
  distinct from the spent 9e6 block and the 6e6 cross-block reserve).
- Pass rule = §6.4 verbatim: pooled < 0 vs BOTH benchmarks AND across-seed one-sided t p < 0.05
  vs adaptive AND cheaper in >= 4/5 valid seeds. Audit before costs. ONE SHOT, no re-runs.
- Dirs: `runs_confirm_v1b/` -> `step5_confirm_v1b/`.

**Multiplicity disclosure (fixed now):** this is the SECOND sealed test in the §6 family (first:
v3a on 9e6, FAIL). Under a global null, the familywise chance of >=1 false pass across the two
tests is ~9.75% at alpha=0.05 each; any pass here is reported WITH that caveat. Expectation on
record: likely FAIL — v1b reads WORSE than TWAP (+0.057) on the independent 1e6 monitor block and
shares the dev-block-luck structure (§6.9). The test is run for completeness of the record.

**Terminality:** whatever the outcome, the confirmation family is CLOSED at two tests. No further
sealed confirmations for any config, regime, or sweep cell except via the §7.5 procedure (which
requires cross-block replication first).

## 7.7 SWEEP EXPANSION (pre-registered 2026-07-13, BEFORE any expansion run) — full grid + cadence
## for maximal null-coverage defensibility (user-approved 2026-07-13)

Rationale: §7 delivered a CROSS (size varied at 5-min; horizon varied at 25 BTC only) and no
cadence axis — faithful to §1 but lighter than the L2 three-axis design. To make the null maximally
defensible (a complete coverage grid an examiner cannot poke a corner in), the sweep is EXPANDED.
Methodologically safe: expanding a robustness analysis on an established null cannot p-hack an edge
that is not being claimed, and §7.5's if-edge procedure (escalate -> cross-block -> at most one
sealed test) caps any accidental trigger. All cells use the §5-selected config (PPO lr 1e-3, else
base, 2M steps, seeds 0/1/2, both regimes), dev block 5e6, n=2000, audit-before-costs.

**PART A+B — full size x horizon grid (4x4 = 16 cells).**
Sizes {5, 12.5, 25, 50} BTC x horizons {2.5, 5, 10, 20 min} = {150, 300, 600, 1200} s (env_steps).
Rationale for 4 horizons incl. 20-min: the long horizon is where an execution edge is MOST likely
(max freedom to time liquidity), so a null there is the single most convincing cell.
- ALREADY RUN (5 cells): 5-min column {5,12.5,25,50 BTC} = `runs_sweep_b5/b12/b50` + selection
  25BTC; and 25 BTC/10-min = `runs_sweep_h600`.
- NEW (11 cells x 6 runs = 66): entire 2.5-min column (4 sizes); 10-min at {5,12.5,50} (3); entire
  20-min column (4). Each new cell -> own `runs_grid_<cell>/` + `step5_grid_<cell>/`, tag encodes
  size+horizon (e.g. `_gS5H150`), judged with matching `--order-btc`/`--env-steps`. NOTE: 20-min
  cells have 1200-decision episodes -> ~4x slower to judge (scheduling only).

**PART C — cadence check (12 runs + a verified env change).**
At the PRIMARY cell only (25 BTC / 5-min), test decision cadence {0.5 s, 2 s} vs the existing 1 s
(2 new cells x 2 regimes x 3 seeds = 12). NOT crossed with size/horizon (a 3-way factorial is
implausible-interaction gold-plating). REQUIRED env work, done + VERIFIED before these 12 runs:
(1) make `INTERVALS_PER_DECISION` an env parameter (thread through episode-length, step loop,
pace-multiple/carry accounting); (2) re-derive gamma by the fixed-rate rule (0.995^(cadence_s/60));
(3) re-derive + sanity-check the fixed/adaptive TWAP baselines at the new cadence (must still
complete ~100%); (4) re-check the behaviour audit at the new decision count (600 decisions at 0.5s
for a 300s episode; verify TWAP residual < 10% cap, as done for the 10-min variant); (5) confirm
realism gates unaffected (env mechanics unchanged); (6) add unit tests; (7) confirm defaults still
reproduce sealed records + 219 tests pass.

**Verdict handling.** Same §7.5 interpretation cap for every new cell. Expected outcome: null
everywhere (four independent falsification lines already stand); the value is COMPLETENESS, stated
as such. Sequencing: A+B first (no code risk), then C as its own verified sub-project. Runs AFTER
the §6.10 remedial confirmation completes.

**PART D — DQN cross-setting probe (added 2026-07-15, BEFORE any probe run; decision pending).**
Motivation (user, 2026-07-15): the grid (Parts A+B) is PPO-only, justified by DQN's audit
disqualification at the primary cell (7/10 collapsed; both §5 fix variants d1/d2 failed to cure
it). That scopes the defensible DQN claim to the primary setting; an examiner may ask whether the
collapse is setting-specific (e.g. a 2.5-min deadline leaves less room to do nothing). Part D
converts that caveat into evidence: train DQN (base config) at a SMALL set of grid cells chosen
for informativeness, both regimes, 3 seeds, audit-before-costs as always. PRIMARY ENDPOINT =
behaviour-audit outcome (does the collapse persist), not the cost edge; any cost claim would go
through the frozen §3/§7.5 machinery unchanged.

**FINALISED REGISTRATION (2026-07-15, user-approved Option B, BEFORE any probe run).**
- Cells (3, chosen to bracket the collapse mechanism): **5 BTC x 2.5-min** (easiest execution
  problem + least room to idle: if DQN trades properly anywhere, here), **25 BTC x 2.5-min**
  (primary size, shortest deadline: isolates the deadline effect), **25 BTC x 20-min**
  (maximum idle room: if idle-driven, collapse should be worst here).
- Config: DQN BASE per §5 (the collapse was diagnosed on base; d1/d2 failed to cure it).
  2M steps, both regimes, seeds 0-2 -> 18 runs. Tags `_dqS{5,25}H{150,1200}`; dirs
  `runs_dqnprobe_<cell>/` (kept separate from the PPO grid dirs); judged with matching
  --order-btc/--env-steps into `step5_dqnprobe_<cell>/`, dev block 5e6, n=2000,
  audit-before-costs.
- PRIMARY ENDPOINT: per-cell behaviour-audit outcome (valid/collapsed counts + do-nothing
  share + deadline-residual). Cost results secondary, frozen rules unchanged; the reserve
  block is NOT touched by this probe (spent, §7.5b).

**PART E — DQN library-default update-rhythm variant "d3" (pre-registered 2026-07-16,
BEFORE any run; user-approved).** Motivation: argument bank §N5+correction disclosed that the
base DQN's update rhythm deviates from the SB3 library default (train_freq 100 / batch 1024 =
~20k large updates vs the default 4/32 = ~500k small ones; total gradient sample-throughput is
MATCHED ~16-20M either way, verified from source + saved models). An examiner can attribute the
collapse to this coarse rhythm rather than to value-based learning. Part E closes that with data.
- Variant d3 = DQN BASE with exactly ONE named change, "library-default update rhythm":
  `--dqn-train-freq 4 --dqn-batch-size 32` (new flags added 2026-07-16; defaults-identity
  verified byte-exact + 219 tests pass, per the §7.6 discipline). Everything else base.
- Cells (3): 25 BTC/5-min (primary; the flagship claim), 25 BTC/2.5-min (the 0/6 total-collapse
  cell), 5 BTC/2.5-min (the healthy control — must stay healthy for interpretability).
  x 2 regimes x 3 seeds = 18 runs. Tags `_d3S25H300`, `_d3S25H150`, `_d3S5H150`;
  dirs `runs_d3_{b25h300,b25h150,b5h150}/`; judged with matching flags into
  `step5_d3_<cell>/`, dev block 5e6, n=2000, audit-before-costs.
- PRIMARY ENDPOINT: behaviour-audit outcome vs the base-config cells (does the collapse
  persist under the default rhythm). Interpretation pre-stated: d3 still collapsing -> the
  failure mode is robust to the rhythm objection; d3 healthy -> the cause is LOCATED (update
  granularity), reported as such. Cost results secondary through the frozen §3/§7.5 machinery;
  no reserve/sealed block touched.

**PART D OUTCOME (recorded 2026-07-16 — executed exactly as registered above).** 18/18
trained, integrity ALL PASS, judged 2026-07-16. Audit verdicts: 5 BTC/2.5-min **4/6 valid**;
25 BTC/2.5-min **0/6 valid**; 25 BTC/20-min **2/6 valid** (primary-campaign reference:
25 BTC/5-min = 3/10). Conclusion: the collapse is SYSTEMATIC at the primary size (all
deadlines), SIZE-driven (the idle-room hypothesis is refuted: shortest deadline at 25 BTC is
the worst cell), and calm-concentrated (1/9 calm valid vs 5/9 volatile). No cost trigger in
any group (only fully-valid group: 5 BTC/2.5-min volatile, pooled -0.021, p=0.26 = null).
Part D CLOSED; nothing escalates. Raw: `$S/step5_dqnprobe_{b5h150,b25h150,b25h1200}/`.

## 6.11 REMEDIAL CONFIRMATION VERDICT (recorded 2026-07-13 — §6.10 executed, outcome verbatim)

Executed exactly as §6.10 pre-registered (target ppo_v1b net 128x128, fresh seeds 10-14, sealed
block 13,000,000 verified untouched beforehand, audit-before-costs, §6.4 rule).

**PASS = FALSE — the literal-rule pick ALSO did not replicate.** pooled vs adaptive = -0.0022 bps
(vs -0.056 on the development block); across-seed one-sided t p = 0.3936 (rule: <0.05); cheaper in
2/5 valid seeds (rule: >=4/5); 95% CI [-0.0232, +0.0188] bracketing zero; all 5 seeds valid.
Raw: `$S/step5_confirm_v1b/judgement.json`.

**CONFIRMATION FAMILY NOW CLOSED at two tests, both FAIL** (v3a on 9e6: -0.0023; v1b on 13e6:
-0.0022). This RESOLVES the §6.9 selection-metric deviation with data: whether you rank by the
volatile-only metric (-> v3a) or the literal across-both-regimes rule (-> v1b), the selected
agent does not beat TWAP out-of-sample. The disclosed deviation is therefore immaterial to the
conclusion; both candidate champions return ~zero on fresh sealed data. No further sealed
confirmations (§6.5/§6.7 terminality + §7.5 for any future sweep trigger).

**PART E OUTCOME (recorded 2026-07-17 — executed exactly as registered).** 18/18 trained,
integrity ALL PASS (both named overrides present, nothing else changed), judged 2026-07-17.
Audit: 25 BTC/5-min **2/6 valid** (base ref ~3/10 -> UNCHANGED at the primary setting);
25 BTC/2.5-min 4/6 (base 0/6 -> severity reduced at the extreme deadline); 5 BTC/2.5-min 4/6
(base 4/6 -> control healthy, variant interpretable). No cost trigger in any group. Pre-stated
interpretation applied: the collapse is NOT an update-rhythm artifact; the rhythm objection is
closed. Raw: `$S/runs_d3_*` + `$S/step5_d3_*`. Part E CLOSED.

## 8. MEASURED-SIGNAL EXTENSION — PHASE A MEASUREMENT PROTOCOL (registered 2026-07-22,
## BEFORE any measurement run; full plan: PLANS/measured_signal_extension_plan.md)

Design invariant (user directive 2026-07-22): the setup must give the agent a fair chance
at a real edge; restrictions exist only to prevent FALSE edges, never to attenuate real
ones. The confirmatory edge criterion for the eventual campaign is the UNCHANGED §3 rule.

**Candidates** (both computed causally on the 0.5 s reconstructed book, background
activity only):
- S1 trade-flow imbalance: signed market-order volume (buy-aggressor positive) summed over
  the trailing 1 s (two 0.5 s bins), normalised by its trailing rolling std (600-bin
  window, 120-bin warm-up, prior values only — strictly causal).
- S2 top-of-book depth imbalance: (bid1 − ask1)/(bid1 + ask1), same normalisation.

**Measurement:** per candidate × regime × split: pooled regression of forward mid return
(bps) on the signal, horizons {0.5, 1, 2, 5, 10, 30, 60} s. Split = the EXISTING step3g
chronological calibrate/holdout hour labels (verified chronological per regime,
calibrate max < holdout min; 278/93 calm, 279/93 volatile hours). DEVIATION NOTE vs the
plan draft (which said a new 70/30 split): reusing the already-registered split removes a
researcher degree of freedom; registered here before measuring.

**Held-out confirmation** (per candidate/regime/horizon): holdout p < 0.01, same sign as
calibrate, |holdout slope| ≥ 0.5 × |calibrate slope|.

**Selection rule (fixed now):** the candidate with the higher holdout R² at the 1 s
horizon in the volatile regime; ties within 10% relative go to S1 (the supervisor-named
candidate). Both candidates' full tables reported regardless.

**Stop/go bound:** clairvoyant schedule on holdout real mid paths — pace 2.0× when
(calibrate 1 s slope × signal) predicts a rise, else 0×, forced completion by the
300-decision (5-minute) deadline, impact ignored (both simplifications bias the bound UP,
i.e. toward proceeding). PROCEED if mean advantage vs TWAP ≥ 0.05 bps (the registered
materiality floor) in at least one regime AND that regime's 1 s held-out confirmation
passed. Otherwise STOP and record the measurement as a result. [USER APPROVED at Gate A0,
2026-07-22, verbatim "approve"; measurement launched immediately after.]

**Implementation:** `src/execution/qrm/signal_measure.py` (constants frozen in-module; no
command-line tunables); unit tests `tests/test_signal_measure.py` (14 tests: binning sign/
alignment, causality/no-future-leakage, warm-up, planted-slope recovery, bound feasibility
+ sign convention, determinism). Output: `$S/signal/measurement.json` with full tables +
provenance.

### §8 AMENDMENT 1 (2026-07-22, registered BEFORE the rerun; user-approved). The first
measurement run exposed two implementation defects, caught by internal cross-checks before
any decision was taken on its output: (i) the trailing-std normalisation divides by ~0+eps
on constant stretches, producing ~1e12 spikes that corrupt the statistics (the apparent
S2 selection was this artifact); (ii) the foresight bound is drift-contaminated (its calm
+0.44 / volatile -0.63 bps mirror the holdout drift, not signal value). Amendments:
(a) signal undefined (NaN) where the trailing std <= 1e-9, instead of dividing by epsilon;
(b) S2 (depth imbalance) used RAW - it is bounded and scale-free by construction, needing
no normalisation; (c) the bound is PLACEBO-CORRECTED: the identical rule is run with the
signal circularly shifted by one episode window (same statistics, no price alignment) and
the reported value is real minus placebo, per window; drift and deferral mechanics cancel.
The PROCEED threshold (0.05 bps) applies to the corrected value. First run's output
retained as measurement_v1_SUPERSEDED.json; never cite.

### §8 AMENDMENT 1 CORRECTION (2026-07-22, recorded on the v2 run's completion). The
amendment's diagnosis (ii) attributed the v1 bound's mirror-signed values to drift
contamination. The v2 placebo run shows the drift contribution is in fact ~0 (placebo
means +0.002 / -0.004 bps): the v1 pattern traced to defect (i) alone — the corrupted
normalisation produced garbage slopes whose SIGN flipped between regimes, inverting the
decision rule in volatile. The placebo correction is retained permanently as the
robustness control that makes this attribution checkable; the v1 numbers remain
superseded either way.

### §8 PHASE A2 SPEC (registered 2026-07-22, BEFORE the run; user approved "Proceed with
phase A2"). Measure the CURRENT simulator's endogenous S2-to-return relationship (the
queue-empty channel), to compute the injection residual.
- Environment: the unmodified current env per regime (`qrm_bundle_{regime}_b.npz` +
  `move_process_{regime}_centered.npz`), background only — no agent, no benchmark trades.
- Sim S2 (mirrors the real definition): at each 0.5 s interval, the first non-empty level
  per side, sizes converted to BTC via the bundle's per-level unit sizes:
  (BTC_bid − BTC_ask)/(BTC_bid + BTC_ask); undefined (NaN) while a side is swept.
- Sampling: 1,200 episodes per regime × 600 intervals (post-warm-up) ≈ 720k points per
  regime (≈ the real holdout n). Seeds 30,000,000+i — a DIAGNOSTIC-ONLY range, disjoint
  from every evaluation block (monitor 1e6, dev 5e6, reserve 6e6, sealed 9e6/13e6, and the
  campaign's reserved 17/18/19e6). Deterministic.
- Statistic: identical regression (forward mid return, bps, on raw S2) at the same seven
  horizons, per regime, within-episode only (no cross-episode returns).
- RESIDUAL (the quantity Phase B injects) = real CALIBRATE-split slope − endogenous slope,
  per horizon per regime. STOP RULE: if the endogenous slope exceeds the real slope at the
  1 s horizon in either regime, stop and report (injection would need to be negative —
  outside the registered design).
- Output: `$S/signal/endogenous_baseline.json` (full tables + residuals + provenance).

### §8 PHASE B REGISTRATION (2026-07-22, BEFORE any environment change; user approved
"proceed" on the presented step plan). Implementation-level realisation of the registered
design, fixed now:
- SHADOW-BACKGROUND SIGNAL: at episode reset (injection ON), the environment first
  simulates the background-only episode (same seed, no agent), with the injection applied
  self-consistently to its own evolution, recording S2_bg per 0.5 s interval. The actual
  episode then applies base + injected moves from that pre-computed path. Consequences:
  the signal is policy-independent (identical for agent, both TWAPs, and the follower on a
  shared seed — CRN preserved exactly) and the self-signal trap is closed by construction.
  Cost ~2x per episode, accepted (rigor over runtime).
- INJECTION TERM per 0.5 s interval: delta_bps = R_0.5s(regime) x (S2_bg − MEAN(regime)),
  with the A2 residual slopes R_0.5s = +0.02846 (calm) / +0.16904 (volatile) bps per unit,
  converted bps -> price ticks with a DETERMINISTIC fractional-carry accumulator (no new
  randomness). Longer-horizon structure is carried by the signal's own persistence and
  verified at the Phase C injection-matching gate (±20%, horizons 1–10 s); a decay kernel
  is the pre-named fallback if the gate fails.
- DEMEANING CONSTANTS (measured 2026-07-22, 300 background episodes x 600 intervals per
  regime, seeds 30,100,000+, `$S/signal/demeaning_constants.json`):
  MEAN(calm) = +0.110767, MEAN(volatile) = +0.069932. Fixed constants; guarantee the
  injected term has ~zero unconditional mean (drift-trap defence; verified empirically at
  Phase C fairness gate).
- OBSERVATION: S2_bg of the current interval appended to the observation (flag-gated);
  follower benchmark maps 1 + S2_bg to the nearest discrete action (slope sign positive
  per Phase A; nothing tuned), same completion semantics as adaptive TWAP.
- IDENTITY ORACLES: pre-change golden traces recorded (2 regimes x 2 seeds, 300 steps,
  observation-stream SHA-256 + reward sums): `$S/signal/golden_prechange/*.json`. The
  flag-OFF environment must reproduce them byte-exactly after the change.

### §8 AMENDMENT 2 (2026-07-22, registered BEFORE execution; user approved "Amendment 2
approved"). Phase C verdict: G1'/G2'/G3 PASS with injection ON; FAIRNESS FAIL (drift
+2.4/+7.4 ticks/ep, t=5.3/6.2; calm pace-1.2 material+significant −0.049 bps, t=−2.6) and
INJECTION-MATCHING FAIL (sim total 27–65% below the real curve, gap growing with horizon).
Diagnosis: (i) the demeaning constant was measured in the uninjected world; injection
feedback shifts the imbalance distribution, leaving a positive remainder = drift (the
drift trap firing, caught by the gate as designed); (ii) the simulator's book imbalance is
far less persistent than the real book's (real slope curve compounds ~14x from 0.5 s to
60 s; sim plateaus ~2.5x), so the instantaneous-residual injection cannot build the
long-horizon effect; integer-tick quantisation adds attenuation. Amendment (the fallback
pre-named in the Phase B registration):
(a) PERSISTENCE-MATCHED DRIVER: the injection is driven by an exponentially-smoothed
    background imbalance e_t = EMA_halflife(S2_bg) (initialised at the registered mean;
    NaN intervals leave e unchanged). e_t becomes the stored signal path, the observation
    feature, and the follower's input (it is the quantity that predicts future moves).
(b) DETERMINISTIC CALIBRATION (no outcome tuning; target = the REAL measured curve):
    half-life grid {1, 2, 5, 10, 20} s. Per regime per half-life: probe run (300
    background episodes, seeds 30,200,000+) at probe gain = the Amendment-1 residual;
    gain solved linearly so the 1 s total slope equals the real calibrate slope;
    refinement run at the solved gain with the probe run's measured E[e] as the demeaning
    constant (fixed-point iteration 1). Selection: the half-life minimising the maximum
    relative gap over the gated horizons {1,2,5,10} s. Parameters then FROZEN and logged.
(c) Re-run the FULL Phase C suite at unchanged bands; stop again on any failure.

### §8 AMENDMENT 3 (2026-07-23, registered BEFORE execution; user approved after the
Amendment-2 calibration outcome). Supersedes the Amendment-2 two-EMA proposal BEFORE any
Phase C re-run. Outcome being amended: the single-EMA grid calibration fit calm within the
band (max gated gap 19.3%) but volatile only to 24.6% (5 s horizon) — a single timescale
cannot represent the volatile curve's fast-rise-then-plateau shape
(`$S/signal/kernel_calibration.json`, superseded as a selection but retained).
DERIVED MULTI-TIMESCALE KERNEL (no search; zero free choices once registered):
(a) MEASURE the simulator's instantaneous-S2 autocorrelation rho(k), k = 0..120 intervals
    (600 background episodes per regime, injection OFF, seeds 30,300,000+).
(b) SOLVE, in closed form, the least-squares gains g_c over an exponential basis with
    half-lives {0.5, 2, 8, 32} s such that the implied injected response matches the
    RESIDUAL curve (real calibrate slope minus endogenous slope) at ALL SEVEN measured
    horizons; the response matrix is built analytically from rho (linear-response model).
(c) ONE empirical refinement iteration (registered max two): run 300 injected episodes,
    measure the achieved total curve, correct the gains by solving the same linear system
    on the achieved-vs-target residual; simultaneously fix-point the per-component
    demeaning means and measure the composite driver's std (the observation/follower
    normalisation constant). All constants then FROZEN.
(d) The observation feature and the follower input become the COMPOSITE driver (the
    injected-move predictor) in units of its own measured std; follower mapping unchanged
    (nearest action to clip(1 + signal, 0, 2)).
(e) Full Phase C suite at UNCHANGED bands. If the volatile 5 s horizon still cannot be
    fit, the band is NOT widened: the best-achievable match is reported to the user as a
    disclosed structural limitation for an explicit proceed/stop decision.

### §8 AMENDMENT 3 CORRECTION 1 (2026-07-23, implementation defect, registered before the
re-run). Phase C v2: G1'/G2'/G3 PASS; VOLATILE MATCHING PASS (gated gaps 11-19% — the
derived kernel works); fairness CATASTROPHIC FAIL both regimes (drift −841/−662 ticks/ep).
Root cause (evidenced by the refinement history): the driver statistics were read from the
measurement helper's instantaneous-imbalance output instead of the composite driver, so the
offset update absorbed E[imbalance] (~0.12, dimensionless) as though it were a bps bias,
injecting a constant negative price trend. Slope measurements regress on the instantaneous
imbalance BY DESIGN (mirroring the real-data measurement), so all gains and matching
verdicts are unaffected; only offset and obs_norm were poisoned. Correction (implementation
only, design unchanged): (i) driver statistics read from the environment's stored composite
path; (ii) gains iterations run with offset fixed at zero (means-only demeaning; the
Amendment-1 evidence bounds the resulting calibration-time drift at a few ticks/episode);
(iii) after freezing gains, a dedicated two-round offset fixed point (150 episodes each);
(iv) final verification run. Phase C v2 gates JSON quarantined as *_v2_FAILED.

### §8 AMENDMENT 3 CORRECTION 2 (2026-07-23, registered BEFORE the re-run; user approved
"ok proceed" on the presented five-step plan). Phase C v3 outcome being amended: G1'/G2'/G3
PASS; fairness GRADIENT PASS both regimes (no constant-pace policy holds a material and
significant advantage — the exploitability requirement); but (a) residual drift +3.3
(t=6.56) / +4.3 (t=2.89) ticks/ep fails the |t|<2 clause while being economically small,
and (b) matching fits 0.5–2 s (9–18%) but misses 5 s (22/23%) and 10 s (30/31%) — the
registered MAX_REFINE_ITERS=2 cap stopped the gains fixed point before convergence (the
iteration history shows monotone gap shrinkage), and the offset nulls the injected term's
MEAN but not the second-order drift arising from the book's asymmetric response to
zero-mean nudges. Corrections (each a calibration against an externally measured quantity;
no experiment outcome enters any target; bands on the matching gate UNCHANGED):
(a) GAINS ITERATIONS TO CONVERGENCE: cap raised 2 -> 5, each correction applied at half
    strength (damping 0.5, guards the feedback loop against overshoot oscillation), early
    stop once all gated horizons {1,2,5,10} s are within 15% relative. Seeds unchanged
    (30,400,000+).
(b) DRIFT-NULLING OFFSET: after freezing gains, the offset is initialised at the
    mean-injected-term fixed point (two rounds, as in Correction 1) and then solved
    against the MEASURED END-TO-END DRIFT itself: do-nothing episodes on the injected
    environment, Newton iteration offset += mean_drift_bps / 600 (600 injected intervals
    per episode, response slope ~= 1), max 4 rounds, n = 8,000 episodes per round,
    calibration seeds 30,550,000+ (diagnostic range, DISJOINT from the fairness
    certification block 3,000,000 — the nulling never sees the seeds it is later judged
    on). Stopping rule IDENTICAL to the registered base-environment neutralisation
    (step3g cmd_neutralise): |mean| < 0.5 ticks/ep OR |mean| < 1.5 x SE. This is the same
    method, tolerance and acceptance the base move process was certified with.
(c) FRESH-SEED FINAL VERIFICATION: the kernel-solution verification pass moves to seeds
    30,500,000+ (never used during any calibration iteration), so the frozen kernel's
    certified matching numbers cannot be flattered by tuning to the calibration episodes.
(d) DRIFT CRITERION RESTATED (fairness gate, drift clause only; gradient clause
    unchanged and always required): PASS iff |t| < 2 (statistical zero, the original
    clause) OR |mean drift| <= 0.5 ticks/ep (the identical magnitude tolerance the base
    environment's neutralisation was accepted at; ~0.05 bps of price over a full episode,
    of which a constant-pace policy can capture only a fraction, bounded below the
    0.02 bps gradient-materiality floor — and the gradient test verifies exploitability
    directly and independently). Rationale, recorded before the re-run: with n = 8,000
    episodes a t-test detects drift far below economic relevance; a pure significance
    clause is an impossible bar in the infinite-power limit and would reward
    under-powering. BOTH the t-statistic and the magnitude are computed and reported in
    the gates JSON and the write-up regardless of which clause passes.
(e) Full Phase C suite re-run (v4) with the frozen corrected kernel. If after CONVERGED
    iterations the 5 s or 10 s horizon still cannot reach the ±20% band, the band is NOT
    widened: the best-achievable curve goes to the user as a disclosed structural
    limitation (the simulator's book decorrelates faster than the real book) for an
    explicit proceed/stop decision. Phase C v3 gates JSON quarantined as *_v3_FAILED.

### §8 AMENDMENT 3 CORRECTION 2a (2026-07-23, registered BEFORE the re-run; user approved
"Proceed" on the presented halt report). Correction-2 solve outcome: DRIFT-NULLING WORKS
(calm +1.60 -> +0.59 ticks/ep t=+1.1 after one Newton step; volatile -0.03 t=-0.02 at
entry — both pass both drift clauses); but the damped gains iterations hit the 5-round cap
while STILL DESCENDING (calm 50->44->41->32->34 in-loop, 27.7% verified; volatile
50->42->37->36->30 in-loop, 27.8% verified; the undamped v3 run had reached 11-19% gated
in volatile), so the fixed point was not reached and no structural conclusion is valid
yet. The chained v4 gate suite was halted seconds after launch (no gates JSON written)
rather than certify an under-converged kernel. Correction (one registered constant + one
registered stopping rule; damping, bands, seeds, and all other Correction-2 machinery
unchanged):
(a) MAX_REFINE_ITERS raised 5 -> 12.
(b) PLATEAU STOP (data-defined convergence): stop the gains iterations early when the
    max gated gap improves by less than 1 percentage point over two consecutive
    iterations (in addition to the existing 15% early stop).
(c) Re-solve, then the full Phase C suite (v4). The structural-limit clause (e) above is
    unchanged and applies to the converged outcome.

### §8 AMENDMENT 3 CORRECTION 2b (2026-07-23, registered BEFORE the re-run; user approved
"approve"). Correction-2a solve + Phase C v4 gate outcome: MATCHING PASS both regimes
(calm 7/8/10/16%, volatile 10/1/6/12% at 1/2/5/10 s — the long-horizon "structural limit"
fully dissolved, confirming it was under-iteration); G1'/G2'/G3 PASS; gradient PASS. But
DRIFT FAILS both regimes on the certification block: calm 1.54 ticks/ep (t=2.71),
volatile 4.88 ticks/ep (t=2.94) — failing BOTH the |t|<2 and |mean|<=0.5-tick clauses.
Diagnosis (correcting a factor-of-ten error in the v3 note: 1 tick ~ 0.1 bps, so these are
0.15 / 0.49 bps, NOT the "0.03-0.04 bps" claimed there — the drift is economically
non-trivial, ~3x / ~10x the 0.05 bps materiality floor): the Correction-2 drift-nulling
was NOISE-LIMITED. Absolute-drift measurement over 8,000 do-nothing episodes has
SE ~ 0.5 (calm) / 1.66 (volatile) ticks — larger than the 0.5-tick tolerance — so the
base-env stopping rule's noise clause (|mean| < 1.5 x SE) fired before the magnitude
clause was met; neither regime was actually nulled to |mean| < 0.5 (calm stopped at
+0.69, volatile at +1.66, both on the noise clause). The certification block reveals the
true residual drift. This drift is large enough that the volatile front-loading paces
(pace 2.0 = -0.088 bps vs TWAP, t=-1.54) sit above the 0.05 bps materiality floor though
below the -2.5 significance threshold — i.e. it could manufacture a FALSE front-loading
edge, which the fairness invariant forbids. Correction (a measurement-variance fix; the
estimand, the criterion, the bands, the gains and all seeds are UNCHANGED — no goalpost
moves):
(a) CRN-PAIRED DRIFT MEASUREMENT. Each seed runs the do-nothing episode twice sharing ALL
    randomness (identical re-seed; same pre-drawn move path and engine RNG): once with the
    injection applied, once with the injected price path zeroed (everything else, incl. the
    shadow-computed observation signal, identical). The paired difference D_on - D_off
    cancels the common background random walk and isolates the injection's drift
    contribution. Because the base move process is separately drift-neutralised
    (E[D_off] ~ 0, certified in step3g cmd_neutralise), the paired mean is a low-variance,
    unbiased control-variate estimate of the total drift the agent faces. Applied in BOTH
    the kernel drift-nulling (sigext_kernel.measure_drift) and the fairness gate
    (sigext_gates._measure_bg_drift_paired). Raw unpaired on/off means also reported.
(b) DRIFT_NULL_ITERS raised 4 -> 8 (the paired estimate converges for real rather than
    stopping early on noise). Stopping rule and the |t|<2 OR |mean|<=0.5-tick certification
    criterion UNCHANGED — the fix only makes the same quantity measurable precisely enough
    that the magnitude clause governs. Gains (matching), bands, and all other machinery
    frozen from the Correction-2a solve.
(c) EARLY CHECKPOINT: after the re-solve the paired nulled drift is inspected before the
    ~3 h gate suite; if pairing does not reduce the SE enough to null below 0.5 ticks, that
    is reported to the user (a genuine limit of the injection method) rather than certified.
(d) Full Phase C suite re-run (v4b). Correction-2a v4 gates JSON quarantined as
    *_v4_FAILED; kernel_solution.json (2a) retained (gains unchanged; only the offset
    re-solves).

### §8 AMENDMENT 3 CORRECTION 2c (2026-07-24, registered BEFORE the re-run; user approved
"yes proceed" on the presented decomposition + fix). Two empirical findings supersede the
2b plan: (i) the CRN pairing gave only ~30% variance reduction (smoke test: correlation
~0.70 — the engine RNG desyncs once the injection perturbs the book), so it cannot resolve
the drift below the noise floor; (ii) a base-vs-injection DECOMPOSITION on the certification
block (n=8000 each, 2026-07-24, `$S/signal/drift_decomp.log`) shows the drift is entirely
the injection's, and it is LARGE and SYSTEMATIC, not lost in noise:
  BASE (inj off):  calm -0.81 (t=-1.86), volatile -2.22 (t=-1.98) — both within the base
    env's own |t|<2 certification; the base env is exonerated and the completed campaigns
    are unaffected.
  INJECTED:        calm +1.54 (t=+2.71), volatile +4.88 (t=+2.94).
  INJECTION marginal: calm +2.35, volatile +7.10 ticks.
Diagnosis: the offset-solve UNDER-SOLVED because its stopping rule quit early on the noise
clause (|mean| < 1.5 SE), applying only a fraction of the correction. The cancellation
mechanism is sound and EXACTLY linear with a known slope (1 bps of per-interval offset
shifts p_ref by INTERVALS_PER_EP=600 intervals x price/tick), so it was simply fed a lazy
stop. Correction (the estimand -- the TOTAL drift the agent faces -- and the |t|<2 OR
|mean|<=0.5-tick criterion are UNCHANGED; gains/matching/bands frozen from 2a):
(a) UNPAIRED TOTAL-DRIFT measure (2b's pairing reverted); precision from episode count.
(b) Offset solved by the EXACT linear correction offset += mean_bps / INTERVALS_PER_EP over
    a FIXED 2 rounds at DRIFT_NULL_EPS = 24,000 (volatile SE ~0.95 ticks), NO early-stop.
    Solved on the calibration block (DRIFT_SEED_BASE = 30,550,000), INDEPENDENT of the
    3,000,000 fairness certification block.
(c) HONEST SCOPE NOTE (recorded before the run): the |mean| <= 0.5-tick magnitude clause is,
    for volatile, below the resolution of any feasible episode count (noise floor ~1.6 ticks
    at n=8000; ~0.5 needs ~350k episodes). Certification therefore rests on the OTHER
    registered pass condition -- |t| < 2 (drift statistically indistinguishable from zero
    after a properly-centred offset) AND the pace-gradient exploitability clause (the direct
    test that no constant-pace timing strategy profits) -- exactly the standard the base env
    and every completed campaign passed. The raw magnitude is reported regardless.
(d) THE OPEN TEST: an offset solved on block 30.55M is certified on block 3e6. If it
    transfers (drift within |t|<2 on 3e6) -> certified. If it does NOT (drift block-
    dependent) -> reported to the user with the numbers for a proceed/stop decision; the
    band and criterion are NOT weakened to force a pass.
(e) Full Phase C suite re-run (v4c). Prior gates JSON already quarantined *_v4_FAILED.

### §8 CORRECTION 2c REVISED (2026-07-25, registered BEFORE the re-run; user approved
"ok proceed" then "proceed"). Two empirical refinements found during the 2c solve, both
by verification catching the defect before certification:
(i) THE SLOPE WAS WRONG. The fixed-step correction assumed 1 bps of offset shifts p_ref by
    INTERVALS_PER_EP*price/tick (~6000 ticks/bps); the injected move is attenuated (carry
    accumulator + book response) so the TRUE slope is ~-4115 (volatile) / ~-3520 (calm)
    ticks/bps. The fixed step therefore under-cancelled ~40%/round and 2 rounds could not
    null volatile (caught live: calm drift dropped only +3.99 -> +1.67 per round). FIX:
    solve the offset by an EXACT 2-POINT slope fit (measure drift at offset 0 and at a probe
    offset, fit the line, solve the zero-crossing) instead of a fixed step. The mean-offset
    pre-phase is dropped (it introduced drift the drift-null then had to remove).
(ii) SINGLE-BLOCK ANCHOR UNDER-CORRECTS. The raw creep varies block-to-block by a few ticks
    (volatile ~24-28); an offset solved to null ONE block leaves ~+5 ticks on the others
    (caught by the fresh-block verify: +4.86, t=2.02). FIX: anchor the offset to the POOLED
    raw drift across independent blocks (ANCHOR_SEED_BASES = 30.55M, 31.0M; n=12000 each),
    so per-block residuals scatter around zero. The blocks EXCLUDE the fairness cert block
    (3e6) and the verify block (30.5M) -> no tuning on the judged seeds.
Estimand (total drift the agent faces), criterion (|t|<2 OR |mean|<=0.5 tick + the
exploitability gate), bands, gains and matching are ALL unchanged; only the offset-solving
method improves. Full suite re-run (v4c). If the cert block still fails |t|<2 after a
pooled-anchored offset, the registered proceed/stop decision goes to the user; no band moves.
