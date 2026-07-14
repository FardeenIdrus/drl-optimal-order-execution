# WRITE-UP ARGUMENT BANK — dissertation-bound reasoning, full detail (started 2026-07-11)

Purpose: the durable home for every INTERPRETIVE argument the dissertation will need — scope
statements, defenses, contribution framing, limitations reasoning — with the supporting numbers
and where each number lives. Results/tables stay in `qrm_step5_remediation.md`; rules/protocol
history stay in `qrm_step4_criteria.md`; THIS file holds the reasoning that turns those into a
write-up. Every claim here cites its evidence source. Add to it at every major finding.

---

## A. THE SIGNAL STRUCTURE OF THE ENVIRONMENT — what is exploitable, what is absent, and why
## (the scope statement; written 2026-07-11 after the confirmation null)

**What the environment CONTAINS (order-flow theory, mechanically embodied).** The queue-reactive
model's premise is that order-flow state drives short-term dynamics: arrival, cancellation, and
trade intensities are calibrated CONDITIONAL ON QUEUE STATES (per regime, from real BTC L4 data).
Consequently thin queues genuinely raise the probability of imminent price moves; deep books
genuinely predict cheaper fills. The agent OBSERVES this state: 20 queue sizes, spread, trailing
trade volume, signed buy/sell imbalance, own fills. On top sits a VERIFIED impact channel (gate
G1', criteria §3b): executing the order raises the cost of an identical follow-on by 1.8x
(volatile) to 2.4x (calm), impact is size-monotone, and the book measurably refills (recovery
0.347->0.124 bps calm, 0.147->0.076 volatile). So order-flow signals and liquidity-timing
opportunities EXIST and are OBSERVABLE. The null is not "the agent found nothing because nothing
was there."

**What the environment LACKS, quantified.** The exogenous price component is drawn i.i.d. per
0.5s interval from a distribution fitted to real mid-price changes (drift-neutralised per the
fairness gate). There is no background momentum or trend to ride. The remaining queue-driven
(predictable-in-principle) share of mid-price variance was MEASURED during remediation R3:
~11.5% in calm, ~3.7% in volatile (endogenous-move deconvolution, logged in
`qrm_step5_remediation.md` R3 status line). So the exploitable structure is: fills, liquidity
timing, impact management, and a small endogenous price component — NOT price prediction.

**Why the i.i.d. choice was right (pre-empting "your null is built in").** The exogenous
component is, in the calibration data, whatever December's mid-moves were; to give it
PREDICTABILITY conditional on observables we would have had to FABRICATE a signal (choose its
strength, form, and horizon ourselves). Any positive result would then be circular: the agent
finding the signal we planted. The conservative choice — inject nothing we cannot measure — makes
a positive finding meaningful at the cost of capping the ceiling. That trade-off is disclosed,
not hidden.

**The honest scope sentence for the write-up:** this study tests whether DRL can exploit
LIQUIDITY-TIMING and IMPACT-MANAGEMENT information (the order-flow channel) for execution, in a
world with realistic, calibrated order-flow dynamics and a genuine impact channel but no
fabricated price predictability. It does NOT test whether RL can exploit strong real-world
price-predictive signals; that question requires a different design (see F, future work).

---

## B. WHY THE NULL IS NOT VACUOUS ("obviously null by construction" — refuted, three arguments)

1. **The cost structure is demonstrably exploitable in principle.** Instant dumping costs 5-7x
   TWAP (G3 gate: calm 0.441 vs 0.059 bps; volatile 0.244 vs 0.052). Fill costs vary strongly
   with book state; self-impact is real and recoverable (G1' numbers above). Whether a learned
   policy could time child orders into deep-book moments and save >= 0.05 bps vs TWAP was a
   GENUINELY OPEN quantitative question. Nothing in the construction forces "no."
2. **The pipeline demonstrably rewards real signals when they exist — the drift episode is the
   positive control.** When the environment accidentally contained a genuine exploitable pattern
   (the residual upward drift, pre-fix), the agents found it IMMEDIATELY and harvested it
   (front-loading; the entire spurious calm "edge," `step5_v2` vs `step5_v3` comparison). The
   machinery detects and rewards edges when present; there simply is not a material honest one.
   A design that "cannot produce positives" would not have produced that positive.
3. **The result is a statement about MAGNITUDES, not existence.** Order-flow information at the
   empirically calibrated magnitudes is not worth a material execution saving (>=0.05 bps) over
   TWAP at 25 BTC / 5 min. That is an informative empirical finding about how much the
   order-flow channel is worth for execution scheduling — not an artifact of removing signals.

---

## C. TESTING-SOUNDNESS DEFENSES (assembled 2026-07-11 after the user challenged the methodology;
## each verified, with evidence locations)

1. **The agent's action set CONTAINS the benchmark.** Action 1.0x = exactly TWAP pace every
   step. Under CRN pairing, an agent playing 1.0x always would measure a difference of exactly
   zero. The test therefore cannot structurally suppress an agent below the benchmark; the
   strategy space includes the benchmark as a member. A "false null from unfair testing" would
   require penalising identical behaviour on identical markets — impossible under paired scoring.
2. **From-scratch reproduction of the scorer.** A recorded confirmation number (seed-5 vs
   adaptive, sealed block) was recomputed with library primitives, bypassing the judge script:
   +0.0222 vs recorded +0.0222, exact match (criteria §6.9; live doc CURRENT RESULTS (0)).
3. **Adversarial audit history, with bugs in BOTH directions.** Three audits (3-agent review
   2026-07-07; 5-agent pre-launch audit 2026-07-08; post-verdict audit 2026-07-11). Bugs found
   included both anti-agent effects (deadline over-buy penalising paced policies, I2) and
   pro-agent artifacts (the drift confound rewarding front-loading, CB1). Both were fixed; the
   pipeline was not tuned toward null.
4. **Power was adequate.** The §6 confirmation had ~80% power at the dev-observed effect size
   (criteria §6.7 M2). Its CI [-0.0217, +0.0171] EXCLUDES the dev-block effect (-0.063): the
   effect was affirmatively absent, not undetected.
5. **Internal consistency.** vs_fixed and vs_adaptive agree to <=0.0006 bps everywhere; audit
   runs before costs by construction; 219 unit tests green throughout.

---

## D. THE ANATOMY OF THE SPURIOUS EDGE (the dissertation's most original empirical material)

The project caught and dissected THREE distinct mechanisms that manufacture fake RL execution
edges — each demonstrated with data, not asserted:

1. **Environmental drift (the artifact channel).** An uncentred background drift mechanically
   rewards front-loading; agents harvest it and look skilled. Demonstrated: the calm "edge"
   existed pre-fix (pooled -0.019, p=0.025, `SUPERSEDED_step5_v2`) and DISSOLVED when the drift
   was neutralised (-0.006, p=0.18, `step5_v3`). Fix + permanent fairness gate documented in
   `qrm_prelaunch_audit_2026-07-08.md`.
2. **Shared-block luck (correlated evidence masquerading as robustness).** All 14 tuning
   configurations were evaluated on the SAME 2,000 dev markets; "7 of 8 variants confirm the
   edge" was ONE lucky block measured repeatedly, not replication. Demonstrated by the
   three-block diagnostic (criteria §6.9): every config flips sign across evaluation blocks
   (1e6 monitor: +0.06..+0.15 WORSE than TWAP; 5e6 dev: -0.03..-0.06 better; 9e6 sealed: ~0).
   Design lesson for the field: across-SEED significance on a shared evaluation block is NOT
   evidence of generalisation across MARKETS; the evaluation block is a sampling unit too.
3. **Winner's curse (selection on noise).** Selecting the best of 14 noisy measurements
   preferentially selects positive noise. Demonstrated twice at different scales: v3b led at 3
   seeds (-0.084) and fell to -0.060 at 5 seeds; the selected champion (-0.063 dev) fell to
   -0.002 on the sealed one-shot. The 3-seed->5-seed re-ranking was the small-scale preview of
   the dev->sealed collapse.

Combined narrative sentence: "a naive version of this study would have reported a significant,
tuning-robust, regime-specific DRL execution edge; a pre-registered sealed replication shows
that edge to be the sum of an environmental artifact, shared-block luck, and selection bias."

---

## E. TWO-TRACK COMPLEMENTARITY (L2 replay + QRM reactive — why together they are stronger)

The two simulator families bracket the fundamental dilemma of execution-RL evaluation:
- **L2 historical replay** (earlier phase): REAL signals present (real order flow, real price
  paths) but a FROZEN book — no impact channel; the agent's trades do not move the market.
  Result: null across the three-axis sweep (sizes, horizons, cadences).
- **QRM reactive** (this phase): GENUINE impact channel (order flow consumed, prices move,
  books refill) but muted exogenous predictability (i.i.d. background). Result: null, confirmed
  out-of-sample after full tuning.
Neither world alone closes the question; together they cover both failure modes an RL edge
could exploit (signal-without-impact and impact-without-signal), and both return null. The
claim "DRL does not beat TWAP" is made ONLY within these two worlds — but the worlds were
chosen to be the two honest ones buildable from real data without fabricating alpha.

---

## F. FUTURE WORK THAT WOULD ACTUALLY ANSWER THE REMAINING QUESTION (name it precisely)

The one channel neither world tests: REAL price predictability + impact simultaneously. The
precise extension: measure the empirical order-flow-imbalance -> future-return relationship
directly from the L4 data (magnitude, horizon, decay), inject THAT measured relationship into
the QRM's exogenous process (a calibrated, data-grounded signal, not an invented one), and
re-run the identical pre-registered pipeline. If DRL cannot monetise even a measured signal
against TWAP, the null hardens; if it can, the edge finally has a mechanism. Second extension
(already documented as a limitation): maker/taker — passive execution changes the action space
entirely. Neither is scheduled; both are named so the scope boundary is explicit, not evasive.

---

## G. CONTRIBUTION FRAMING (what the dissertation claims, in one place)

1. Infrastructure: a calibrated, reaction-inclusive QRM execution environment built from real
   BTC per-order (L4) data, gate-validated (impact, cost-vs-size, benchmark sanity), fairness-
   gated (drift-neutral), and bug-hardened through three adversarial audits.
2. Method: a fully pre-registered evaluation pipeline for execution RL — frozen edge criteria,
   audit-before-costs, CRN pairing, sealed one-shot out-of-sample confirmation — with every
   deviation disclosed and dated (criteria §5c, §6.9).
3. Findings: (a) DQN collapses to degenerate policies where PPO does not (replicating the L2
   pathology in a reactive world); (b) PPO matches but does not beat TWAP, confirmed
   out-of-sample; (c) the anatomy of the spurious edge (D above) — drift artifact, shared-block
   luck, winner's curse — each demonstrated with data; (d) regime contrast: the apparent signal
   concentrated in the volatile regime and its attribution (RQ3, exploratory) explains what the
   agent actually learns there.
4. Scope honesty: the null is a property of two deliberately conservative worlds (E), with the
   remaining open channel named precisely (F).

---

## H. LIMITATIONS PROSE (ready-to-lift; no em dashes in the prose itself)

(1) **Market-order-only execution (maker/taker)** — full paragraph already drafted in
`qrm_step5_remediation.md` "WRITE-UP LIMITATIONS" section; keep single source there.
(2) **Exogenous i.i.d. background (signal ceiling)** — draft from section A above when writing;
the key sentences: "The background price component is resampled independently per interval from
a distribution fitted to the calibration data. This deliberately avoids fabricating predictive
structure, at the cost that the agent cannot be rewarded for price prediction, only for
liquidity timing and impact management. The reported null therefore concerns the value of
order-flow information for execution scheduling at calibrated magnitudes, and does not extend
to markets or simulators with exploitable price predictability."
(3) **Single asset, single month, two regimes** - BTC, December 2025 calibration; regime
definitions are within-month; generalisation across assets/periods untested.
(4) **Evaluation-block sensitivity** - demonstrated directly (D2); mitigated by the sealed
confirmation; residual lesson: report block-resampled uncertainty, not only within-block CIs.

---

## I. SELECTION-METRIC ANALYSIS — which ranking is "correct"? (logged 2026-07-12, user-prompted)

**The three defensible rankings and their winners (from `step5_selection_v3`, valid-only pooled):**
| ranking | question it answers | winner | winner's numbers |
|---|---|---|---|
| across both regimes (strict health, simple avg) | best all-weather policy | bigger-network 128 | vol -0.0562 + calm -0.0211 -> avg -0.0386 |
| across both regimes (valid-run-weighted) | same, weighting evidence | bigger-network 128 | -0.0430 |
| volatile only | best where the signal lives | faster-learning lr 1e-3 | -0.0628 (calm +0.0013) |
| calm only | best in quiet markets | no-exploration-bonus | -0.0213 (immaterial, under the 0.05 floor) |

**Analysis.** No ranking is universally correct; the metric must match the claim. For THIS project's
purpose the volatile-only ranking was arguably the coherent choice, because the confirmation was
pre-registered (§6.7 M1, before tuning ran) as SINGLE-REGIME VOLATILE: selecting on calm+volatile
performance for a volatile-only exam is a metric mismatch. The actual defect was that §5-rule-2
("across both regimes", written 2026-07-06 when confirmation was a two-regime concept) and §6.7
(volatile-only confirmation, 2026-07-09) came into unreconciled CONFLICT, and the selection
(2026-07-10) implicitly followed the newer logic without amending the older rule first — a process
failure (disclosed, §6.9), not a metric error per se. Additionally "across both regimes" is itself
underspecified (simple vs run-weighted vs worst-regime average; rankings shift with the choice) —
an undisclosed researcher degree of freedom inside the rule as written.

**The 90+ standard applied (and what we do):**
1. Report the full agent x regime matrix (done, live doc) + all three rankings (this table).
2. One pre-specified aggregation per decision, matched to the claim; reconcile conflicting
   pre-registered rules IN WRITING before acting on them.
3. NO sealed confirmation per ranking-winner (multiplicity): volatile-only winner tested (FAIL);
   across-both winner gets the ONE remedial test (user-scheduled, after sweeps); calm-only winner
   is REPORTED but NOT tested (immaterial magnitude -0.021 < 0.05 floor; calm excluded from
   confirmation a priori by §6.7; a third test = pure multiplicity).
4. Write-up angle: two defensible aggregations picking two different champions is a live
   demonstration of researcher degrees of freedom (Gelman's forking paths) — same family as the
   winner's curse (D3) and block-luck (D2) mechanisms this dissertation dissects. Present it as
   methodology-chapter content, alongside the §6.9 disclosure.

---

## D4 addition (2026-07-13): THE SIZE-RESPONSE ARGUMENT — fourth independent line against the dev edge

The §7 sweep (criteria §7; results `step5_sweep_*`, live doc CURRENT RESULTS (C)) gives a
mechanism-based falsification: if the 25-BTC dev-block edge (-0.063) had been genuine IMPACT
MANAGEMENT, it should scale with order size (larger order -> more impact -> more for a smart
scheduler to save). Measured volatile size-response: +0.043 (5 BTC), -0.016 (12.5), -0.063 (25,
dev), +0.002 (50) — non-monotone, with the "edge" existing ONLY at the exact size where all
development and selection occurred, and the agent actively WORSE than TWAP at 5 BTC (tiny order:
nothing to manage; deviating from even pacing only adds noise). Combined with (D1) drift, (D2)
shared-block luck, (D3) winner's curse, this completes four independent falsification lines. The
10-min horizon cell (-0.062 pooled, p=0.11, seed spread -0.10..+0.01) illustrates §7.5's value:
a nominally interesting pooled number correctly capped as non-evidence by the pre-registered
trigger. Null verdict: holds across sizes 5-50 BTC, both horizons, both regimes.

---

## C6 addition (2026-07-13): BEHAVIOUR-AUDIT DEFENSE — thresholds, rationale, and the
## over-conservatism check (data-grounded)

Thresholds (audit_one, 200 episodes, BEFORE any cost is computed): INVALID iff (a) deadline
residual > 1 unit (~0.55 BTC; the sub-unit tail is unbuyable by ANY policy incl. TWAP — counting
it flagged 20/20 agents on 2026-07-06, fixed) in > 10% of episodes, or (b) one action > 90% of
decisions AND that action is do-nothing (constant NON-zero pace legitimised by revision 4b — the
one part of the audit that WAS over-conservative and was fixed with logged rationale). Rationale
for (a): deadline reliance outsources execution to a mechanism outside the policy that is both
economically punishing (forced sweep costs 5-7x TWAP) and partly GENEROUSLY priced beyond-window
(so the rule also prevents exploiting that leniency).

Empirical over-conservatism check (all current campaigns, 147 audited runs, 19 flagged):
15/19 flags are GROSS (residual reliance 26-100% of episodes, 2.6-10x the cap; mostly DQN
collapse) — the threshold value is not doing the work for these. 4/19 are marginal (11-18%);
all four also show elevated do-nothing shares (24-39%); sensitivity checks (counting invalid
seeds anyway) never flipped any verdict; the survivorship guard prevents reduced-seed cells
from claiming edges. Direction-of-error: the audit can only EXCLUDE agents from claims, never
improve them, so a too-strict audit cannot manufacture the null headline (the excluded DQN runs
had catastrophic +0.4 bps costs — exclusion flattered the agent family, not the null).
Honest residuals for the limitations section: the 0.10 cap is a hard cliff on n=200 (±~0.02
sampling noise at the boundary; mitigated by reporting raw fractions + sensitivity); the rule's
justification is environment-specific (rational deferral could be optimal in an env with cheap
end-of-window liquidity — not this one).

---

## J. THE MEASURED-SIGNAL EXTENSION — history, rationale, design, and sequencing
## (logged 2026-07-13 from the scope discussion; supersedes the short note in F)

### J1. The factual history of the environment's information content (correct the record)
Nothing was REMOVED from the Queue-Reactive Model. The pure QRM moves price endogenously (a queue
empties -> price ticks), which is predictable-in-principle from the observable queues — an
information-carrying mechanism. Calibrated to BTC, that mechanism produced almost NO movement:
BTC's book is so deep the queues rarely empty, so the pure model was unrealistically static. The
background move process was therefore ADDED (sizes fitted to real BTC mid-changes) to reach
realistic volatility. The added component is drawn independently each interval — unpredictable by
construction. The queue-driven channel was KEPT and still contributes ~11.5% (calm) / ~3.7%
(volatile) of move variance; that share is small because of BTC's depth, not because it was
suppressed. Separately, the added component's upward mean (drift) was removed — correctly: an
unconditional drift is harvestable by a dumb constant-pace rule and is a bias, not information.

### J2. The 2x2 coverage map (what each experiment tests)
| world | real predictive signals? | market reacts to agent? | result |
|---|---|---|---|
| Track 1: frozen L2 replay | YES — fully real | no | null (3 axes) |
| Track 2: reactive QRM (this study) | mostly removed (small queue channel only) | YES — verified | null (2 sealed tests, all sizes/horizons) |
| EXTENSION (untested) | YES — the MEASURED real amount | YES | open |
The prediction channel was tested FIRST, in its most real form (Track 1). Track 2 answered the
reaction objection to Track 1. The untested cell is the combination — the world closest to the
real market. The extension is therefore the SAME research question under the last untested
condition, not a new question.

### J3. Why the original design was defensible + the honestly-conceded miss
Defensible: (i) it follows the standard published QRM structure; (ii) the only calibrations
available at design time were "fit the unconditional move distribution" (simple, honest) or
"invent a conditional signal strength" (circular — the agent finds treasure we planted); the
third option (MEASURE the conditional relationship) is a genuinely harder estimation problem;
(iii) the track's mission was the reaction channel, and prediction was already covered by Track 1.
THE MISS (state plainly in the write-up): when the background component was added, its scope
consequence — "most simulated price variance becomes unpredictable by construction, narrowing what
RQ1 can mean in this world" — was treated as a calibration detail and never surfaced as an explicit
design decision. A framing/communication failure at design time, not an execution error. The weeks
of runs were NOT invalidated or wasted: they answered the liquidity-timing/impact question (a real,
open question needing no prediction), caught the drift confound (which lived INSIDE the added
component and would have contaminated a more complex conditional design invisibly), and built the
exact machinery (deconvolution, fairness gate, sealed-test pipeline) the extension requires.

### J4. What the extension requires (the design, with its three traps)
1. MEASURE from the L4 data: order-flow imbalance on the 0.5s grid -> future mid-move relationship
   (magnitude, horizon, decay), per regime. Nothing invented.
2. INJECT into the simulator's background process with three traps avoided:
   - DRIFT trap: conditional mean may depend on book state; UNCONDITIONAL mean must stay ~0
     (fairness gate must re-pass) or the front-loading bias returns.
   - DOUBLE-COUNT trap: the queue channel already carries part of the measured relationship;
     inject only the RESIDUAL (measured total minus what the sim already generates) — same
     deconvolution discipline as the drift fix.
   - SELF-SIGNAL trap: condition on BACKGROUND flow only, never the agent's own trades — else the
     agent can manufacture its own signal (manipulation, not execution skill) and CRN pairing
     degrades.
3. STRUCTURAL env change: background moves are currently PRE-DRAWN at episode start (that is what
   makes them independent); a state-conditional signal must be sampled interval-by-interval from
   the live background state. Moderate change + tests + in-sim validation that the injected curve
   reproduces the measured one.
4. RE-VERIFY every gate on the modified env (fairness/drift, impact realism, audit thresholds),
   logged before any training.
5. REDUCED pre-registered campaign (not the full tuning odyssey): base PPO + the selected config,
   5 seeds, both regimes, dev block; pre-committed sealed confirmation on a fresh virgin block
   (e.g. 17,000,000) ONLY if a dev signal appears. All machinery exists.
6. Outcome handling (pre-commit): third null -> "even genuine, measured, real-market
   predictability at its actual strength is not monetisable by DRL against TWAP on this book" (a
   much stronger null); positive -> the edge finally has a data-grounded mechanism and RQ3
   attribution returns as the primary contribution. Effort: ~4-6 working days, compute in
   background.

### J5. Sequencing decision (2026-07-13) + the viva defence
User dates: submission 1 Sep; code freeze target 25 Jul; Carlo meeting ~17 Jul requiring a
populated Results section (figures) — so figures from EXISTING results start immediately and do
not wait for anything. Extension runs 15-21 Jul (near the FRONT of the queue, before AC/VWAP and
attribution) because it is the only remaining experiment that can change the headline, title, and
the labelling of the attribution work. RQ3 attribution deliberately AFTER the extension verdict.
Cadence sweep demoted to LAST (run only if the queue clears by ~23-24 Jul; else descoped with the
documented reason that decision frequency was already varied on the L2 track, 1-min vs 10-s).
Viva defence if asked "why wasn't the extension the original design": J3 verbatim — standard
model, measure-vs-invent dilemma resolved conservatively at design time, prediction channel
already covered by Track 1, gap articulated and then CLOSED (or: named precisely as future work,
if it ends up descoped).
