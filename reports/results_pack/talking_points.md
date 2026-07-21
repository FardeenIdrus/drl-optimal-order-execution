Meeting Prep 17/07/2026

# Meeting talking points (status update since last meeting)

Every section below says where the evidence lives in meeting_pack.pdf, so I can jump to it
if probed. The pack has seven numbered sections; I reference them as "pack S1" to "pack S7".

## 1. Quick recap of the direction we agreed last time

Last meeting the reactive simulator was built and validated, the drift confound had been
found and removed, and the first clean campaign had produced a formal null with a real
looking signal underneath: PPO in the volatile market was cheaper than the self-correcting
TWAP in all five runs, about 0.047 basis points on average, odds of chance about one in
160, but short of my pre-set bar. The agreed plan had two steps. First, tuning: retrain
with pre-planned variations to answer whether the null was just an under-tuned agent, and
to pick the single best configuration. Second, confirmation: test that best configuration
exactly once on a sealed set of runs that played no part in finding the signal. Everything
below is what happened when I executed that plan, and what I built on top of the answer.

## 2. Everything I have run since last meeting, in order (the full ledger)

I have run a lot, so here is the complete list in the order it happened, with one line on
what each stage found. Every stage below has its own section later in this script.

1. Finished the tuning and selection: 98 runs in total. Picked a winner.
2. First sealed confirmation of the winner: 5 fresh agents, tested once. Failed.
3. Second sealed confirmation of the runner-up: 5 fresh agents, tested once. Failed.
4. Robustness sweeps: 24 runs at other order sizes and a longer deadline. Null everywhere.
5. The full grid: 66 runs covering every combination of four order sizes and four
   deadlines, both markets. Null in 20 of 22 new groups, two suspicious groups.
6. The follow-up ladder on those two groups: 4 more training runs, then a re-test of all
   ten agents on a reserve set never used before. Both groups died. No sealed test spent.
7. A DQN probe: 18 runs mapping the DQN failure across three more settings. It is
   systematic and driven by order size.
8. A second DQN experiment: 18 runs with the standard library training rhythm, to rule out
   my configuration as the cause. Ruled out.
9. A diagnostic reading of the trained DQN networks themselves. Found the mechanism.
10. The per-episode replay of all 20 primary agents to get full cost distributions, the
    thing you asked for. Done, with an exact integrity check.
11. On the old frozen-replay track: trained 19 missing agents plus 4 replacements for
    corrupt legacy runs, so all 70 agents now exist with complete validation results. The
    one-shot test on that track's untouched data is built next and not yet spent.

That is 247 new training runs plus two sealed tests and one reserve-set test since we last
spoke. The rest of this script walks through what each stage means.

## 3. The tuning, and then the confirmations, which are the headline (pack S2)

What the tuning was. Twelve pre-planned variations of the agent: bigger networks, stronger
reward signal, faster and slower learning rates, more and less exploration, longer memory,
two fixes aimed at the DQN laziness, and longer training at ten million steps. With
escalations that came to 98 runs. The point was to answer the obvious challenge, that the
null was just an under-tuned agent, and to pick one best configuration for the sealed test.

What the tuning found. In the volatile market, seven of the eight healthy variations
reproduced the small advantage. At the time this looked like strong corroboration that
something real was there. The winner was the faster learning rate: 0.063 basis points
cheaper on average across five seeds, odds of chance about one in 230. In the calm market,
nothing was consistent. The ten-million-step runs also killed the "it just needed longer
training" explanation. (Pack S2: the two forest plots, one per market, plus the group
table and the full 98-run appendix table.)

What a sealed confirmation is. A block of 2,000 simulated runs that has never been looked
at by me or touched by any decision. The winner is retrained on five completely fresh
seeds, tested on that block exactly once, with the pass rule fixed in advance, and there
are no second attempts whatever the result.

What happened. The winner returned essentially zero: 0.002 basis points, odds of chance 38
percent. Because there was a defensible ambiguity in how the winner should have been picked
(rank on the volatile market only, or on both markets), I also tested the second candidate,
a bigger network, on its own separate sealed block. Also essentially zero. So the question
of which selection rule was right turned out not to matter: both answers fail.

The conclusion, stated plainly. The signal that was consistent across five seeds and twelve
variations does not exist on fresh data. The headline of the dissertation is a boundary
null, established twice, under rules I fixed before running anything. And in hindsight the
illusion is visible: the very same configurations read WORSE than TWAP on a third,
independent monitoring set (pack S2, the three-block figure). Agreement across seeds could
not catch this, because every seed was graded on the same 2,000 runs. Only fresh data
catches it.

## 4. How the two TWAP benchmarks work, precisely (likely question; pack S1)

I compare against two versions of TWAP, and the pass rules require beating both.

Fixed TWAP, the textbook version:
- Take the order, divide by the number of decisions, trade that same slice every second,
  regardless of what happens.

Adaptive TWAP, the harder version and my headline benchmark, step by step:
1. At each second, look at how much of the order is still unfilled.
2. Look at how many seconds remain until the deadline.
3. Divide one by the other: that is the pace that finishes exactly on schedule from here.
4. Trade at that pace this second, then repeat the calculation next second.
- The effect: if a fill comes up short, the next second's slice grows automatically. It
  self-corrects and always finishes.

Why adaptive is the headline. First, it is literally the agent's own neutral action: the
agent's actions are multiples of exactly this pace, so beating it means deviating from
neutral pacing adds value, which makes it a fair test by construction. Second, every
production TWAP has this catch-up logic, so it is the realistic desk implementation. Third,
it is the stricter target. In practice the choice does not matter: the two versions agree
to a thousandth of a basis point at the primary setting, and no conclusion changes under
either.

## 5. The robustness grid, and the best story in the project (pack S3)

Why I ran it. A null at one order size and one deadline could be a blind spot, so I
retrained the selected configuration across the full grid: four order sizes (5, 12.5, 25,
50 BTC) by four deadlines (2.5, 5, 10, 20 minutes), both markets, 66 new runs.

What it found. Null in 20 of the 22 newly tested groups. In several groups the agent is
significantly WORSE than TWAP, which an agent with real skill should never be. And the
size direction is wrong for a real edge: a genuine impact-management skill should help
more as orders get bigger, but the original signal existed only at the 25 BTC development
size and nowhere else (pack S3, size-response figure).

The two exceptions, and what happened to them. Two calm-market groups met my pre-set
follow-up trigger, and one looked extraordinary: three independently trained agents landing
within 0.005 basis points of each other, odds of chance about one in 3,000. My rules said:
do not believe it, escalate. Step one, I trained two more agents per group; both groups
still passed, one now at odds of one in 10,000. Step two, I re-tested the same ten agents,
no retraining, on a reserve set of 2,000 runs that had never been used for anything, all
project. Both groups died instantly. One flipped sign entirely, from clearly cheaper to
clearly more expensive. (Pack S3: the ladder figure with the crossing lines, plus the
per-seed ladder table.)

Why this is the best material we have. Total cost of catching it: four training runs. Sealed
tests spent: none. And it is now the third demonstrated case in this project where a result
that would pass normal reporting standards (consistent across seeds, statistically strong,
economically material) turned out to be a property of the specific evaluation data rather
than of the agent. The first was the drift confound; the second was the volatile signal
that died on two sealed sets; this is the third and cleanest. The methodology chapter
writes itself around these three.

## 6. Why DQN fails, now diagnosed rather than described (pack S4)

The failure itself. DQN does not just underperform. At realistic order sizes it stops
trading and lets the environment's forced deadline purchase execute the order, the most
expensive possible behaviour. A behaviour audit catches this before any cost comparison,
so a broken agent can never post a flattering cost number.

Mapping it. A pre-planned 18-run probe across three more settings: DQN is mostly healthy
at 5 BTC, collapses completely at 25 BTC even with the shortest deadline, and collapses at
every 25 BTC deadline. So the driver is order size, not idle time: the problem appears
exactly when the agent's own trades move the market.

Ruling out my configuration. Two earlier fix attempts (stronger exploration; rescaled
reward) did not cure it. And because my DQN uses an unusual update rhythm (fewer, larger
gradient steps), I retrained 18 more agents with the standard library rhythm: the primary
setting was unchanged. Total learning compute is matched with PPO throughout, so the
comparison is fair.

The mechanism. I read the trained networks directly. Their action-value estimates barely
separate the actions at all: the differences are ten times smaller than the per-step reward
noise, for every agent, healthy or not. What distinguishes the collapsed ones is where that
near-flat surface tips: they rank "do nothing" first in about half of all states, healthy
agents in about one in seven. (Pack S4: the by-setting bar chart and the tilt figure.)

My interpretation, clearly labelled as interpretation: PPO rescales its learning signal
every batch, so microscopic differences still teach it; DQN regresses raw values, and at
raw scale these differences are invisible. I claim this for this configuration on this
problem class only; testing Double-DQN and prioritised replay is named future work.

## 7. The cost distributions you asked for (pack S2, last three exhibits)

What I did. I replayed all twenty primary agents deterministically and kept every one of
the 2,000 per-episode costs, for the agents and for both benchmarks. The replay provably
describes the same evaluations as the verdicts: every recomputed average matches the sealed
record exactly, twenty out of twenty.

The two takeaways. First, the agents' full cost distributions are visually
indistinguishable from the benchmarks': execution cost is decided by market conditions,
not by policy, and the volatile market widens every distribution by roughly four times.
Second, the paired differences dwarf their means: the signal we spent months chasing was
about a thousand times smaller than the run-to-run noise, which is exactly why 2,000 paired
runs and strict pre-set rules were necessary in the first place.

## 8. The frozen-replay track, brought up to full strength (pack S5)

Where that track stood. It had 51 trained agents with uneven coverage: some comparisons
were missing entirely, one had a single agent instead of five.

What I did. Trained the 19 missing agents, with configurations copied field by field from
their siblings and verified identical, so every comparison panel is now complete: 70 agents
across data resolution (1-minute and 10-second bars), order size (0.5, 1 and 2 percent of
daily volume) and deadline (30 and 10 minutes), five seeds each, all with validation
results.

The validation picture. No arm beats TWAP consistently. On 10-second bars at the 30-minute
deadline the agents are consistently worse than TWAP. One arm looks good: PPO at the
smallest size on 1-minute bars, cheaper in four of five seeds, 0.089 basis points on
average. But with 14 arms screened, one such arm is expected by pure chance, so it gets no
special treatment: every arm sits the same final exam. The DQN laziness also appears on
this track (six flagged runs), mirroring the reactive track.

Two integrity items I want to be open about, because they are strengths, not embarrassments.
First, a full census exposed four legacy runs that were actually 200,000-step stubs with no
saved models; the old figure built on them was retracted, the seeds were retrained at full
budget, and the true figure is three times weaker than the stub-based one. Caught before
any test data was spent. Second, one claim in my old notes (a deliberately planted fake
signal that the pipeline detected, used as a credibility check) turned out to have no
artifact on disk; I have quarantined that claim and use it nowhere until the small
experiment is regenerated.

What remains. The track's final exam: a one-shot evaluation of all 70 agents on the
untouched test data. The evaluation program is being built now, and before it touches test
data it must reproduce known validation numbers exactly. I will not run it without deciding
together that we are ready, because it can only be run once.

## 9. What I am doing right now, and the step after

- Right now: the frozen-replay test evaluator and its reproduction proof.
- Next, the measured-signal extension, which is the one experiment that can still change
  the headline. The idea: both tracks so far each lack one ingredient. The replay track has
  real predictive signals but no market reaction; the simulator has real reaction but, by
  construction, no predictive signal for the agent to time. The extension closes that gap:
  measure from the order-level data how strongly order-flow imbalance predicts short-term
  returns (how big, over what horizon, how fast it fades, per market), inject exactly that
  measured relationship into the reactive simulator, and re-run a reduced pre-registered
  campaign with a fresh sealed set. If an edge exists anywhere, it should be here.
- After that, in order: intra-day liquidity and volume profiles (also feeds a proper VWAP
  benchmark), Almgren-Chriss and VWAP added to the benchmark set, the attribution study for
  research question 3 (deliberately after the extension verdict, because how we label the
  drivers depends on whether an edge exists), and a decision-frequency check at 0.5 and 2
  seconds.

## 10. Where I would value your steer

1. The headline framing. The null is now established about as thoroughly as it can be: two
   sealed failures, a full size-by-deadline grid, and three demonstrated cases of
   convincing-but-fake edges. My plan is to frame the dissertation around three
   contributions: the rigorous null, the evaluation methodology that caught the fake edges,
   and the DQN diagnosis. Do you agree that is the strongest framing, and does it match the
   publication direction you had in mind?
2. The measured-signal extension. Before I run it I will fix the design in writing: the
   measurement method, the injection (drift-neutral, not double-counting the simulator's
   own price-move channel, the agent's own trades excluded from the signal), and a fresh
   sealed set. I would value your reaction to the design before it runs, since it is the
   one remaining experiment that could overturn the headline.
3. The frozen-replay final exam. All 70 agents, once, on untouched test data, all 14 arms
   reported with the number of comparisons stated. Any concerns before I spend it?
4. Benchmarks. TWAP in two forms now; Almgren-Chriss and VWAP added last, recalibrated to
   this environment. Still the right set and the right order?
5. The quarantined planted-signal check: regenerate the small experiment to restore that
   claim, or drop the claim entirely?
