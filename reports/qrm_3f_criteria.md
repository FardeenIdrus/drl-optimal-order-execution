# QRM 3f validation gate — pre-registered criteria (FROZEN 2026-07-04)

**Status of this freeze.** Two prior gate iterations (2026-07-04, same day) changed the
*mechanism* (exogenous reference-move process; empirical book shape + trades-volume anchor),
not tolerance thresholds — no numeric pass bar existed before this document. This freeze is
pre-committed relative to the quiet-spell iteration and any later ones: from here, further
iterations are **mechanism-gated** (allowed only when a specific defect mechanism is
identified and stated in advance), never tolerance-chasing. Everything below was written
BEFORE the quiet-spell calibration, the burst measurement, or any new simulation was run.
Tolerance NUMBERS are filled by the pre-stated formulas from real data only (no simulator
output involved); the fill happens before any simulation.

---

## 1. Hold-out design

- **Calibration window:** Dec-1 2025, hours 00–02 UTC (3 h of the L4 event stream).
- **Gate window (held out):** Dec-1 2025, hour 03 UTC. Disjoint, same day, adjacent hour —
  tests generalization without confounding regime change.
- **Representativeness rule:** for each gated metric computable from the 0.5 s snapshots
  (M1 vol, M2 spread, M3-proxy wide-spread fraction), the gate hour must lie within the
  5th–95th percentile of the month's hourly distribution. If hour 03 is extreme on any,
  fall back to hour 04, then hour 05 (in that order, no other choices).
- The exogenous move process, the invariant (book-shape) distributions, the AES scale, and
  the market-volume anchor are ALL measured on hours 00–02 only. Nothing in the simulator
  is fitted to hour 03.

## 2. Simulation protocol

- 8 independent simulations, seeds 0–7, 600 s each, K = 5 (locked earlier), Q per the
  pre-stated rule in §5, theta_reinit = 0.1 (unchanged from the prior iteration).
- Initial price = the real mid at the start of the gate hour.
- Gate statistics are computed per seed; the gate is scored on the 8-seed mean.
- Stability guard: in addition to the mean passing, at least 6 of 8 seeds must individually
  fall within the band widened by 50 %. (A passing mean produced by wildly scattered seeds
  is not a pass.)

## 3. Metrics, targets, and tolerance formulas

Real targets come from the gate hour: M1/M2/M3-proxy from the 0.5 s snapshot
reconstruction; M3/M4/M5 targets from a reference-frame walk of the gate hour's event
stream (same code path as calibration, so the real and simulated quantities are defined
identically).

| # | Metric | Sim quantity | Real target (gate hour) | Tolerance (formula) |
|---|--------|--------------|--------------------------|---------------------|
| M1 | Price volatility | std of 1 s mid changes, $ | same, from 0.5 s snapshots | ± T_vol (relative), T_vol = median over the month's days of \|vol(hour 3 of day d) − vol(hours 0–2 of day d)\| / vol(hours 0–2 of day d) — the real data's own calibration-window→gate-window drift, i.e. the deviation even a PERFECT simulator of hours 0–2 would show against hour 3 |
| M2 | Spread | mean spread in ticks over 0.5 s samples | same | ± T_spr (relative), same matched-window drift formula on mean spread |
| M3 | Inner-slot emptiness | fraction of 0.5 s boundaries with ask depth-1 window slot empty (in AES-rounded units: < half an average event size counts as empty, the engine's own state resolution) | time-weighted fraction of the gate hour the ask depth-1 window slot is empty, measured by the same reference-frame walk and the same AES rounding (all-time, unconditioned — matches the sim's all-boundary sampling) | ± T_emp (relative), matched-window drift formula on the hourly wide-spread fraction P(spread ≥ 2 ticks) (month-wide proxy for the same phenomenon: an empty inner slot forces a ≥ 2-tick spread; relative form because the proxy's level differs from the target's level) |
| M4 | Event mix | limit and cancel shares of simulated events | limit and cancel shares of QUIET-SPELL events in the gate hour (the sim's endogenous events correspond to quiet-spell dynamics by construction) | ± 5 pp each (fixed, judgment-based: event mix is a slow structural statistic; a month-wide per-hour event-mix distribution would require ~30 full-day event walks for marginal benefit — documented as the one non-data-derived tolerance) |
| M5 | Market volume rate | realized simulated market-unit consumption × AES(depth 1), BTC/s | (a) internal consistency: the anchor target itself (from hours 00–02); (b) out-of-sample report vs the gate hour's quiet-spell BTC/s | (a) ± 20 % (fixed; checks the anchoring machinery realizes its own target under simulation); (b) report-only, no pass/fail (hour-to-hour traded volume varies too much for a cheap principled band) |

Also reported, not gated: P(spread ≥ 2 ticks) sim vs real (the M3 proxy on both sides),
the full simulated spread distribution, and the anchor scale factor.

**Overshoot check (part of M1):** the band is two-sided; a simulated volatility above the
upper band FAILS even though "more volatility" might look harmless — background price
motion is owned by the exogenous process, and overshoot would mean endogenous depletion is
double-counting it.

## 4. Burst measurement (pre-commitments)

Exposure-normalized: rate at lag τ = (limit arrivals at inner depths 1–2, both sides, at
lag τ since the last reference move) ÷ (total window-slot exposure time observed at lag τ).
Lag bins: 50 ms wide, 0–3 s, plus overflow. Steady-state rate = pooled rate over lags
1–3 s.

- **Burst confirmed** iff the rate in the first bin [0, 50 ms) ≥ 1.5 × steady.
- **Guard window** = the left edge of the first bin from which 3 consecutive bins are all
  ≤ 1.1 × steady; capped at 500 ms.
- **Refutation branch (pre-committed):** if no burst is confirmed, NO guard is implemented,
  the quiet-spell conditioning is dropped, and the double-counting diagnosis is
  reconsidered — the gate is then re-run on the existing (iteration-2) calibration against
  the held-out hour, and the outcome reported as-is.

## 5. Other pre-stated calibration rules

- **Quiet spell** = time since the last reference-price move > guard. Exposure time and
  event counts accumulate only in quiet time; the event that itself triggers a move is
  counted (it arrives during quiet time; consistency of counts with exposure).
- **Same-clock anchor:** market-volume anchor = (BTC traded during quiet time of hours
  00–02) ÷ (quiet seconds) ÷ AES(depth 1). Numerator and denominator on the identical
  quiet-time clock as the intensities.
- **AES** = mean arriving limit-order size per depth, quiet-time only (from the burst
  pass's lag-binned size sums, bins ≥ guard).
- **Q rule:** smallest Q covering 95 % of quiet-time ask depth-1 queue-size exposure (in
  AES units), floor 30 — the same 95 % rule as the prior iteration, now on quiet exposure.
- **Warm-up:** each walk starts on a cold book; accumulation begins only after all of:
  5 min of stream time, 100k events, a set reference price, and one observed reference
  move.
- **Coverage rule:** report exposure seconds and event counts per (side, depth, queue
  bucket). If cells covering > 10 % of total exposure have < 100 events, coarsen the queue
  buckets (halve Q) and re-run — documented, not silent.

## 6. Decision rule (pre-committed)

1. **All of M1–M4 pass + M5(a) passes → 3f PASSED.** Proceed to 3g (regime-conditional
   calibration), then Step 4.
2. **M1–M3 pass; M4 or M5(a) misses by ≤ 1.5 × its band, and the miss direction is
   provably conservative for the execution use-case** (touch too liquid → agent impact
   understated → any RL edge underestimated) → **accept with caveats**, documented in
   BUILD_PLAN and the write-up. Proceed.
3. **Any of M1–M3 fails →** no acceptance. One further iteration is permitted ONLY with a
   specific, stated defect mechanism (as with the two prior iterations); otherwise revert
   to the best prior bundle and reassess direction (Option B / HYPE fallbacks).
4. **Regression guard:** if the quiet-spell calibration is WORSE than iteration 2 on ≥ 2
   gated metrics, revert to iteration 2 and treat the quiet-spell hypothesis as refuted in
   its implemented form.

## 7. Tolerance numbers — PENDING the v2 reconstruction (blocked 2026-07-04)

**Discovery during the fill (2026-07-04, before any simulation):** computing the
tolerances exposed that the Step-2 reconstruction (`book_05s`, "v1") contains ~117
fully-frozen hours and 32.6 % of all 0.5 s snapshots inside ≥ 60 s frozen-mid runs,
clustered on quiet days from Dec-12 22:08 onward. Root causes diagnosed and fixed
(evidence-based crossing guard replacing the "heal bids first" convention; end-of-hour
retry for cancel-vs-new sub-second races); the month is being re-reconstructed to
`book_05s_v2`. Dec-1 (calibration + gate day) is clean in v1 and unaffected.

**Formula revision, made before any simulator run:** the first draft used adjacent-hour
variability; it was revised to the matched calibration-window→gate-window drift (§3)
because that is the exact null of the holdout design. The numbers below are to be filled
from the **v2** reconstruction by the §3 formulas, still before any new simulation:

- T_vol = ____ (median matched-window relative drift, 1 s mid-change std)
- T_spr = ____ (same formula, mean spread in ticks)
- T_emp = ____ (same formula, hourly P(spread ≥ 2 ticks), relative)
- Gate-hour representativeness (hour 03 UTC, Dec-1) vs the v2 month distribution: ____
  (must lie within the 5th–95th percentile on M1/M2/M3-proxy, else fall back per §1).
