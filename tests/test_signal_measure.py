"""Tests for the measured-signal extension's Phase A1 measurement machinery.

Covers: trade binning sign/alignment, causality of the normalisation (no future
leakage), depth-imbalance formula, planted-slope recovery by the streaming regression,
foresight-bound feasibility/completion and its sign convention, and determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from execution.data.l4.book_diffs_reader import ASK, BID
from execution.qrm.signal_measure import (
    RegAccum,
    bin_signed_trades,
    causal_normalise,
    depth_imbalance,
    foresight_bound,
    trailing_sum,
)


# ------------------------------------------------------------------ trade binning
def test_bin_signed_trades_sign_and_alignment():
    grid = np.array([1000, 1500, 2000, 2500], dtype=np.int64)
    trades = [
        (900, ASK, 100.0, 2.0),    # buy-aggressor before/at first grid ts -> bin 0, +2
        (1200, BID, 100.0, 1.0),   # sell-aggressor in (1000,1500] -> bin 1, -1
        (1500, ASK, 100.0, 3.0),   # exactly on a grid ts -> that bin (searchsorted left)
        (2600, BID, 100.0, 5.0),   # after last grid ts -> clipped to last bin, -5
    ]
    out = bin_signed_trades(grid, trades)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(-1.0 + 3.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(-5.0)


def test_bin_signed_trades_empty():
    grid = np.array([1000, 1500], dtype=np.int64)
    assert np.all(bin_signed_trades(grid, []) == 0.0)


# ------------------------------------------------------------------ trailing sum
def test_trailing_sum_window():
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    out = trailing_sum(raw, 2)
    assert out.tolist() == [1.0, 3.0, 5.0, 7.0]


# ------------------------------------------------------------------ causality
def test_causal_normalise_no_future_leakage():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=400)
    a = causal_normalise(raw, norm_bins=50, min_bins=20)
    raw2 = raw.copy()
    raw2[300:] += 100.0          # perturb the future only
    b = causal_normalise(raw2, norm_bins=50, min_bins=20)
    assert np.allclose(a[:300], b[:300], equal_nan=True)


def test_causal_normalise_excludes_current_value():
    rng = np.random.default_rng(7)
    raw = rng.normal(scale=0.1, size=100)
    raw[50] = 1000.0             # spike must not scale ITSELF down via its own std
    out = causal_normalise(raw, norm_bins=30, min_bins=10)
    # normaliser comes from PRIOR values (std ~0.1) -> spike reads ~1000/0.1
    assert abs(out[50]) > 1e3


def test_causal_normalise_constant_input_is_undefined():
    # amendment 1a: constant stretches (zero trailing std) yield NaN, never
    # division by epsilon (the defect that corrupted the first measurement run)
    raw = np.ones(200)
    out = causal_normalise(raw, norm_bins=30, min_bins=10)
    assert np.isnan(out).all()


def test_causal_normalise_warmup_nan():
    raw = np.ones(50)
    out = causal_normalise(raw, norm_bins=30, min_bins=20)
    assert np.isnan(out[:20]).all()


# ------------------------------------------------------------------ depth imbalance
def test_depth_imbalance_formula_and_edge():
    bid = np.array([3.0, 1.0, 0.0])
    ask = np.array([1.0, 3.0, 0.0])
    out = depth_imbalance(bid, ask)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(-0.5)
    assert out[2] == 0.0         # empty book -> defined as 0, not nan/inf


# ------------------------------------------------------------------ regression accumulator
def test_regaccum_recovers_planted_slope():
    rng = np.random.default_rng(1)
    acc = RegAccum()
    for _ in range(20):          # streamed in chunks, as in production
        x = rng.normal(size=500)
        y = 0.7 * x + rng.normal(scale=0.1, size=500)
        acc.add(x, y)
    s = acc.stats()
    assert s["slope"] == pytest.approx(0.7, abs=0.01)
    assert s["r2"] > 0.9
    assert s["p"] < 1e-6
    assert s["n"] == 10000


def test_regaccum_ignores_nan():
    acc = RegAccum()
    x = np.array([1.0, np.nan, 2.0])
    y = np.array([1.0, 1.0, np.nan])
    acc.add(x, y)
    assert acc.stats()["n"] == 1


def test_regaccum_null_slope_high_p():
    rng = np.random.default_rng(2)
    acc = RegAccum()
    acc.add(rng.normal(size=5000), rng.normal(size=5000))
    assert acc.stats()["p"] > 0.01


# ------------------------------------------------------------------ foresight bound
def test_foresight_bound_completes_and_signs():
    n = 10
    # price rises linearly: a clairvoyant buyer front-loads and beats TWAP
    mids = np.linspace(100.0, 101.0, n)
    sigs = np.ones(n)            # positive signal
    advs = foresight_bound(mids, sigs, slope_1s=+1.0, n_decisions=n, fast=2.0)
    assert len(advs) == 1
    assert advs[0] > 0.0         # cheaper than TWAP -> positive advantage

    # with a NEGATIVE slope the same signal predicts falls -> defer, forced tail buy
    advs_neg = foresight_bound(mids, sigs, slope_1s=-1.0, n_decisions=n, fast=2.0)
    assert len(advs_neg) == 1
    assert advs_neg[0] < 0.0     # deferring into a rising market loses vs TWAP


def test_foresight_bound_feasibility_forced_completion():
    n = 8
    mids = np.full(n, 100.0)
    sigs = -np.ones(n)           # never predicts a rise -> buying only when forced
    advs = foresight_bound(mids, sigs, slope_1s=+1.0, n_decisions=n, fast=2.0)
    # flat prices: any completed schedule ties TWAP exactly
    assert len(advs) == 1
    assert advs[0] == pytest.approx(0.0, abs=1e-9)


def test_foresight_bound_nonfinite_window_is_nan_placeholder():
    # amendment 1c pairs real and placebo windows by index, so bad windows must
    # occupy a slot (NaN) rather than be silently dropped
    n = 5
    mids = np.array([100.0, np.nan, 100.0, 100.0, 100.0])
    sigs = np.ones(n)
    out = foresight_bound(mids, sigs, 1.0, n_decisions=n)
    assert len(out) == 1 and np.isnan(out[0])


def test_placebo_correction_cancels_pure_drift():
    # drift-only world: an uninformative signal wins/loses purely via drift; the
    # circular-shift placebo enjoys the same drift, so the correction ~cancels
    rng = np.random.default_rng(11)
    n = 60
    n_windows = 40
    mids = np.linspace(100.0, 102.0, n * n_windows)     # strong upward drift
    sigs = rng.choice([-1.0, 1.0], size=n * n_windows)  # coin-flip signal
    real = np.array(foresight_bound(mids, sigs, +1.0, n_decisions=n))
    plac = np.array(foresight_bound(mids, np.roll(sigs, n), +1.0, n_decisions=n))
    raw_mean = np.nanmean(real)
    corrected = np.nanmean(real - plac)
    assert abs(raw_mean) > 5 * abs(corrected)   # drift dominates raw, cancels in corrected
    assert abs(corrected) < 0.5                  # corrected is near zero (bps)


def test_placebo_correction_preserves_genuine_signal():
    # informative signal: the signal IS the next move's sign; correction must
    # keep a clearly positive advantage (placebo destroys the alignment)
    rng = np.random.default_rng(13)
    n = 60
    n_windows = 40
    steps = rng.choice([-0.02, 0.02], size=n * n_windows)
    mids = 100.0 + np.concatenate([[0.0], np.cumsum(steps[:-1])])
    sigs = np.sign(steps)                        # signal predicts the next move
    real = np.array(foresight_bound(mids, sigs, +1.0, n_decisions=n))
    plac = np.array(foresight_bound(mids, np.roll(sigs, n), +1.0, n_decisions=n))
    corrected = np.nanmean(real - plac)
    assert corrected > 0.1                       # genuine value survives correction


def test_foresight_bound_deterministic():
    rng = np.random.default_rng(3)
    mids = 100.0 + np.cumsum(rng.normal(size=600)) * 0.01
    sigs = rng.normal(size=600)
    a = foresight_bound(mids, sigs, 0.5, n_decisions=300)
    b = foresight_bound(mids, sigs, 0.5, n_decisions=300)
    assert a == b and len(a) == 2

# ------------------------------------------------------------------ A2 endogenous sampler
class _FakeEpisode:
    pass


class _FakeEnv:
    """Duck-typed stand-in for ReactiveQRMEnv: mid climbs 1 tick per interval whenever
    the scripted S2 is positive, else falls; lets tests plant a known relationship."""

    class _B:
        aes = [0.5, 0.5, 0.5, 0.5, 0.5]

    def __init__(self, script):
        self.K = 5
        self.bundle = self._B()
        self._script = script          # list of (bid_units, ask_units) per interval
        self._i = 0

    def reset(self, seed):
        self._i = 0
        self._ep = _FakeEpisode()
        self._ep.p_mid = 100_000.0
        self._set_state()
        return None

    def _set_state(self):
        import numpy as _np
        b, a = self._script[min(self._i, len(self._script) - 1)]
        st = _np.zeros(2 * self.K, dtype=_np.int8)
        st[0], st[self.K] = b, a
        self._ep.state = st

    def _best_slots(self, ep):
        b = ep.state[: self.K]
        a = ep.state[self.K:]
        bi = int(np.argmax(b > 0)) if b.any() else self.K
        ai = int(np.argmax(a > 0)) if a.any() else self.K
        return bi, ai

    def _run_interval(self, track_flow):
        b, a = self._script[min(self._i, len(self._script) - 1)]
        if b + a > 0:
            self._ep.p_mid += 1.0 if b > a else (-1.0 if a > b else 0.0)
        self._i += 1
        self._set_state()


def test_sample_episode_records_before_advancing_and_planted_slope():
    from execution.qrm.signal_measure import (
        HORIZONS_S,
        RegAccum,
        accumulate_episode,
        sample_episode,
    )
    rng = np.random.default_rng(5)
    script = [(3, 1) if rng.random() < 0.5 else (1, 3) for _ in range(400)]
    env = _FakeEnv(script)
    mids, s2 = sample_episode(env, seed=0, n_intervals=400)
    # recorded BEFORE advancing: the first mid is the reset mid
    assert mids[0] == 100_000.0
    # S2 sign matches the scripted imbalance at each recorded instant
    assert (s2[0] > 0) == (script[0][0] > script[0][1])
    accums = {h: RegAccum() for h in HORIZONS_S}
    accumulate_episode(accums, mids, s2)
    s = accums["0.5"].stats()
    assert s["slope"] > 0 and s["p"] < 1e-6   # planted positive relation recovered


def test_sample_episode_swept_side_is_nan():
    from execution.qrm.signal_measure import sample_episode
    env = _FakeEnv([(0, 2)] * 10)             # bid side fully swept
    _, s2 = sample_episode(env, seed=0, n_intervals=10)
    assert np.isnan(s2).all()


def test_accumulate_episode_within_episode_only():
    from execution.qrm.signal_measure import HORIZONS_S, RegAccum, accumulate_episode
    accums = {h: RegAccum() for h in HORIZONS_S}
    mids = np.linspace(100.0, 101.0, 50)
    s2 = np.ones(50)
    accumulate_episode(accums, mids, s2)
    # 60 s horizon = 120 steps > episode length -> contributes nothing
    assert accums["60"].stats()["n"] == 0
    assert accums["0.5"].stats()["n"] == 49
