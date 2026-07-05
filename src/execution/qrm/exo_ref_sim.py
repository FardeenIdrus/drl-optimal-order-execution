"""QRM extension: empirically-calibrated EXOGENOUS reference-price moves.

Why (evidence in BUILD_PLAN "Step 3 — Option A executed", 2026-07-04): the base QRM moves
its price ONLY via queue depletion, but on BTC price motion dominates queue dynamics
(~7 ticks/s, information-driven). Faithfully calibrated queues never empty, so the base
model's price is provably frozen. Huang et al. (2015) note the base model underestimates
volatility and add reference moves beyond pure queue dynamics (their Models II/III — the
vendored ``theta``/``theta_reinit`` machinery, which never engages here because no
endogenous mid-move ever fires). This module supplies that missing component,
**measured from our reconstructed book** rather than assumed:

* :func:`measure_move_process` — the empirical distribution of per-interval mid-price
  tick-moves from the 0.5s reconstruction (Stage L4-2d output).
* :func:`run_exo_qrm` — simulate the calibrated QRM and, at each 0.5s boundary, draw a
  signed tick-move from that distribution and apply it through the engine's own
  ``update_LOB`` (queues shift with AES rescaling / occasionally redraw). Endogenous
  queue dynamics — and the agent's own depth consumption — persist between moves, which
  is the entire point of the reactive environment.

Honesty note for the write-up: with this extension the simulated price VOLATILITY matches
reality partly *by construction* (we inject the measured move distribution). The gate's
informative content therefore shifts to the statistics we do NOT inject — spread, queue
occupancy/depth, event mix, book sanity — plus out-of-sample checks (calibrate the move
process on subset A, compare on subset B) in Stage 3g.

The vendored engine is reused as a library (via :mod:`execution.qrm.vendored`), never edited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from execution.qrm.assemble import QRMBundle
from execution.qrm.vendored import add_vendored_path

logger = logging.getLogger(__name__)

MAX_ABS_MOVE = 40  # clip per-interval moves to +/- this many ticks (tail safety)


def _seed_numba(seed: int) -> None:
    """Seed numba's per-thread RNG (separate from plain numpy's global state)."""
    from numba import njit  # noqa: PLC0415 -- deferred: numba import is slow

    @njit(cache=True)
    def _s(x):  # pragma: no cover -- trivial jitted body
        np.random.seed(x)

    _s(seed)


@dataclass(frozen=True)
class MoveProcess:
    """Empirical per-interval reference-move distribution (signed ticks)."""

    interval_s: float
    moves: np.ndarray  # support, e.g. [-40..40]
    probs: np.ndarray  # P(move = k ticks) per interval

    @property
    def std_per_s(self) -> float:
        m2 = float((self.probs * self.moves.astype(float) ** 2).sum())
        return (m2 / self.interval_s) ** 0.5

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.moves, p=self.probs))


def move_process_from_mids(mid: np.ndarray, tick: float) -> MoveProcess:
    """Empirical per-0.5s tick-move distribution from an array of 0.5s-grid mids.

    Moves are rounded to whole ticks and clipped at +/-``MAX_ABS_MOVE`` (0.5s moves
    beyond that are vanishingly rare; clipping caps tail influence without touching
    the bulk).
    """
    d = np.diff(np.asarray(mid, float)) / tick
    d = np.clip(np.rint(d), -MAX_ABS_MOVE, MAX_ABS_MOVE).astype(int)
    support = np.arange(-MAX_ABS_MOVE, MAX_ABS_MOVE + 1)
    counts = np.bincount(d + MAX_ABS_MOVE, minlength=len(support)).astype(float)
    probs = counts / counts.sum()
    mp = MoveProcess(interval_s=0.5, moves=support, probs=probs)
    logger.info(
        "move process: %d intervals, P(move!=0)=%.3f, std=%.2f ticks (per-s std %.2f)",
        len(d), 1.0 - probs[MAX_ABS_MOVE], float(np.std(d.astype(float))), mp.std_per_s,
    )
    return mp


def measure_move_process(parquet_paths: Sequence[str], tick: float) -> MoveProcess:
    """Empirical distribution of per-0.5s mid tick-moves from the reconstructed book.

    Uses the Stage-2d 0.5s snapshots (`mid` column); see :func:`move_process_from_mids`.
    """
    mids: List[np.ndarray] = [
        pd.read_parquet(p, columns=["mid"])["mid"].to_numpy() for p in parquet_paths
    ]
    return move_process_from_mids(np.concatenate(mids), tick)


def run_exo_qrm(
    bundle: QRMBundle,
    move_process: MoveProcess,
    horizon_s: float,
    initial_price: float,
    theta_reinit: float = 0.1,
    seed: int = 0,
    max_events_per_interval: int = 4000,
) -> dict:
    """Simulate the calibrated QRM with exogenous reference moves; return gate statistics.

    Alternates: run the engine's endogenous dynamics for one ``move_process.interval_s``
    chunk, then draw a signed tick-move and apply it step-by-step through the vendored
    ``update_LOB`` (forced move; ``theta_reinit`` = probability of a full invariant redraw
    per step — kept LOW so queues mostly SHIFT, preserving any standing footprint, which
    is what the agent's impact persistence will rely on).

    Returns a dict of the 3f gate statistics (per-1s mid vol, mean spread in ticks,
    event-type mix, depth-1 occupancy) plus the raw 0.5s mid path.
    """
    add_vendored_path()
    from qrm_core.engine import simulate_QRM_jit  # noqa: PLC0415 -- vendored import
    from qrm_core.sampling import sample_stationary_lob, update_LOB  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    np.random.seed(seed)   # plain-numpy draws in the vendored code
    _seed_numba(seed)      # numba keeps a SEPARATE per-thread RNG state: without this,
    #                        jitted draws depend on whatever ran before (non-reproducible
    #                        seeds; order-dependent tests). Seeding inside njit fixes it.
    K = bundle.K
    tick = bundle.tick

    # initial book from the invariant distributions (as the vendored simulator does)
    state = np.empty(2 * K, np.int8)
    state[:K] = sample_stationary_lob(bundle.inv_bid, np.empty((0,), np.int8))
    state[K:] = sample_stationary_lob(bundle.inv_ask, np.empty((0,), np.int8))
    p_ref = initial_price
    bid_i = int(np.argmax(state[:K] > 0))
    ask_i = int(np.argmax(state[K:] > 0))
    p_mid = 0.5 * ((p_ref + tick * (ask_i + 0.5)) + (p_ref - tick * (bid_i + 0.5)))

    t = 0.0
    mids: List[float] = [p_mid]
    spreads: List[float] = [(ask_i + bid_i + 1.0)]
    ev_counts = np.zeros(4)  # limit, cancel, market, (unused)
    q1_zero = 0
    n_boundaries = 0

    while t < horizon_s:
        t_next = t + move_process.interval_s
        # 1) endogenous QRM dynamics for one interval
        (times, p_mids_c, _p_refs, _sides, _depths, events, _redr, states) = simulate_QRM_jit(
            t, p_mid, p_ref, state, bundle.rate_int_all, tick, 0.7, theta_reinit,
            t_next, bundle.inv_bid, bundle.inv_ask, max_events_per_interval,
            bundle.aes, np.nan,
        )
        if len(times):
            state = states[-1].copy()
            p_mid = float(p_mids_c[-1])
            for k in (1, 2, 3):
                ev_counts[k - 1] += int((events == k).sum())
        t = t_next
        # 2) exogenous reference move.
        #    |move| == 1 tick -> the engine's own shift machinery (queues persist, footprint
        #    preserved). |move| >= 2 ticks (an information jump) -> the book RE-FORMS around
        #    the new price from the invariant distributions -- the paper's Model-III
        #    reinitialisation; repeated one-tick shifts would instead push one side's volume
        #    off the K-level window until it is empty (the engine then rightly refuses).
        dm = move_process.sample(rng)
        if abs(dm) == 1:
            try:
                p_mid, p_ref, state, _r = update_LOB(
                    K, p_ref, state, 1 if dm > 0 else -1, 1.0, theta_reinit, tick,
                    bundle.inv_bid, bundle.inv_ask, bundle.aes,
                )
            except ValueError:  # shift emptied a side (thin book) -> re-form instead
                p_ref += dm * tick
                state[:K] = sample_stationary_lob(bundle.inv_bid, np.empty((0,), np.int8))
                state[K:] = sample_stationary_lob(bundle.inv_ask, np.empty((0,), np.int8))
        elif dm != 0:
            p_ref += dm * tick
            state[:K] = sample_stationary_lob(bundle.inv_bid, np.empty((0,), np.int8))
            state[K:] = sample_stationary_lob(bundle.inv_ask, np.empty((0,), np.int8))
        # bookkeeping at the boundary
        bid_i = int(np.argmax(state[:K] > 0))
        ask_i = int(np.argmax(state[K:] > 0))
        p_mid = 0.5 * ((p_ref + tick * (ask_i + 0.5)) + (p_ref - tick * (bid_i + 0.5)))
        mids.append(p_mid)
        spreads.append(ask_i + bid_i + 1.0)
        q1_zero += int(state[K] == 0)
        n_boundaries += 1

    mids_a = np.asarray(mids)
    one_s = mids_a[:: max(1, int(round(1.0 / move_process.interval_s)))]
    tot = max(ev_counts[:3].sum(), 1.0)
    return {
        "mid_path_05s": mids_a,
        "vol_1s": float(np.std(np.diff(one_s))),
        "spread_ticks_mean": float(np.mean(spreads)),
        "event_mix": {"limit": ev_counts[0] / tot, "cancel": ev_counts[1] / tot,
                      "market": ev_counts[2] / tot},
        "n_events": int(tot),
        "q1_zero_frac": q1_zero / max(n_boundaries, 1),
    }
