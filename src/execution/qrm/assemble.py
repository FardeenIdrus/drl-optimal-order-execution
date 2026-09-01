"""Package the measured calibration into the QRM engine's format.

Turns the raw ``counts`` / ``time_in_state`` accumulators into the three objects the
vendored ``qrm_core`` engine samples from:

    * ``rate_int_all`` (2, K, Q+1, 3) - intensities as (side, depth, queue, type). Our
      calibrator produces (depth, queue, side, type), so this is a transpose.
    * ``inv_bid`` / ``inv_ask`` (K, Q+1) - the stationary queue-size distribution per depth,
      derived ANALYTICALLY from the intensities (birth-death balance; Huang et al. 2015
      Sec 2.3.3): rho_i(q) = lambda_limit(q) / (lambda_cancel(q+1) + lambda_market(q+1)),
      pi(0) = 1/(1 + sum cumprod rho), pi(q) = pi(0)*cumprod(rho). No separate measurement.
    * ``aes`` (K,) - average event size per depth (already measured).

Everything is checked (rates >= 0, distributions sum to 1) before use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from execution.qrm.calibrate_intensities import intensities_from

logger = logging.getLogger(__name__)

_EPS = 1e-12


def invariant_distribution(lam_side: np.ndarray) -> np.ndarray:
    """Stationary queue-size distribution per depth from one side's intensities.

    ``lam_side`` is (K, Q+1, 3) with type order (limit, cancel, market). Returns (K, Q+1),
    each row a probability distribution summing to 1.
    """
    K, Q1, _ = lam_side.shape
    pi = np.zeros((K, Q1))
    for i in range(K):
        lam_limit = lam_side[i, :-1, 0]   # arrivals at queue 0..Q-1
        lam_cancel = lam_side[i, 1:, 1]   # departures at queue 1..Q
        lam_market = lam_side[i, 1:, 2]
        denom = lam_cancel + lam_market
        rho = np.where(denom > _EPS, lam_limit / np.where(denom > _EPS, denom, 1.0), 0.0)
        cum = np.cumprod(rho)
        p0 = 1.0 / (1.0 + cum.sum())
        pi[i, 0] = p0
        pi[i, 1:] = p0 * cum
        s = pi[i].sum()
        if s > 0:
            pi[i] /= s
    return pi


def build_rate_array(lam: np.ndarray) -> np.ndarray:
    """(K, Q+1, 2, 3) intensities -> engine's (2, K, Q+1, 3) = (side, depth, queue, type)."""
    return np.ascontiguousarray(lam.transpose(2, 0, 1, 3))


def sanitize_impossible_transitions(lam: np.ndarray) -> np.ndarray:
    """Zero intensities for transitions the engine's state space cannot represent.

    The engine holds queues in WHOLE AES units, so state q=0 is literally empty: a
    cancel or market order there is impossible. The calibration, however, measures
    CONTINUOUS volumes and rounds — a level holding < 0.5 AES rounds to bucket 0 yet
    still hosts real cancels/fills, so the raw table carries event mass the engine can
    only reject (its sampler retries 10x then aborts — observed as gate seed crashes).
    Same logic at the cap: a limit arrival at q=Q would leave the modelled state space.
    Zeroing these cells makes the table consistent with the engine's dynamics; the
    discarded mass is the sub-half-AES micro-queue activity, which the unit-quantised
    engine cannot express by construction (documented approximation).
    """
    out = lam.copy()
    out[:, 0, :, 1] = 0.0   # cancel at an empty queue
    out[:, 0, :, 2] = 0.0   # market order at an empty queue
    out[:, -1, :, 0] = 0.0  # limit arrival at the queue cap
    return out


def empirical_invariant(time_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Measured stationary queue-size distributions ``(inv_bid, inv_ask)`` per depth.

    Normalises the calibrator's ``time_in_state`` (K, Q+1, 2) per (depth, side): the
    fraction of TIME each window slot spent at each size IS the empirical stationary
    distribution. Preferred over the analytic birth-death derivation for the engine's
    redraws: the analytic version inherits any residual bias in the intensities (on BTC
    it carries almost no empty-slot mass, while the real inner slot is empty ~32% of the
    time), whereas this is a direct measurement of the book's shape.
    """
    tot = time_in.sum(axis=1, keepdims=True)  # (K, 1, 2)
    pi = np.where(tot > 0, time_in / np.maximum(tot, 1e-12), 0.0)
    K, Q1, _ = time_in.shape
    for si in range(2):  # a never-observed depth (all-zero row) -> degenerate at q=0
        for d in range(K):
            if pi[d, :, si].sum() <= 0:
                pi[d, 0, si] = 1.0
    return pi[:, :, 0].copy(), pi[:, :, 1].copy()


def scale_market_to_rate(
    lam: np.ndarray, inv_bid: np.ndarray, inv_ask: np.ndarray, target_units_per_s: float
) -> Tuple[np.ndarray, float]:
    """Scale the market-order intensities so the book-wide expected unit-consumption rate
    matches the trades file.

    The engine's market events consume ONE queue unit (~1 AES) each, so the anchor is the
    trades file's traded volume per second divided by AES ("units/s"), not the raw trade
    count. Expected sim rate = sum over sides/depths/queues of pi(q) * lambda_market(q);
    the market intensities are multiplied by ``target / expected``. Limit and cancel
    intensities are untouched (their mix already matches). Returns ``(lam_scaled, factor)``.
    """
    pi = np.stack([inv_bid, inv_ask], axis=2)          # (K, Q+1, 2)
    expected = float((pi * lam[:, :, :, 2]).sum())     # units per second, whole book
    factor = 1.0 if expected <= 0 else target_units_per_s / expected
    out = lam.copy()
    out[:, :, :, 2] *= factor
    return out, factor


@dataclass
class QRMBundle:
    """A calibrated QRM ready to hand to the engine."""

    rate_int_all: np.ndarray  # (2, K, Q+1, 3)
    inv_bid: np.ndarray       # (K, Q+1)
    inv_ask: np.ndarray       # (K, Q+1)
    aes: np.ndarray           # (K,)
    tick: float
    K: int
    Q: int

    def check(self) -> None:
        """Assert the bundle is well-formed (shapes, non-negative rates, valid distributions)."""
        assert self.rate_int_all.shape == (2, self.K, self.Q + 1, 3)
        assert self.inv_bid.shape == self.inv_ask.shape == (self.K, self.Q + 1)
        assert self.aes.shape == (self.K,)
        assert (self.rate_int_all >= 0).all(), "negative intensity"
        for pi in (self.inv_bid, self.inv_ask):
            assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-6), "invariant rows must sum to 1"

    def save(self, path: str) -> None:
        np.savez(path, rate_int_all=self.rate_int_all, inv_bid=self.inv_bid,
                 inv_ask=self.inv_ask, aes=self.aes, tick=self.tick, K=self.K, Q=self.Q)


def load_bundle(path: str) -> QRMBundle:
    d = np.load(path)
    return QRMBundle(d["rate_int_all"], d["inv_bid"], d["inv_ask"], d["aes"],
                     float(d["tick"]), int(d["K"]), int(d["Q"]))


def assemble(
    counts: np.ndarray,
    time_in: np.ndarray,
    aes,
    tick: float,
    invariant: str = "analytic",
    market_target_units_per_s: float | None = None,
) -> QRMBundle:
    """Build a checked :class:`QRMBundle` from calibration accumulators.

    Args:
        invariant: ``"analytic"`` (birth-death derivation from the intensities) or
            ``"empirical"`` (the measured time-weighted queue-size distributions --
            preferred on BTC; see :func:`empirical_invariant`).
        market_target_units_per_s: if given, market intensities are rescaled so the
            book-wide expected unit-consumption rate matches this trades-file anchor
            (see :func:`scale_market_to_rate`).
    """
    lam = sanitize_impossible_transitions(intensities_from(counts, time_in))  # (K, Q+1, 2, 3)
    K, Q1 = lam.shape[0], lam.shape[1]
    if invariant == "empirical":
        inv_bid, inv_ask = empirical_invariant(time_in)
    elif invariant == "analytic":
        inv_bid = invariant_distribution(lam[:, :, 0, :])
        inv_ask = invariant_distribution(lam[:, :, 1, :])
    else:
        raise ValueError(f"unknown invariant mode: {invariant!r}")
    if market_target_units_per_s is not None:
        lam, factor = scale_market_to_rate(lam, inv_bid, inv_ask, market_target_units_per_s)
        logger.info("market intensities scaled by %.3f to hit %.1f units/s",
                    factor, market_target_units_per_s)
    bundle = QRMBundle(
        rate_int_all=build_rate_array(lam),
        inv_bid=inv_bid,
        inv_ask=inv_ask,
        aes=np.asarray(aes, float),
        tick=float(tick), K=K, Q=Q1 - 1,
    )
    bundle.check()
    return bundle
