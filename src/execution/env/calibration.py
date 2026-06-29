"""Calibrate impact parameters from the real book: linear (Almgren-Chriss) and
square-root (the deadline residual), from one sampled cost-vs-size curve.

Both are estimated on TRAIN data only and reported, so the parameters are grounded
in observed Hyperliquid liquidity rather than chosen by hand.

* **Linear temporary impact** ``cost(q) ~ eta * q^2`` -- the Almgren-Chriss
  reference. The real convex book only approximates it; the reported ``r2_linear``
  quantifies that approximation, and it is the documented basis for treating AC as
  a theoretical reference rather than a like-for-like optimum.
* **Square-root law** ``frac_cost(q) ~ (Y*sigma) * sqrt(q / V)`` (V = ADV) -- used
  to price any inventory left unexecuted at the 30-min deadline (beyond visible
  depth). Calibrated from the SAME sampled fills as ``eta`` (a single pass), so the
  residual cost is continuous with the same real book Almgren-Chriss is fit to.
  Applying a metaorder impact law to a within-book residual is an approximation
  (stated openly); ``r2_sqrt`` quantifies the fit and the implied ``Y`` (reported
  via :func:`estimate_relative_daily_vol`) can be sanity-checked against the
  literature's near-universal ``Y ~ 1`` (Almgren 2005; Toth 2011).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution.env.episode_store import EpisodeStore
from execution.env.fills import walk_ask_ladder

MINUTES_PER_DAY = 1440  # crypto trades 24/7; minute-resolution data


@dataclass(frozen=True)
class ImpactCalibration:
    """Calibrated impact parameters from one real-book cost-vs-size curve."""

    eta: float          # linear temporary impact (AC): cost(q) ~ eta * q^2  [USD/BTC^2]
    r2_linear: float    # linear-fit R^2 (how convex the real book is)
    sqrt_coef: float    # Y*sigma in frac_cost(q) ~ sqrt_coef * sqrt(q/V); NaN if no ADV
    r2_sqrt: float      # square-root-fit R^2; NaN if no ADV
    n_points: int       # number of complete-fill q points used in the fits


def estimate_volatility(store: EpisodeStore, regime: str | None = None) -> float:
    """Per-step absolute mid volatility (USD): std of within-episode mid changes."""
    mid = store.mid
    if regime is not None:
        mid = mid[store.regime == regime]
    if mid.shape[0] == 0:
        return float("nan")
    return float(np.std(np.diff(mid, axis=1)))


def estimate_relative_daily_vol(store: EpisodeStore, regime: str | None = None) -> float:
    """Dimensionless daily volatility: per-minute relative mid std scaled to a day.

    Used to decompose the calibrated ``sqrt_coef`` (= Y*sigma) into an implied
    near-universal ``Y`` for reporting; it is NOT needed by the environment (which
    stores the combined ``sqrt_coef`` per regime).
    """
    mid = store.mid
    if regime is not None:
        mid = mid[store.regime == regime]
    if mid.shape[0] == 0:
        return float("nan")
    per_min_abs = float(np.std(np.diff(mid, axis=1)))
    mean_mid = float(np.mean(mid))
    if mean_mid <= 0:
        return float("nan")
    return per_min_abs / mean_mid * np.sqrt(MINUTES_PER_DAY)


def calibrate_temporary_impact(
    store: EpisodeStore,
    q_grid,
    *,
    adv: float | None = None,
    n_samples: int = 3000,
    regime: str | None = None,
    seed: int = 0,
) -> ImpactCalibration:
    """Fit linear ``eta`` and square-root ``sqrt_coef`` from real books, one pass.

    For a sample of (episode, step) books, each size ``q`` in ``q_grid`` is filled
    by walking the real ask ladder; the USD cost vs mid feeds the linear fit and
    the fractional cost (cost / mid-notional) feeds the square-root fit. Samples
    that exhaust the book are dropped per-``q`` (incomplete fills would bias both).

    Args:
        q_grid: trade sizes (BTC) to probe; keep within typical visible depth.
        adv: average daily volume (BTC). If ``None`` the square-root fit is
            skipped (``sqrt_coef``/``r2_sqrt`` = NaN) -- used when only the AC
            ``eta`` is needed.
    Returns:
        :class:`ImpactCalibration`.
    """
    rng = np.random.default_rng(seed)
    eps = np.arange(store.n_episodes)
    if regime is not None:
        eps = eps[store.regime == regime]
    if eps.size == 0:
        return ImpactCalibration(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    qg = np.asarray(q_grid, dtype=float)
    sums = np.zeros_like(qg)       # sum of USD cost per q
    frac_sums = np.zeros_like(qg)  # sum of fractional cost (cost / mid-notional) per q
    counts = np.zeros_like(qg)     # number of complete fills per q
    for _ in range(n_samples):
        e = int(rng.choice(eps))
        s = int(rng.integers(store.n_steps))
        px, sz, mid = store.ask_px[e, s], store.ask_sz[e, s], store.mid[e, s]
        for i, q in enumerate(qg):
            f = walk_ask_ladder(px, sz, q)
            if f.exhausted:
                continue                       # don't bias the fits with partial fills
            cost = f.cash - mid * f.filled
            sums[i] += cost
            frac_sums[i] += cost / (mid * f.filled)
            counts[i] += 1

    valid = counts > 0
    if valid.sum() < 2:
        raise ValueError("insufficient complete-fill samples to calibrate impact")
    qv = qg[valid]
    mean_cost = sums[valid] / counts[valid]

    # Linear temporary impact (AC): cost ~ eta * q^2, OLS through origin.
    eta, r2_linear = _ols_through_origin(qv ** 2, mean_cost)

    # Square-root law: frac_cost ~ sqrt_coef * sqrt(q / V), OLS through origin.
    sqrt_coef, r2_sqrt = float("nan"), float("nan")
    if adv is not None:
        if adv <= 0:
            raise ValueError("adv must be positive for the square-root fit")
        mean_frac = frac_sums[valid] / counts[valid]
        sqrt_coef, r2_sqrt = _ols_through_origin(np.sqrt(qv / adv), mean_frac)

    return ImpactCalibration(eta, r2_linear, sqrt_coef, r2_sqrt, int(valid.sum()))


def _ols_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y ~ coef * x`` (no intercept) and its R^2."""
    sxx = float(np.sum(x * x))
    if sxx == 0:
        return float("nan"), float("nan")
    coef = float(np.sum(x * y) / sxx)
    ss_res = float(np.sum((y - coef * x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2
