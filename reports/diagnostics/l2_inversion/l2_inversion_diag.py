"""L2 inversion investigation, stage 1: PRICE-PATH structure only.

Question: why does the sealed TEST period reward deviation from TWAP when the
adjacent VALIDATION period punishes it -- for every agent, including one
independently diagnosed as non-functional?

Stage 1 tests the two hypotheses that need NO agent and NO re-evaluation, so the
spent test block is not touched: they are properties of the price paths alone.

  H-A  within-episode drift.  The order is a BUY. TWAP pays the average book price
       over the episode; a front-loader pays the arrival price. So the payoff to
       front-loading is, to first order, -(mean_t mid_t - mid_0)/mid_0. If test
       episodes drift UP and validation episodes drift DOWN, deviation is rewarded
       on one period and punished on the other MECHANICALLY, with no skill involved
       -- which is exactly the pattern a broken learner would also enjoy.

  H-B  mean reversion.  Variance ratio VR(k) = Var(r_k) / (k Var(r_1)) on the
       within-episode mid returns. VR < 1 = mean-reverting, VR > 1 = trending.

  H-C  time-of-day composition (cheap, computed here too): do the two eval subsets
       draw from different hours of the day?

Episode subsets are reconstructed with the SAME rule the evaluator uses
(rng(12345).choice(n, 400), sorted) so the episodes analysed are the episodes scored.
Memory: only 4 columns are read from parquet; the (E,T,20) book is never built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = Path(__file__).resolve().parent / "l2_inversion_stage1.json"
VAL_SUBSET_SEED, N_EVAL = 12345, 400
COLS = ["episode_id", "step", "ts", "mid"]

DATASETS = {                       # runs_dir -> (dataset dir, n_steps, val_frac)
    "runs":            ("dataset", 30, 0.15),
    "runs_10s":        ("dataset_10s", 180, 0.15),
    "runs_10s_10min":  ("dataset_10s_10min", 60, 0.15),
}


def load_mid(path: Path, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (episode_ids, mid (E,T), first-step ts (E,)). Chronological order preserved."""
    df = pd.read_parquet(path, columns=COLS)
    eids = df["episode_id"].to_numpy()
    uniq, first = np.unique(eids, return_index=True)
    order = np.argsort(first)                       # chronological, as written
    uniq = uniq[order]
    E = len(uniq)
    if len(df) != E * n_steps:
        raise SystemExit(f"{path.name}: {len(df)} rows != {E}x{n_steps}")
    mid = df["mid"].to_numpy().reshape(E, n_steps)
    ts = df["ts"].to_numpy().reshape(E, n_steps)[:, 0]
    return uniq, mid, ts


def eval_subset(n: int) -> np.ndarray:
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    k = min(N_EVAL, n)
    return np.sort(rng.choice(n, size=k, replace=False))


def variance_ratio(mid: np.ndarray, k: int) -> float:
    """VR(k) on within-episode log returns, pooled across episodes."""
    lg = np.log(mid)
    r1 = np.diff(lg, axis=1).ravel()
    rk = (lg[:, k::k] - lg[:, :-k:k]).ravel() if mid.shape[1] > k else np.array([])
    if rk.size < 30 or r1.var() == 0:
        return float("nan")
    return float(rk.var() / (k * r1.var()))


def describe(mid: np.ndarray, ts: np.ndarray, label: str) -> dict:
    m0 = mid[:, :1]
    # payoff to front-loading a BUY, in bps: TWAP pays the mean path, arrival pays mid_0
    drift_bps = (mid.mean(axis=1) - mid[:, 0]) / mid[:, 0] * 1e4
    term_bps = (mid[:, -1] - mid[:, 0]) / mid[:, 0] * 1e4
    lg = np.log(mid)
    r1 = np.diff(lg, axis=1)
    ac1 = float(np.corrcoef(r1[:, :-1].ravel(), r1[:, 1:].ravel())[0, 1])
    hours = pd.to_datetime(ts, unit="ms", errors="coerce")
    if hours.isna().all():
        hours = pd.to_datetime(ts, errors="coerce")
    hr = hours.hour.to_numpy() if hasattr(hours, "hour") else np.full(len(ts), -1)
    return {
        "label": label,
        "n_episodes": int(mid.shape[0]),
        "n_steps": int(mid.shape[1]),
        # H-A
        "drift_mean_bps": float(drift_bps.mean()),
        "drift_se_bps": float(drift_bps.std(ddof=1) / np.sqrt(len(drift_bps))),
        "drift_median_bps": float(np.median(drift_bps)),
        "drift_frac_positive": float((drift_bps > 0).mean()),
        "terminal_mean_bps": float(term_bps.mean()),
        "terminal_se_bps": float(term_bps.std(ddof=1) / np.sqrt(len(term_bps))),
        # H-B
        "ac1_step_returns": ac1,
        "vr2": variance_ratio(mid, 2),
        "vr5": variance_ratio(mid, 5),
        "vr10": variance_ratio(mid, 10),
        "step_vol_bps": float(r1.std() * 1e4),
        # H-C
        "hour_hist": {int(h): int(c) for h, c in zip(*np.unique(hr, return_counts=True))},
        "ts_first": str(hours[0]) if len(hours) else "",
        "ts_last": str(hours[-1]) if len(hours) else "",
    }


def main() -> None:
    out = {}
    for runs_dir, (dd, n_steps, val_frac) in DATASETS.items():
        print(f"### {runs_dir}  ({dd}, {n_steps} steps)", flush=True)
        res = {}
        # --- validation = last val_frac of train.parquet, chronologically ---
        eids, mid, ts = load_mid(S / dd / "train.parquet", n_steps)
        n = len(eids)
        n_val = max(1, min(n - 1, int(round(n * val_frac))))
        cut = n - n_val
        vmid, vts = mid[cut:], ts[cut:]
        idx = eval_subset(len(vmid))
        res["validation"] = describe(vmid[idx], vts[idx], f"{runs_dir} validation")
        res["validation"]["eval_indices_head"] = idx[:5].tolist()
        del mid, ts, vmid, vts, eids

        # --- sealed test split ---
        eids, mid, ts = load_mid(S / dd / "test.parquet", n_steps)
        idx = eval_subset(len(mid))
        res["test"] = describe(mid[idx], ts[idx], f"{runs_dir} test")
        res["test"]["eval_indices_head"] = idx[:5].tolist()
        del mid, ts, eids

        v, t = res["validation"], res["test"]
        res["delta_drift_bps"] = t["drift_mean_bps"] - v["drift_mean_bps"]
        print(f"  drift (payoff to front-loading = MINUS this):"
              f"  val {v['drift_mean_bps']:+.3f}+-{v['drift_se_bps']:.3f}"
              f"  test {t['drift_mean_bps']:+.3f}+-{t['drift_se_bps']:.3f} bps")
        print(f"  VR(10): val {v['vr10']:.3f}  test {t['vr10']:.3f}"
              f"   AC1: val {v['ac1_step_returns']:+.4f}  test {t['ac1_step_returns']:+.4f}")
        out[runs_dir] = res
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
