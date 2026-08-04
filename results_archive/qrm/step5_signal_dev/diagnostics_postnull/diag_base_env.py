"""Base-env diagnostic (post-hoc, clearly labelled): does the ORIGINAL no-signal
reactive env contain a hand-capturable endogenous imbalance edge? Three fixed,
untuned imbalance-readers vs adaptive TWAP, CRN, dev seeds 18e6, n=2000, base env.
  raw : pace nearest 1 + (S2_inst - s2_mean)          (Phase-B naive form)
  x3  : pace nearest 1 + 3*(S2_inst - s2_mean)        (stronger fixed tilt)
  ema8: pace nearest 1 + 3*(EMA_8s(S2) - s2_mean)     (dominant kernel timescale)
s2_mean = the measured unconditional mean frozen in kernel_solution.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from execution.qrm.step5_judgement import _core
from execution.qrm.reactive_baselines import adaptive_twap, run_episodes
from execution.qrm.reactive_env import ACTIONS

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = SCRATCH / "step5_signal_dev" / "diagnostics_postnull"
SEED0, N = 18_000_000, 2000
S2_MEAN = {"calm": 0.1177860291469997, "volatile": 0.06834063432944774}
EMA_ALPHA = 1.0 - 2.0 ** (-1.0 / 8.0)   # 8 s half-life at the 1 s decision cadence


def _s2(env):
    v = env._s2_bg(env._ep)
    return v if np.isfinite(v) else S2_MEAN[env._regime_tag]


def _snap(want):
    want = min(max(want, 0.0), 2.0)
    return int(np.argmin([abs(a - want) for a in ACTIONS]))


def make_policies(regime):
    mean = S2_MEAN[regime]
    state = {"ema": mean}

    def raw(env, _obs):
        return _snap(1.0 + (_s2(env) - mean))

    def x3(env, _obs):
        return _snap(1.0 + 3.0 * (_s2(env) - mean))

    def ema8(env, _obs):
        if env._ep.step_idx == 0:
            state["ema"] = mean               # reset at episode start
        state["ema"] += EMA_ALPHA * (_s2(env) - state["ema"])
        return _snap(1.0 + 3.0 * (state["ema"] - mean))

    return {"raw": raw, "x3": x3, "ema8": ema8}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + N))
    report = {"seed0": SEED0, "n": N, "env": "BASE (no injection)",
              "note": "post-hoc diagnostic; fixed untuned rules", "regimes": {}}
    for regime in ("calm", "volatile"):
        env = _core(SCRATCH, regime, 25.0, 300, inject=False)
        env._regime_tag = regime
        base = run_episodes(env, adaptive_twap, seeds)
        rows = {}
        for name, pol in make_policies(regime).items():
            r = run_episodes(env, pol, seeds)
            d = r["cost_bps"] - base["cost_bps"]
            nz = d[d != 0]
            rows[name] = {"mean_diff_bps": float(d.mean()),
                          "se": float(d.std(ddof=1) / np.sqrt(N)),
                          "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0,
                          "frac_cheaper": float((d < 0).mean()),
                          "executed_frac": float(r["executed_frac"].mean())}
            print("BASE-ENV", regime, name, json.dumps(rows[name]), flush=True)
        report["regimes"][regime] = {"adaptive_mean_cost_bps": float(base["cost_bps"].mean()),
                                     "exploiters": rows}
        (OUT / "diag_base_env.json").write_text(json.dumps(report, indent=1))
    print("BASE-ENV DONE", flush=True)


if __name__ == "__main__":
    main()
