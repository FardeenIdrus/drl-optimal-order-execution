"""L2 inversion investigation, stage 2: the REGISTERED FIXED-PACING PROBE.

Stage 1 found that within-episode mid drift flips sign between the validation and
sealed-test periods on exactly the two datasets whose agents invert, and does not
flip on the one that does not. For a BUY order that is a MECHANICAL reason to expect
deviation from TWAP to pay on one period and cost on the other, with no skill
involved -- which is what a diagnosed-broken learner enjoying the same inversion
would require.

Stage 2 tests that directly, WITHOUT ANY TRAINED AGENT. It runs a family of fixed
pacing rules through the identical evaluation machinery on both splits:

    qty_t = m * inventory_t / steps_left_t ,   m in the agents' OWN action grid

m = 1 reproduces TWAP exactly; m > 1 front-loads; m < 1 delays. There are ZERO
fitted parameters and no selection: the whole grid is reported, both splits, always.

Why this is the right probe
  * It isolates PACING. If the period, not the policy, is what pays, then a rule
    that cannot learn anything must still show the flip.
  * It is simultaneously the registered dose-response test (hypothesis 4 on the
    investigation list): the premium should scale with |m - 1| and take its sign
    from the direction of deviation.

GOVERNANCE. The sealed test block is SPENT and its verdict is published and FIXED.
This is diagnostic attribution of an already-reported result, and NOTHING here can
revise that verdict: no pass/fail rule is applied, no agent is selected, no claim of
an edge is made. The probe is run on BOTH splits so it cannot be a test-only
exercise. Result is reported whichever way it comes out.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from execution.env.benchmarks import TWAP
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv
from execution.eval.runner import run_episodes

REPO = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution")
S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = Path(__file__).resolve().parent / "l2_inversion_stage2.json"
VAL_SUBSET_SEED, N_EVAL = 12345, 400

# Every dataset, so the flip can be compared against how strongly each inverted.
TARGETS = [
    ("runs",           "configs/experiment.yaml",            96.57),
    ("runs_10s",       "configs/experiment_10s.yaml",        96.57),
    ("runs_10s_10min", "configs/experiment_10s_10min.yaml",  96.57),
]


class FixedPace:
    """Constant multiple of the adaptive TWAP pace, expressed as a grid action."""
    absolute = False

    def __init__(self, actions, mult):
        self.idx = int(np.argmin([abs(a - mult) for a in actions]))
        self.mult = float(actions[self.idx])

    def reset(self, env=None):
        pass

    def action(self, obs=None, info=None):
        return self.idx


def eval_indices(n):
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    return np.sort(rng.choice(n, size=min(N_EVAL, n), replace=False))


def meta_for(runs_dir, size):
    """Residual coefficients / ADV of record, taken from a trained run's meta.json so
    the probe faces the EXACT environment the agents faced."""
    for d in sorted((S / runs_dir).glob(f"*size{size}_seed*")):
        m = json.loads((d / "meta.json").read_text())
        return {k: float(v) for k, v in m["residual_coef"].items()}, m.get("adv_btc"), d.name
    raise SystemExit(f"no run found under {runs_dir} for size {size}")


def run_split(store, cfg, size, resid, adv, norm):
    actions = cfg["env"]["actions"]
    env = RealDataExecutionEnv(
        store, initial_inventory=size, actions=actions,
        action_mode=cfg["env"]["action_mode"], obs_features=cfg["obs"]["features"],
        normalizer=norm, test_mode=True, residual_coef_by_regime=resid, adv_btc=adv,
    )
    idx = eval_indices(store.n_episodes)
    n_steps = cfg["env"]["n_steps"]
    twap = run_episodes(env, TWAP(size, n_steps), idx)["is_bps"].to_numpy()
    rows = {}
    for m in actions:
        if m == 0.0:            # pure abstention cannot complete; the grid's degenerate end
            continue
        df = run_episodes(env, FixedPace(actions, m), idx)
        d = df["is_bps"].to_numpy() - twap
        rows[f"{m:g}"] = {
            "mean_diff_bps": float(d.mean()),
            "se_bps": float(d.std(ddof=1) / np.sqrt(len(d))),
            "frac_cheaper": float((d < 0).mean()),
            "mean_residual_bps": float(df["residual_bps"].mean()),
            "per_episode": d.tolist(),
        }
        print(f"    m={m:<4g} {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} bps"
              f"  ({(d < 0).mean():.0%} cheaper)", flush=True)
    return rows, twap.tolist(), idx.tolist()


def main():
    out = {}
    for runs_dir, cfg_path, size in TARGETS:
        cfg = yaml.safe_load((REPO / cfg_path).read_text())
        n_steps, val_frac = cfg["env"]["n_steps"], cfg["split"]["val_frac"]
        dd = S / cfg["dataset_dir"]
        resid, adv, src = meta_for(runs_dir, size)
        print(f"### {runs_dir}  (env of record from {src})", flush=True)
        res = {"size_btc": size, "meta_source": src, "n_steps": n_steps}

        print("  VALIDATION", flush=True)
        full = EpisodeStore.from_parquet(dd / "train.parquet", n_steps=n_steps)
        train_sub, val_store = full.chrono_split(val_frac)
        norm = FeatureNormalizer.fit(train_sub.features, train_sub.feature_names)
        r, tw, ix = run_split(val_store, cfg, size, resid, adv, norm)
        res["validation"] = {"grid": r, "twap_is_bps": tw, "eval_indices": ix}
        del full, train_sub, val_store
        gc.collect()

        print("  SEALED TEST", flush=True)
        test_store = EpisodeStore.from_parquet(dd / "test.parquet", n_steps=n_steps)
        r, tw, ix = run_split(test_store, cfg, size, resid, adv, norm)
        res["test"] = {"grid": r, "twap_is_bps": tw, "eval_indices": ix}
        del test_store
        gc.collect()

        out[runs_dir] = res
        OUT.write_text(json.dumps(out, indent=1))       # checkpoint per dataset
        print(f"  -> checkpointed {runs_dir}", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
