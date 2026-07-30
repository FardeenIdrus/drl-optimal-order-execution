"""L2 inversion investigation, stage 3: does the AGENTS' inversion ride on the
pacing channel stage 2 identified?

Stage 2 showed, with no learning involved, that a fixed front-loading rule LOSES on
validation and WINS on the sealed test period, monotonically in the size of the
deviation. Stage 3 closes the loop on the actual agents.

For every runs_10s agent (30 = PPO+DQN x 3 sizes x 5 seeds), on BOTH splits, we
recover the per-episode paired difference vs TWAP and regress it on the per-episode
premium of the pure front-loading probe (m = 2):

    (agent - TWAP)_e  =  alpha + beta * (frontload - TWAP)_e  + noise

  * beta   = the agent's EFFECTIVE FRONT-LOADING DOSE. It is a property of the
             POLICY, so if the inversion is a period effect it should be roughly
             STABLE across the two splits.
  * alpha  = whatever the agent achieves that pacing does not explain -- the part
             that could be skill. If the inversion is a period effect, alpha should
             be small and should NOT flip.

The prediction being tested is therefore sharp: beta stable, alpha small, and
beta * (probe premium) accounting for most of the observed val->test shift.

GOVERNANCE: as stage 2. The sealed block is spent, its verdict published and FIXED;
this applies no pass/fail rule, selects no agent, and claims no edge. Both splits
are always run. Reported whichever way it comes out.
"""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from execution.env.benchmarks import TWAP
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv
from execution.eval.runner import run_episodes
from execution.agents.policy import SB3Policy

REPO = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution")
S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
STAGE2 = Path(__file__).resolve().parent / "l2_inversion_stage2.json"
OUT = Path(__file__).resolve().parent / "l2_inversion_stage3.json"
RUNS_DIR, CFG = "runs_10s", "configs/experiment_10s.yaml"
VAL_SUBSET_SEED, N_EVAL = 12345, 400
_RUN_RE = re.compile(r"^(?P<algo>ppo|dqn)_size(?P<size>[0-9.]+)_seed(?P<seed>\d+)$")


class FixedPace:
    absolute = False

    def __init__(self, actions, mult):
        self.idx = int(np.argmin([abs(a - mult) for a in actions]))

    def reset(self, env=None):
        pass

    def action(self, obs=None, info=None):
        return self.idx


def _load_model(algo, path):
    from stable_baselines3 import DQN, PPO
    return (PPO if algo == "ppo" else DQN).load(str(path), device="cpu")


def eval_indices(n):
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    return np.sort(rng.choice(n, size=min(N_EVAL, n), replace=False))


def score_split(store, cfg, split, runs):
    actions = cfg["env"]["actions"]
    n_steps = cfg["env"]["n_steps"]
    idx = eval_indices(store.n_episodes)
    twap_cache, probe_cache, out = {}, {}, {}
    for rd in runs:
        meta = json.loads((rd / "meta.json").read_text())
        algo, size = meta["algo"], float(meta["size_btc"])
        resid = {k: float(v) for k, v in meta["residual_coef"].items()}
        adv = meta.get("adv_btc")
        norm = FeatureNormalizer.from_json(rd / "normalizer.json")
        env = RealDataExecutionEnv(
            store, initial_inventory=size, actions=actions,
            action_mode=cfg["env"]["action_mode"], obs_features=cfg["obs"]["features"],
            normalizer=norm, test_mode=True, residual_coef_by_regime=resid, adv_btc=adv,
        )
        key = (size, tuple(sorted(resid.items())), adv)
        if key not in twap_cache:
            twap_cache[key] = run_episodes(env, TWAP(size, n_steps), idx)["is_bps"].to_numpy()
            probe_cache[key] = (run_episodes(env, FixedPace(actions, 2.0), idx)["is_bps"]
                                .to_numpy() - twap_cache[key])
        twap, probe = twap_cache[key], probe_cache[key]
        agent = run_episodes(env, SB3Policy(_load_model(algo, rd / "model.zip")),
                             idx)["is_bps"].to_numpy()
        d = agent - twap
        beta, alpha = np.polyfit(probe, d, 1)
        r = float(np.corrcoef(probe, d)[0, 1])
        out[rd.name] = {
            "algo": algo, "size_btc": size, "split": split,
            "mean_diff_bps": float(d.mean()),
            "se_bps": float(d.std(ddof=1) / np.sqrt(len(d))),
            "beta_frontload_dose": float(beta),
            "alpha_bps": float(alpha),
            "corr_with_probe": r,
            "probe_mean_bps": float(probe.mean()),
            "explained_by_pacing_bps": float(beta * probe.mean()),
        }
        print(f"    {rd.name:<26} mean {d.mean():+.4f}  beta {beta:+.3f}  "
              f"alpha {alpha:+.4f}  r {r:+.3f}", flush=True)
    return out


def main():
    cfg = yaml.safe_load((REPO / CFG).read_text())
    n_steps, val_frac = cfg["env"]["n_steps"], cfg["split"]["val_frac"]
    dd = S / cfg["dataset_dir"]
    runs = sorted(d for d in (S / RUNS_DIR).iterdir()
                  if d.is_dir() and _RUN_RE.match(d.name) and (d / "model.zip").exists())
    print(f"{len(runs)} agents under {RUNS_DIR}", flush=True)
    res = {}

    print("VALIDATION", flush=True)
    full = EpisodeStore.from_parquet(dd / "train.parquet", n_steps=n_steps)
    train_sub, val_store = full.chrono_split(val_frac)
    del full
    gc.collect()
    res["validation"] = score_split(val_store, cfg, "val", runs)
    del train_sub, val_store
    gc.collect()
    OUT.write_text(json.dumps(res, indent=1))

    print("SEALED TEST", flush=True)
    test_store = EpisodeStore.from_parquet(dd / "test.parquet", n_steps=n_steps)
    res["test"] = score_split(test_store, cfg, "test", runs)
    del test_store
    gc.collect()
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
