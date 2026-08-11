"""Post-hoc critic diagnostic on the PRIMARY (base, non-injected) reacting-market track.

WHY THIS EXISTS. Addendum (P) measured the critic on the INJECTED track only: explained
variance -0.004 across ten agents, rising to +0.405 once price-vs-arrival was added to the
observation. Amendment A4.3 then re-ran the observation fix on the PRIMARY track -- the track
carrying the dissertation's central claim -- and reported the COST verdict, but no critic
diagnostic was ever run there. So the mechanism ("the value function was unfittable because the
dominant driver of the return was unobserved") is measured on the third environment and merely
assumed on the first. This closes that.

WHAT IT DOES. Loads the trained PPO agents from both primary-track campaigns, rolls each one
out deterministically on the campaign's own evaluation seeds, and measures how well its critic
predicts the realised discounted return-to-go. No retraining, no registered rule touched, no
sealed block opened: the evaluation seeds are the development block the campaigns were judged
on (eval_seed0 = 5,000,000, matching step5_v3/judgement.json).

  runs_primary_v3         obs_dim 27  the campaign of record
  runs_primary_v3_obsfix  obs_dim 28  A4.3, price-vs-arrival added

DQN IS EXCLUDED, DELIBERATELY. Explained variance of a critic is a policy-gradient quantity.
DQN has no critic in this sense -- one network, no separate value head fitted against returns --
so the measurement is undefined for it, not merely inconvenient. A4.3's DQN arm was 10/10
behaviourally invalid in any case.

SIGNAL RESPONSE IS ALSO DROPPED, and for a concrete reason: the base environment has no
injected signal, so observation index 27 is price-vs-arrival in the fixed agents and does not
exist at all in the originals. The sweep that diag_learning.py performs has no counterpart here.

Sources (absolute):
  .../scratch_hyperliquid/oxford_l4/runs_primary_v3/ppo_*/model.zip
  .../scratch_hyperliquid/oxford_l4/runs_primary_v3_obsfix/ppo_*/model.zip
Output:
  .../scratch_hyperliquid/oxford_l4/diagnostics_primary/diag_learning_primary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/"
                   "drl-optimal-order-execution/src")
from stable_baselines3 import PPO                                       # noqa: E402
from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv          # noqa: E402

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = S / "diagnostics_primary"
GAMMA = 0.995 ** (1.0 / 60.0)
N_EP = 40                       # episodes per agent, as in diag_learning.py
EVAL_SEED0 = 5_000_000          # the primary track's development block

ARMS = {
    "original": ("runs_primary_v3", False),
    "obsfix": ("runs_primary_v3_obsfix", True),
}


def make_base_env(regime: str, price_vs_arrival: bool) -> ReactiveQRMEnv:
    """The base reacting environment. make_env() in ac_vwap.py cannot express the
    observation flag, so the env is constructed directly with the same bundle paths."""
    return ReactiveQRMEnv(
        str(S / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(S / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=25.0, n_steps=300,
        obs_price_vs_arrival=price_vs_arrival)


def rollout(model, env, seeds):
    OBS, REW = [], []
    for seed in seeds:
        obs = env.reset(seed=seed)
        done, o_ep, r_ep = False, [], []
        while not done:
            o_ep.append(obs.copy())
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(int(a))
            r_ep.append(r)
        OBS.append(np.array(o_ep, dtype=np.float32))
        REW.append(np.array(r_ep, dtype=np.float64))
    return OBS, REW


def returns_to_go(rews):
    g = np.zeros_like(rews)
    acc = 0.0
    for i in range(len(rews) - 1, -1, -1):
        acc = rews[i] + GAMMA * acc
        g[i] = acc
    return g


def analyse(arm: str, run: str, regime: str) -> dict:
    run_dir, pva = ARMS[arm][0], ARMS[arm][1]
    model = PPO.load(str(S / run_dir / run / "model.zip"), device="cpu")
    env = make_base_env(regime, pva)
    OBS, REW = rollout(model, env, list(range(EVAL_SEED0, EVAL_SEED0 + N_EP)))

    allobs = np.concatenate(OBS)
    allret = np.concatenate([returns_to_go(r) for r in REW])
    t = torch.as_tensor(allobs)
    with torch.no_grad():
        V = model.policy.predict_values(t).numpy().ravel()
        dist = model.policy.get_distribution(t)
        logits = dist.distribution.logits.numpy()
        ent = dist.entropy().numpy()

    err = allret - V
    ev = 1.0 - err.var() / allret.var() if allret.var() > 0 else float("nan")
    inv = allobs[:, 0]
    gaps = np.sort(logits, axis=1)
    top2 = gaps[:, -1] - gaps[:, -2]
    ep_returns = np.array([returns_to_go(r)[0] for r in REW])

    return {
        "arm": arm, "run": run, "regime": regime,
        "obs_dim": int(allobs.shape[1]),
        "critic": {
            "explained_variance": float(ev),
            "mean_V": float(V.mean()), "mean_return": float(allret.mean()),
            "bias": float(err.mean()), "err_std": float(err.std()),
            "corr_V_inventory": float(np.corrcoef(V, inv)[0, 1]),
            "corr_V_return": float(np.corrcoef(V, allret)[0, 1]),
        },
        "policy": {
            "mean_state_entropy": float(ent.mean()),
            "entropy_uniform": float(np.log(len(ACTIONS))),
            "mean_top2_logit_margin": float(top2.mean()),
            "frac_states_confident": float((top2 > 1.0).mean()),
        },
        "learnability": {
            "episode_return_std_bps": float(ep_returns.std(ddof=1)),
            "mean_abs_V": float(np.abs(V).mean()),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARMS:
        for regime in ("calm", "volatile"):
            for s in range(5):
                r = analyse(arm, f"ppo_{regime}_s{s}", regime)
                rows.append(r)
                c, p = r["critic"], r["policy"]
                print(f"{arm:9s} {r['run']:18s} obs{r['obs_dim']:3d} "
                      f"EV {c['explained_variance']:+.4f} | bias {c['bias']:+.4f} "
                      f"| corr(V,ret) {c['corr_V_return']:+.3f} "
                      f"| corr(V,inv) {c['corr_V_inventory']:+.3f} "
                      f"| ent {p['mean_state_entropy']:.3f}/{p['entropy_uniform']:.3f}",
                      flush=True)

    (OUT / "diag_learning_primary.json").write_text(json.dumps(rows, indent=1))

    print("\n--- SUMMARY: critic explained variance, primary track ---", flush=True)
    for arm in ARMS:
        for regime in ("calm", "volatile"):
            v = [r["critic"]["explained_variance"] for r in rows
                 if r["arm"] == arm and r["regime"] == regime]
            print(f"{arm:9s} {regime:9s} mean {np.mean(v):+.4f}  "
                  f"range {min(v):+.4f} to {max(v):+.4f}", flush=True)
    print(f"\nwrote {OUT / 'diag_learning_primary.json'}", flush=True)


if __name__ == "__main__":
    main()
