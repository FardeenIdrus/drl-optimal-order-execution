"""Post-hoc learning diagnostic on the TRAINED PPO models (no retraining).

Answers, per agent: is the critic learning anything, and are the action-value
differences it implies large enough to learn a policy from?

Measured per agent (dev seeds, injected env, deterministic policy):
  A. CRITIC QUALITY
     - explained variance of V(s) against the realised discounted return-to-go
     - bias (mean error) vs noise (std of error): is the error systematic?
     - correlation of V with remaining inventory (the "easy" part of the return)
  B. POLICY DECISIVENESS
     - per-state entropy of the action distribution (vs uniform = ln 7)
     - logit spread within a state (how strongly it prefers its top action)
     - marginal action distribution across states (the earlier audit number)
  C. THE DECISIVE QUANTITY
     - spread of the empirical action-value differences the agent would need to
       resolve, expressed against the per-episode return noise. If within-state
       action differences are far below the noise, no policy gradient can learn
       them regardless of how well the critic fits.
  D. SIGNAL RESPONSE
     - d(chosen pace)/d(signal) and d(V)/d(signal): does either object respond to
       feature 27 (0-indexed) at all, and with which sign?

Diagnostic only; no registered rule touched. Output: JSON + console table.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from stable_baselines3 import PPO
from execution.qrm.ac_vwap import make_env
from execution.qrm.reactive_env import ACTIONS

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = S / "step5_signal_dev" / "diagnostics_postnull"
GAMMA = 0.995 ** (1.0 / 60.0)
SIG_IDX = 27
N_EP = 40                      # episodes per agent for return-to-go estimation
SWEEP = np.array([-3., -2., -1., 0., 1., 2., 3.])


def rollout(model, env, seeds):
    """Deterministic rollout; keep obs, rewards, and per-step V(s)."""
    OBS, REW, EPI = [], [], []
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
        EPI.append(len(r_ep))
    return OBS, REW, EPI


def returns_to_go(rews):
    g = np.zeros_like(rews)
    acc = 0.0
    for i in range(len(rews) - 1, -1, -1):
        acc = rews[i] + GAMMA * acc
        g[i] = acc
    return g


def analyse(run, regime):
    model = PPO.load(str(S / "runs_signal_phaseD" / run / "model.zip"), device="cpu")
    env = make_env(S, regime, inject=True)
    seeds = list(range(18_000_000, 18_000_000 + N_EP))
    OBS, REW, _ = rollout(model, env, seeds)

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
    inv = allobs[:, 0]                                   # remaining inventory fraction
    # C: within-state preference gaps, vs the noise the learner must beat
    gaps = np.sort(logits, axis=1)
    top2 = gaps[:, -1] - gaps[:, -2]                     # logit margin of the top action
    ep_returns = np.array([returns_to_go(r)[0] for r in REW])

    # D: response of policy and critic to the signal feature alone
    bank = allobs[:: max(1, len(allobs) // 400)].copy()
    paces, vs = [], []
    for s in SWEEP:
        b = bank.copy(); b[:, SIG_IDX] = s
        tb = torch.as_tensor(b)
        with torch.no_grad():
            a, _ = model.predict(b, deterministic=True)
            vv = model.policy.predict_values(tb).numpy().ravel()
        paces.append(float(np.mean([ACTIONS[int(x)] for x in np.atleast_1d(a)])))
        vs.append(float(vv.mean()))
    d_pace = (paces[-1] - paces[0]) / 6.0
    d_v = (vs[-1] - vs[0]) / 6.0

    return {
        "run": run, "regime": regime,
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
        "signal_response": {
            "d_pace_per_sd": d_pace, "d_V_per_sd": d_v,
            "pace_curve": paces, "V_curve": vs,
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in ("calm", "volatile"):
        for s in range(5):
            r = analyse(f"ppo_{regime}_s{s}", regime)
            rows.append(r)
            c, p, d = r["critic"], r["policy"], r["signal_response"]
            print(f"{r['run']:18s} EV {c['explained_variance']:+.3f} | bias {c['bias']:+.4f} "
                  f"| corr(V,ret) {c['corr_V_return']:+.3f} | corr(V,inv) {c['corr_V_inventory']:+.3f} "
                  f"| ent {p['mean_state_entropy']:.3f}/{p['entropy_uniform']:.3f} "
                  f"| margin {p['mean_top2_logit_margin']:.2f} | dpace/sd {d['d_pace_per_sd']:+.4f} "
                  f"| dV/sd {d['d_V_per_sd']:+.4f}", flush=True)
    (OUT / "diag_learning.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote diag_learning.json")


if __name__ == "__main__":
    main()
