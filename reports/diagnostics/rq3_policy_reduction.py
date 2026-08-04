"""RQ3, second measurement: how much of an agent's behaviour is a single constant?

Companion to rq3_attribution.py. Feature attribution says which inputs a policy responds to;
it does NOT say how much of the resulting behaviour is a response to anything at all. A policy
that always requests the same pace multiple still yields well-defined SHAP values, because
SHAP apportions whatever variation exists without reporting how much variation there is.

This script supplies the missing quantity. For each agent, in its own environment, over the
same episodes:

    agent cost, TWAP cost, and the cost of a FIXED front-loading rule (the 2.0x pace multiple,
    a member of the agent's own action set) are measured per episode. Then

        (agent - TWAP)_e  =  alpha + beta * (frontload - TWAP)_e

    beta  = the agent's effective front-loading dose -- how much of a fixed deviation from
            uniform pacing its behaviour amounts to.
    alpha = what the agent achieves that a constant deviation does not explain.
    r     = how completely one constant reproduces its per-episode pattern.

This is the identical decomposition already applied to the frozen-replay track, where it gave
mean |r| = 0.95 across thirty agents with an intercept indistinguishable from zero. Running it
in the reacting and injected environments extends that from one environment to three.

ALSO REPORTED, because it is the cheapest direct evidence of state-dependence: the share of
decisions spent on the agent's single most-used action, and the entropy of its action
distribution. A policy that spends 90% of its decisions on one action is not responding to
much, whatever the attribution says.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = S / "rq3_attribution"


def entropy_stats(model, states, n_actions):
    """PER-STATE action entropy, not the entropy of the marginal action distribution.

    These differ sharply and the difference has already misled this project once (live doc
    addendum (P)): a policy can be CONFIDENT in every individual state while its marginal
    distribution over a run looks near-uniform, because it picks different confident actions
    in different states. The marginal entropy then reads as "the policy is indecisive" when
    the opposite is true. Per-state entropy is the quantity that answers whether the policy is
    responding to state.

    PPO exposes a categorical distribution, so per-state entropy is exact. DQN is a
    deterministic argmax over Q-values and has no action distribution; for DQN this returns
    NaN and the marginal is reported alone, labelled as such.
    """
    import numpy as np, torch
    out = {"marginal_entropy": None, "mean_state_entropy": None,
           "entropy_uniform": float(np.log(n_actions)), "frac_states_confident": None}
    a, _ = model.predict(states, deterministic=True)
    a = np.asarray(a).reshape(-1)
    p = np.bincount(a, minlength=n_actions).astype(float); p /= p.sum()
    out["marginal_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    try:
        obs_t = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()
        ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
        out["mean_state_entropy"] = float(ent.mean())
        out["frac_states_confident"] = float((probs.max(axis=1) > 0.5).mean())
    except Exception:
        pass                                  # DQN: no action distribution
    return out



def guarded_out(path, force: bool):
    """Refuse to silently overwrite an existing result file.

    An earlier version of this script built its output name without the algorithm, so a
    second pass over the same runs directory OVERWROTE the first and 28 rows were lost
    without any warning. The name is fixed, but a name can be got wrong again; this makes
    the failure loud instead of silent. Pass --force to overwrite deliberately.
    """
    if path.exists() and not force:
        raise SystemExit(
            f"REFUSING TO OVERWRITE an existing result file:\n    {path}\n"
            f"Delete it, change --tag, or pass --force if overwriting is intended.")
    return path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--obs-price-vs-arrival", action="store_true")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result file (off by default)")
    args = ap.parse_args()

    from execution.qrm.reactive_env import ACTIONS
    from execution.qrm.train_reactive import _core, _load_kernel
    from execution.qrm.reactive_baselines import adaptive_twap, run_episodes
    from stable_baselines3 import PPO, DQN

    A_MAX = len(ACTIONS) - 1                      # the 2.0x front-loading action
    seeds = list(range(9_500_000, 9_500_000 + args.n_episodes))

    runs = sorted(d for d in (S / args.runs).iterdir()
                  if d.is_dir() and (d / "model.zip").exists()
                  and d.name.startswith(args.algo))
    if args.limit:
        runs = runs[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    # filename MUST carry the algo: without it the second pass over the same
    # runs directory silently overwrites the first (it did -- 28 PPO rows lost).
    out_path = guarded_out(OUT / f"rq3_reduction_{args.runs}_{args.algo}{args.tag}.json", args.force)
    print(f"{len(runs)} agents from {args.runs} (inject={args.inject})", flush=True)

    # TWAP and the fixed probe are agent-independent within a (regime, order, steps) cell.
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    results, t0 = [], time.time()

    for i, rd in enumerate(runs, 1):
        meta = json.loads((rd / "meta.json").read_text())
        regime = meta["regime"]
        # meta.json for the primary campaign predates the env_steps field. That
        # campaign used the trained default of 300 s, confirmed by criteria section 3
        # ("Cadence 1 s; horizon 300 s").
        env_steps = int(meta.get("env_steps", 300))
        key = (regime, meta["order_btc"], env_steps, bool(args.inject),
               bool(args.obs_price_vs_arrival))
        kernel = _load_kernel(S, regime) if args.inject else None
        core = _core(S, regime, meta["order_btc"], env_steps=env_steps,
                     kernel=kernel, obs_pva=args.obs_price_vs_arrival)

        if key not in cache:
            twap = run_episodes(core, adaptive_twap, seeds)["cost_bps"]
            probe = run_episodes(core, lambda e, o: A_MAX, seeds)["cost_bps"]
            cache[key] = (np.asarray(twap, float), np.asarray(probe, float))
        twap, probe = cache[key]
        probe_prem = probe - twap                       # what pure front-loading earns

        model = (PPO if args.algo == "ppo" else DQN).load(str(rd / "model.zip"), device="cpu")
        acts: list[int] = []
        states_seen: list = []

        def policy(env, obs):
            a, _ = model.predict(obs, deterministic=True)
            acts.append(int(a))
            if len(states_seen) < 400:          # sample for the per-state entropy
                states_seen.append(np.asarray(obs, dtype=np.float32))
            return int(a)

        agent = np.asarray(run_episodes(core, policy, seeds)["cost_bps"], float)
        d = agent - twap

        if np.std(probe_prem) > 1e-12:
            beta, alpha = np.polyfit(probe_prem, d, 1)
            r = float(np.corrcoef(probe_prem, d)[0, 1])
        else:
            beta = alpha = r = float("nan")

        counts = np.bincount(np.array(acts), minlength=len(ACTIONS)).astype(float)
        p = counts / counts.sum()
        results.append({
            "run": rd.name, "algo": args.algo, "regime": regime,
            "n_episodes": len(seeds),
            "mean_diff_bps": float(d.mean()),
            "beta_frontload_dose": float(beta),
            "alpha_bps": float(alpha),
            "corr_with_probe": r,
            "probe_mean_bps": float(probe_prem.mean()),
            "top_action": float(ACTIONS[int(np.argmax(counts))]),
            "top_action_share": float(p.max()),
            **entropy_stats(model, np.array(states_seen, dtype=np.float32), len(ACTIONS)),
        })
        print(f"  [{i}/{len(runs)}] {rd.name:<22} {regime:<9} mean {d.mean():+.4f}  "
              f"beta {beta:+.3f}  alpha {alpha:+.4f}  r {r:+.3f}  "
              f"top {ACTIONS[int(np.argmax(counts))]}x @{p.max():.0%}", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
        del core, model

    print(f"wrote {out_path}  ({len(results)} agents, {(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
