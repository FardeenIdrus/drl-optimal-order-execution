"""RQ3: which order-book features drive the agents' execution decisions, and does this
differ by market regime?

TWO INDEPENDENT MEASUREMENTS, deliberately. Either alone is arguable; together they are hard
to dismiss, and they can disagree, which is the point of running both.

  (1) FEATURE ATTRIBUTION on the trained policy. For each agent we collect states visited
      under its own deterministic policy, then attribute its action choice to the observation
      features using SHAP. Reported as mean |SHAP| per feature group, normalised to shares so
      agents with different action scales are comparable, and split by regime.

  (2) POLICY REDUCTION. Independently, we ask how much of the agent's behaviour a single
      number explains: regress its per-episode cost against that of a fixed front-loading
      rule, and report the correlation and the intercept. This is the measurement already
      completed on the frozen-replay track (mean |r| 0.95); here it is extended to the
      reacting and injected environments.

WHY BOTH. (1) says which inputs the policy responds to. (2) says how much of the resulting
behaviour is a constant rather than a response to anything. A policy can have non-zero
attributions and still be almost entirely a constant, because SHAP attributes the variation
that exists without saying how much variation there is. Reporting only (1) would overstate
how state-dependent these policies are.

FEATURE GROUPING. The raw observation is 28-29 dimensional and 2K of those entries are
individual queue-size slots, which are not individually interpretable and would fragment the
attribution across dozens of near-duplicate columns. Features are therefore grouped into the
seven economically meaningful blocks the environment actually constructs (see reactive_env
._observe). Grouping is done BEFORE looking at any attribution, and the grouping is fixed here.

Scope: the reacting and injected environments, where regime labels exist and where the central
claim lives. The frozen-replay track has no calm/volatile split and is covered by (2) alone.

Outputs: rq3_attribution.json (per agent, per feature group, per regime) + a run log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = S / "rq3_attribution"


def feature_groups(obs_dim: int, K: int, injected: bool, pva: bool) -> dict[str, list[int]]:
    """Index map from the observation layout in reactive_env._observe.

    Layout: [remaining_frac, time_frac] + log1p(state)[2K] + [spread, last_fill,
             cum_fill, flow_market, flow_signed] (+ [signal] if injected) (+ [pva] if pva).
    The 2K state block is bid queues then ask queues, K each.
    """
    g: dict[str, list[int]] = {}
    g["inventory remaining"] = [0]
    g["time remaining"] = [1]
    g["bid queue depth"] = list(range(2, 2 + K))
    g["ask queue depth"] = list(range(2 + K, 2 + 2 * K))
    base = 2 + 2 * K
    g["spread"] = [base]
    g["own recent fills"] = [base + 1, base + 2]
    g["order flow"] = [base + 3, base + 4]
    nxt = base + 5
    if injected:
        g["injected signal"] = [nxt]; nxt += 1
    if pva:
        g["price vs arrival"] = [nxt]; nxt += 1
    total = sum(len(v) for v in g.values())
    if total != obs_dim:
        raise SystemExit(f"feature map covers {total} of {obs_dim} dims -- layout changed?")
    return g


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


def collect_states(core, model, seeds, max_states):
    """States actually visited under the agent's own deterministic policy."""
    from execution.qrm.reactive_baselines import run_episodes
    seen = []

    def policy(env, obs):
        if len(seen) < max_states:
            seen.append(np.asarray(obs, dtype=np.float32))
        a, _ = model.predict(obs, deterministic=True)
        return int(a)

    run_episodes(core, policy, list(seeds))
    return np.array(seen[:max_states], dtype=np.float32)


def attribute(model, states, background, nsamples):
    """Mean |SHAP| per raw observation dimension, over the policy's chosen action.

    The explained function returns the action INDEX. Attribution is therefore over the
    decision the agent actually takes, which is the quantity RQ3 asks about.
    """
    import shap
    def f(x):
        a, _ = model.predict(np.asarray(x, dtype=np.float32), deterministic=True)
        return np.asarray(a, dtype=float).reshape(-1)
    expl = shap.KernelExplainer(f, background)
    sv = expl.shap_values(states, nsamples=nsamples, silent=True)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]
    return np.abs(sv).mean(axis=0)



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
    ap.add_argument("--runs", required=True, help="runs dir under oxford_l4")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--obs-price-vs-arrival", action="store_true")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--background", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N agents only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result file (off by default)")
    args = ap.parse_args()

    # Reuse the TRAINING module's own environment builder and kernel loader rather than
    # reconstructing them here: any divergence would mean attributing decisions made in one
    # environment to features of a different one.
    from execution.qrm.reactive_env import ACTIONS
    from execution.qrm.train_reactive import _core, _load_kernel
    from stable_baselines3 import PPO, DQN

    runs = sorted(d for d in (S / args.runs).iterdir()
                  if d.is_dir() and (d / "model.zip").exists()
                  and d.name.startswith(args.algo))
    if args.limit:
        runs = runs[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    # filename MUST carry the algo: without it the second pass over the same
    # runs directory silently overwrites the first (it did -- 28 PPO rows lost).
    out_path = guarded_out(OUT / f"rq3_attribution_{args.runs}_{args.algo}{args.tag}.json", args.force)
    print(f"{len(runs)} agents from {args.runs} (inject={args.inject}, "
          f"pva={args.obs_price_vs_arrival})", flush=True)

    results, t0 = [], time.time()
    for i, rd in enumerate(runs, 1):
        meta = json.loads((rd / "meta.json").read_text())
        regime = meta["regime"]
        kernel = _load_kernel(S, regime) if args.inject else None
        # meta.json for the primary campaign predates the env_steps field. That
        # campaign used the trained default of 300 s, confirmed by criteria section 3
        # ("Cadence 1 s; horizon 300 s").
        env_steps = int(meta.get("env_steps", 300))
        core = _core(S, regime, meta["order_btc"], env_steps=env_steps,
                     kernel=kernel, obs_pva=args.obs_price_vs_arrival)
        model = (PPO if args.algo == "ppo" else DQN).load(str(rd / "model.zip"), device="cpu")

        seeds = range(9_000_000, 9_000_000 + args.n_episodes)
        states = collect_states(core, model, seeds, args.max_states)
        if len(states) < args.background + 10:
            print(f"  SKIP {rd.name}: only {len(states)} states", flush=True)
            continue
        rng = np.random.default_rng(0)
        bg = states[rng.choice(len(states), args.background, replace=False)]
        mean_abs = attribute(model, states, bg, args.nsamples)

        groups = feature_groups(states.shape[1], core.K, args.inject,
                                args.obs_price_vs_arrival)
        gv = {g: float(mean_abs[idx].sum()) for g, idx in groups.items()}
        tot = sum(gv.values())
        share = {g: (v / tot if tot > 0 else 0.0) for g, v in gv.items()}
        acts, _ = model.predict(states, deterministic=True)
        acts = np.asarray(acts).reshape(-1)
        counts = np.bincount(acts, minlength=len(ACTIONS)).astype(float)
        p = counts / counts.sum()
        est = entropy_stats(model, states, len(ACTIONS))

        results.append({
            "run": rd.name, "algo": args.algo, "regime": regime,
            "n_states": int(len(states)), "obs_dim": int(states.shape[1]),
            "group_mean_abs_shap": gv, "group_share": share,
            "top_action": float(ACTIONS[int(np.argmax(counts))]),
            "top_action_share": float(p.max()),
            **est,
        })
        top = max(share, key=share.get)
        se = est["mean_state_entropy"]
        print(f"  [{i}/{len(runs)}] {rd.name:<22} {regime:<9} top={top} ({share[top]:.1%})  "
              f"per-state entropy {se if se is None else round(se, 2)}"
              f"/{est['entropy_uniform']:.2f}  marginal {est['marginal_entropy']:.2f}",
              flush=True)
        out_path.write_text(json.dumps(results, indent=1))   # checkpoint every agent
        del core, model

    print(f"wrote {out_path}  ({len(results)} agents, {(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
