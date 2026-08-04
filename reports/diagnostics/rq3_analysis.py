"""RQ3 analysis: pool the attribution and policy-reduction outputs and test the contrasts.

RQ3, final wording: *which order-book features drive the agents' execution decisions, and does
this differ by market regime?*

This script does no new simulation. It reads the JSON written by `rq3_attribution.py`,
`rq3_attribution_frozen.py` and `rq3_policy_reduction.py`, plus the frozen-replay decomposition
already produced by the inversion study, and reduces them to the numbers the write-up quotes.

WHAT IS COMPUTED, AND WHY EACH CHOICE WAS MADE BEFORE LOOKING AT ANY RESULT
--------------------------------------------------------------------------
1. SHARE OF ATTRIBUTION PER FEATURE GROUP, per environment / algorithm / regime, as the mean
   over agents with the spread across agents. Shares, not raw |SHAP|, because agents differ in
   how much they move at all and raw magnitudes are not comparable across them.

2. THE CLOCK CONTRAST -- the pre-specified headline. `inventory remaining` and `time remaining`
   are the two observation entries that carry NO market information: together they are exactly
   the state a TWAP schedule needs. Every other entry is book or flow state. "Clock share" is
   therefore the fraction of the decision explained by where the agent is in its own schedule
   rather than by what the market is doing.

   THE DIMENSION BIAS RUNS AGAINST THIS FINDING, WHICH IS WHY THE RAW SHARE IS REPORTED.
   Group attribution is the SUM of |SHAP| over the dimensions in the group. The clock is 2 of
   27-28 dimensions; the bid and ask queue blocks are 10 each, so 20 of 27 dimensions are book
   depth alone. A large clock share is thus obtained despite the aggregation favouring the book,
   not because of it. The per-dimension normalisation is also reported, as the honest
   counter-view, and it moves the result further in the same direction.

3. THE PAID-SIGNAL CONTRAST. In the injected environment the observation contains the measured
   signal that a one-line rule monetised for 0.313 / 0.625 bps; in the frozen-replay observation
   the corresponding entry is `queue imbalance`, the same venue signal. Asking what share of the
   decision each receives answers a question nothing else in the study can: did the agents place
   weight on the specific feature that was demonstrably worth money? A share at or below the
   equal-attribution reference gives the null a NAMED mechanism rather than leaving it as an
   unexplained absence of skill.

   REFERENCE LINE: 1/obs_dim, the share a single dimension would receive if attribution were
   spread evenly. Both the injected signal and queue imbalance are single dimensions, so this is
   the correct like-for-like comparison and it is fixed by the observation layout, not chosen.

4. REGIME CONTRAST, with the two tracks treated differently BECAUSE THEY ARE DIFFERENT DESIGNS:
     - frozen replay: each agent trains on mixed data and is attributed TWICE, once per regime,
       on episodes of that regime. Regime is therefore WITHIN agent -> paired Wilcoxon.
     - reacting / injected: each agent is trained in ONE regime, so a calm-vs-volatile comparison
       is BETWEEN agents and confounds regime with the agent -> Mann-Whitney, and the confound is
       stated rather than glossed.
   Multiplicity: only the two contrasts named above (clock share, paid-signal share) are tested;
   every other feature group is reported descriptively. Holm correction across the tested family.

5. POLICY REDUCTION. Attribution says which inputs a policy responds to. It does NOT say how much
   of the behaviour is a response to anything: SHAP apportions whatever variation exists without
   reporting how much variation there is. The reduction regresses each agent's per-episode
   premium over TWAP on that of a FIXED 2.0x front-loading rule, and reports beta (effective
   dose), alpha (what a constant dose does not explain) and r (how completely one constant
   reproduces the pattern).

6. ACTION CONCENTRATION AND PER-STATE ENTROPY, because they can contradict 1-3 and must be
   allowed to. Per-state entropy, never the marginal: the two differ sharply and the difference
   has already misled this project once (live doc addendum (P)). DQN is a deterministic argmax
   with no action distribution, so per-state entropy is undefined for it and is reported as such
   rather than silently imputed as zero.

SCOPE OF THE PRIMARY TABLES. The injected campaign contains 10 base agents plus 18 tuning
variants (two alternative architectures and a reward-scaled variant). The primary tables use the
10 BASE agents so that every environment/algorithm cell has the same n; the variants are
reported separately as a robustness check. The reward-scaled variant had 0 audit-valid seeds in
Amendment A4.1 and is flagged wherever it appears.

Sources (absolute):
  .../scratch_hyperliquid/oxford_l4/rq3_attribution/*.json
  .../scratch_hyperliquid/l2_test_results/l2_inversion_stage3.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
RQ3 = S / "oxford_l4" / "rq3_attribution"
OUT = Path(__file__).resolve().parents[1] / "figures" / "qrm"
JSON_OUT = RQ3 / "rq3_analysis_summary.json"

CLOCK = ("inventory remaining", "time remaining")

# environment label -> (attribution files, reduction files)
FROZEN_SETS = ["runs", "runs_10s", "runs_10s_10min"]


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_reactive_attr(runs: str, algo: str) -> list[dict]:
    p = RQ3 / f"rq3_attribution_{runs}_{algo}.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_frozen_attr(ds: str, algo: str) -> list[dict]:
    p = RQ3 / f"rq3_attribution_frozen_{ds}_{algo}.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_reduction(runs: str, algo: str) -> list[dict]:
    p = RQ3 / f"rq3_reduction_{runs}_{algo}.json"
    return json.loads(p.read_text()) if p.exists() else []


# Behaviour audit: an agent whose deadline residual exceeds 10% did not complete the parent
# order and its cost is therefore not comparable with TWAP's. The project rule is that such
# agents are excluded from COST comparisons. It applies here: alpha, beta, r and mean_diff are
# all cost quantities. It does NOT apply to attribution shares, which describe what the policy
# responds to and are meaningful for an agent that executes badly -- those are reported over
# every agent, and the split is shown so the reader can see both.
AUDIT = {"runs_primary_v3": "step5_v3", "runs_signal_phaseD": "step5_signal_dev"}


def audit_valid(runs: str) -> dict[str, bool]:
    p = S / "oxford_l4" / AUDIT[runs] / "behaviour_audit.json"
    return {r["run"]: bool(r.get("valid", True)) for r in json.loads(p.read_text())}


BASE_RE = re.compile(r"^(ppo|dqn)_(calm|volatile)_s\d+$")


def is_base(run: str) -> bool:
    """Base agent, i.e. not one of the tuning variants (_v1a / _v1b / _v2).

    Matched against the FULL name pattern, not by substring. A first version tested
    `"_v" not in run` and silently deleted every `*_volatile_*` agent -- half the sample,
    with no error, leaving cells that looked complete at n=5 and were calm-only.
    """
    return bool(BASE_RE.match(run))


def is_degenerate(row: dict) -> bool:
    """A policy that emits ONE action in every sampled state.

    Kernel SHAP explains the map from observation to chosen action. If that map is constant
    over the sampled states, every attribution is identically zero -- not because no feature
    matters, but because there is no decision to attribute. Such rows carry
    `top_action_share == 1.0` and a total |SHAP| of exactly zero.

    They are EXCLUDED from the share means (averaging an undefined quantity in as zero would
    drag every feature's share toward zero and manufacture a result) and reported separately
    as a count, which is the honest reading: the policy is a constant.
    """
    return sum(row["share"].values()) < 1e-9


# --------------------------------------------------------------------------------------
# statistics -- small, explicit, no scipy dependency assumed at import time
# --------------------------------------------------------------------------------------
def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray):
    from scipy.stats import wilcoxon
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return float("nan"), 1.0
    s, p = wilcoxon(x, y)
    return float(s), float(p)


def mannwhitney(x, y):
    from scipy.stats import mannwhitneyu
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    s, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(s), float(p)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, NaNs passed through untouched."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)       # enforce monotonicity
        out[k] = running
    for k, v in pvals.items():
        out.setdefault(k, v)
    return out


def mean_sd(v):
    v = np.asarray([x for x in v if x is not None and x == x], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------
def reactive_rows(runs: str, algo: str, base_only: bool) -> list[dict]:
    """Normalise a reacting/injected attribution file to (agent, regime, shares)."""
    rows = []
    for r in load_reactive_attr(runs, algo):
        if base_only and not is_base(r["run"]):
            continue
        rows.append({"run": r["run"], "regime": r["regime"], "obs_dim": r["obs_dim"],
                     "share": r["group_share"], "n_states": r["n_states"],
                     "top_action": r["top_action"], "top_action_share": r["top_action_share"],
                     "mean_state_entropy": r["mean_state_entropy"],
                     "marginal_entropy": r["marginal_entropy"],
                     "entropy_uniform": r["entropy_uniform"],
                     "frac_states_confident": r["frac_states_confident"]})
    return rows


def frozen_rows(algo: str) -> list[dict]:
    """Normalise the frozen files. One row per (agent, dataset, regime); regime is WITHIN agent."""
    rows = []
    for ds in FROZEN_SETS:
        for r in load_frozen_attr(ds, algo):
            for regime, v in r["by_regime"].items():
                rows.append({"run": f"{ds}/{r['run']}", "dataset": ds, "regime": regime,
                             "obs_dim": len(r["features"]), "share": v["share"],
                             "n_states": v["n_states"], "top_action": v["top_action"],
                             "top_action_share": v["top_action_share"]})
    return rows


def clock_share(row: dict) -> float:
    return float(sum(row["share"].get(k, 0.0) for k in CLOCK))


def per_dim_shares(row: dict, group_sizes: dict[str, int]) -> dict[str, float]:
    """Share divided by the number of raw dimensions in the group, renormalised.

    The honest counter-view to the raw share: it removes the advantage a 10-dimensional
    queue block has over a 1-dimensional signal, and answers 'per input, which matters most'.
    """
    v = {g: row["share"].get(g, 0.0) / group_sizes[g] for g in group_sizes}
    tot = sum(v.values())
    return {g: (x / tot if tot > 0 else 0.0) for g, x in v.items()}


def group_sizes_reactive(obs_dim: int, injected: bool) -> dict[str, int]:
    K = (obs_dim - 7 - (1 if injected else 0)) // 2
    g = {"inventory remaining": 1, "time remaining": 1,
         "bid queue depth": K, "ask queue depth": K,
         "spread": 1, "own recent fills": 2, "order flow": 2}
    if injected:
        g["injected signal"] = 1
    assert sum(g.values()) == obs_dim, (sum(g.values()), obs_dim)
    return g


def summarise_cell(rows: list[dict], label: str, group_sizes_fn) -> dict:
    """Per-regime feature-share summary for one environment/algorithm cell.

    Degenerate (single-action) agents are counted and then dropped from the share means;
    see `is_degenerate`.
    """
    out = {"label": label, "n_rows": len(rows), "by_regime": {}}
    for regime in sorted({r["regime"] for r in rows}):
        allsub = [r for r in rows if r["regime"] == regime]
        sub = [r for r in allsub if not is_degenerate(r)]
        n_deg = len(allsub) - len(sub)
        if not sub:
            out["by_regime"][regime] = {"n_agents": 0, "n_degenerate": n_deg,
                                        "note": "every agent in this cell is a constant policy"}
            continue
        feats = sorted({f for r in sub for f in r["share"]})
        shares = {f: mean_sd([r["share"].get(f, 0.0) for r in sub]) for f in feats}
        pd_rows = [per_dim_shares(r, group_sizes_fn(r)) for r in sub]
        per_dim = {f: mean_sd([p.get(f, 0.0) for p in pd_rows]) for f in feats}
        out["by_regime"][regime] = {
            "n_agents": len(sub),
            "n_degenerate_excluded": n_deg,
            "obs_dim": int(sub[0]["obs_dim"]),
            "equal_share_reference": 1.0 / float(sub[0]["obs_dim"]),
            "share_mean_sd_n": shares,
            "per_dim_share_mean_sd_n": per_dim,
            "clock_share": mean_sd([clock_share(r) for r in sub]),
            "top_action_share": mean_sd([r["top_action_share"] for r in sub]),
            "n_states": mean_sd([r["n_states"] for r in sub]),
        }
    return out


def check_shares_sum_to_one(rows, label):
    """Non-degenerate rows must have shares summing to 1; degenerate rows sum to 0 by design."""
    bad = [(r["run"], s) for r in rows if not is_degenerate(r)
           and abs((s := sum(r["share"].values())) - 1.0) > 1e-6]
    if bad:
        print(f"  !! {label}: {len(bad)} non-degenerate rows whose shares do not sum to 1 "
              f"(e.g. {bad[0]})")
    return len(bad)


# --------------------------------------------------------------------------------------
def main() -> None:
    summary: dict = {"cells": {}, "tests": {}, "reduction": {}, "integrity": {}}
    print("=" * 96)
    print("RQ3 ANALYSIS")
    print("=" * 96)

    # ---------------- attribution cells ----------------
    cells = {}
    for algo in ("ppo", "dqn"):
        cells[("reacting", algo)] = reactive_rows("runs_primary_v3", algo, base_only=True)
        cells[("injected", algo)] = reactive_rows("runs_signal_phaseD", algo, base_only=True)
        cells[("frozen", algo)] = frozen_rows(algo)
    variants = {algo: [r for r in reactive_rows("runs_signal_phaseD", algo, base_only=False)
                       if not is_base(r["run"])] for algo in ("ppo", "dqn")}

    n_bad = 0
    print("\n--- row counts and integrity ---")
    for (env, algo), rows in cells.items():
        n_bad += check_shares_sum_to_one(rows, f"{env}/{algo}")
        regs = {}
        for r in rows:
            regs[r["regime"]] = regs.get(r["regime"], 0) + 1
        deg = sum(is_degenerate(r) for r in rows)
        print(f"  {env:<9} {algo.upper():<4} {len(rows):>3} rows   "
              f"{', '.join(f'{k} {v}' for k, v in sorted(regs.items()))}   "
              f"obs_dim {sorted({r['obs_dim'] for r in rows})}   "
              f"single-action (attribution undefined): {deg}")
        summary["integrity"][f"degenerate|{env}|{algo}"] = int(deg)
    print(f"  injected tuning variants held out of the primary tables: "
          f"PPO {len(variants['ppo'])}, DQN {len(variants['dqn'])}")
    summary["integrity"]["rows_with_bad_share_sum"] = n_bad
    summary["integrity"]["total_attribution_rows"] = sum(len(v) for v in cells.values()) \
        + sum(len(v) for v in variants.values())

    def gsz(env):
        if env == "frozen":
            return lambda r: {f: 1 for f in r["share"]}
        return lambda r: group_sizes_reactive(r["obs_dim"], env == "injected")

    for (env, algo), rows in cells.items():
        if rows:
            summary["cells"][f"{env}|{algo}"] = summarise_cell(rows, f"{env}/{algo}", gsz(env))

    # ---------------- printed share tables ----------------
    for (env, algo) in [("reacting", "ppo"), ("reacting", "dqn"),
                        ("injected", "ppo"), ("injected", "dqn"),
                        ("frozen", "ppo"), ("frozen", "dqn")]:
        c = summary["cells"].get(f"{env}|{algo}")
        if not c:
            continue
        print(f"\n--- attribution shares: {env.upper()} / {algo.upper()} ---")
        regimes = [r for r in sorted(c["by_regime"]) if c["by_regime"][r]["n_agents"] > 0]
        if not regimes:
            print("  every agent in this cell is a constant policy; attribution undefined")
            continue
        feats = sorted(c["by_regime"][regimes[0]]["share_mean_sd_n"],
                       key=lambda f: -c["by_regime"][regimes[0]]["share_mean_sd_n"][f][0])
        ref = c["by_regime"][regimes[0]]["equal_share_reference"]
        hdr = "  " + f"{'feature':<22}" + "".join(f"{r:>22}" for r in regimes)
        print(hdr)
        for f in feats:
            line = f"  {f:<22}"
            for r in regimes:
                m, sd, n = c["by_regime"][r]["share_mean_sd_n"][f]
                line += f"{m*100:>13.1f}% ±{sd*100:>5.1f}"
            print(line)
        line = f"  {'CLOCK (inv + time)':<22}"
        for r in regimes:
            m, sd, n = c["by_regime"][r]["clock_share"]
            line += f"{m*100:>13.1f}% ±{sd*100:>5.1f}"
        print(line)
        print("  " + f"{'n agents':<22}" +
              "".join(f"{c['by_regime'][r]['n_agents']:>21}" for r in regimes))
        print("  " + f"{'excluded (constant)':<22}" +
              "".join(f"{c['by_regime'][r]['n_degenerate_excluded']:>21}" for r in regimes))
        print(f"  equal-attribution reference for ONE dimension: {ref*100:.1f}%"
              f"   (obs_dim {c['by_regime'][regimes[0]]['obs_dim']})")

    # ---------------- pre-specified contrasts ----------------
    print("\n" + "=" * 96)
    print("PRE-SPECIFIED CONTRASTS")
    print("=" * 96)
    raw_p: dict[str, float] = {}
    tests: dict[str, dict] = {}

    # (a) clock share, regime contrast. Degenerate agents carry no attribution and are excluded;
    # for the paired frozen test that means an agent is dropped if EITHER regime is degenerate,
    # which is what pairing requires.
    for (env, algo), cell_rows in cells.items():
        rows = [r for r in cell_rows if not is_degenerate(r)]
        if not rows:
            continue
        calm = [clock_share(r) for r in rows if r["regime"] == "calm"]
        vol = [clock_share(r) for r in rows if r["regime"] == "volatile"]
        if not calm or not vol:
            continue
        if env == "frozen":                      # paired: same agent, both regimes
            by = {}
            for r in rows:
                by.setdefault(r["run"], {})[r["regime"]] = clock_share(r)
            pairs = [(v["calm"], v["volatile"]) for v in by.values()
                     if "calm" in v and "volatile" in v]
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            stat, p = wilcoxon_signed_rank(a, b)
            kind, n = "paired Wilcoxon", len(pairs)
        else:
            stat, p = mannwhitney(calm, vol)
            kind, n = "Mann-Whitney (BETWEEN agents: regime confounded with agent)", \
                      f"{len(calm)}v{len(vol)}"
        key = f"clock_share|{env}|{algo}"
        raw_p[key] = p
        tests[key] = {"test": kind, "n": n,
                      "calm_mean": float(np.mean(calm)), "volatile_mean": float(np.mean(vol)),
                      "stat": stat, "p_raw": p}

    # (b) paid-signal share vs the equal-attribution reference (one-sample), and by regime
    for (env, algo), cell_rows in cells.items():
        feat = {"injected": "injected signal", "frozen": "queue imbalance"}.get(env)
        rows = [r for r in cell_rows if not is_degenerate(r)]
        if not rows or feat is None:
            continue
        # INDEPENDENCE. On the frozen track each agent contributes TWO rows (one per regime)
        # and those rows are strongly correlated -- they come from the same weights. Testing
        # 70 rows as if they were 70 independent observations would roughly double the
        # effective sample and make the p-value anti-conservative. Average the regimes within
        # each agent first, so one agent is one observation. The reacting and injected agents
        # are trained in a single regime and already contribute one row each.
        if env == "frozen":
            per_agent: dict[str, list[float]] = {}
            for r in rows:
                per_agent.setdefault(r["run"], []).append(r["share"].get(feat, np.nan))
            vals = np.array([np.mean(v) for v in per_agent.values()], float)
        else:
            vals = np.array([r["share"].get(feat, np.nan) for r in rows], float)
        ref = 1.0 / rows[0]["obs_dim"]
        from scipy.stats import wilcoxon as _w
        try:
            _, p = _w(vals - ref)
        except Exception:
            p = float("nan")
        key = f"paid_signal|{env}|{algo}"
        raw_p[key] = p
        tests[key] = {"test": f"one-sample Wilcoxon vs equal-attribution reference {ref:.4f}",
                      "feature": feat, "n": int(len(vals)),
                      "mean": float(np.nanmean(vals)), "reference": ref,
                      "p_raw": p,
                      "calm_mean": float(np.nanmean([r["share"].get(feat, np.nan)
                                                     for r in rows if r["regime"] == "calm"])),
                      "volatile_mean": float(np.nanmean([r["share"].get(feat, np.nan)
                                                         for r in rows
                                                         if r["regime"] == "volatile"]))}

    adj = holm(raw_p)
    for k, t in tests.items():
        t["p_holm"] = adj[k]
        print(f"  {k:<28} {t['test']}")
        if "reference" in t:
            print(f"      mean {t['mean']*100:.2f}%  vs reference {t['reference']*100:.2f}%   "
                  f"calm {t['calm_mean']*100:.2f}%  volatile {t['volatile_mean']*100:.2f}%   "
                  f"p={t['p_raw']:.4f}  Holm p={t['p_holm']:.4f}   n={t['n']}")
        else:
            print(f"      calm {t['calm_mean']*100:.1f}%  volatile {t['volatile_mean']*100:.1f}%"
                  f"   p={t['p_raw']:.4f}  Holm p={t['p_holm']:.4f}   n={t['n']}")
    summary["tests"] = tests

    # ---------------- per-dimension view (the counter-view to the raw share) -------------
    print("\n" + "=" * 96)
    print("PER-DIMENSION ATTRIBUTION -- share divided by group size, renormalised")
    print("Raw group share is the SUM over the group's dimensions, so a 10-dimensional queue")
    print("block is favoured over a 1-dimensional signal purely by arithmetic. This view")
    print("removes that. Reference: 1/n_groups after renormalisation is NOT the right line --")
    print("the right line is 1/obs_dim on the RAW share, printed above. This table exists to")
    print("show which single INPUT matters most, not to replace the share.")
    print("=" * 96)
    for (env, algo) in [("reacting", "ppo"), ("reacting", "dqn"),
                        ("injected", "ppo"), ("injected", "dqn")]:
        c = summary["cells"].get(f"{env}|{algo}")
        if not c:
            continue
        regimes = [r for r in sorted(c["by_regime"]) if c["by_regime"][r]["n_agents"] > 0]
        pdv = c["by_regime"][regimes[0]]["per_dim_share_mean_sd_n"]
        order = sorted(pdv, key=lambda f: -pdv[f][0])
        print(f"\n  {env.upper()} / {algo.upper()}")
        for f in order:
            line = f"    {f:<22}"
            for r in regimes:
                m, sd, n = c["by_regime"][r]["per_dim_share_mean_sd_n"][f]
                line += f"{m*100:>13.1f}% ±{sd*100:>5.1f}"
            print(line)

    # ---------------- exploratory: per-feature regime contrast on the paired track --------
    print("\n" + "=" * 96)
    print("EXPLORATORY per-feature regime contrast, FROZEN track only (paired, same agent)")
    print("NOT pre-specified. Holm-corrected within this family and reported as exploratory;")
    print("it is included because the descriptive shifts are large enough that omitting them")
    print("would be selective reporting, not because it was planned.")
    print("=" * 96)
    explor: dict[str, dict] = {}
    for algo in ("ppo", "dqn"):
        rows = [r for r in cells[("frozen", algo)] if not is_degenerate(r)]
        by: dict[str, dict] = {}
        for r in rows:
            by.setdefault(r["run"], {})[r["regime"]] = r["share"]
        pairs = [v for v in by.values() if "calm" in v and "volatile" in v]
        feats = sorted({f for v in pairs for f in v["calm"]})
        praw = {}
        for f in feats:
            a = np.array([v["calm"].get(f, 0.0) for v in pairs])
            b = np.array([v["volatile"].get(f, 0.0) for v in pairs])
            _, p = wilcoxon_signed_rank(a, b)
            praw[f] = p
            explor[f"{algo}|{f}"] = {"calm": float(a.mean()), "volatile": float(b.mean()),
                                     "delta": float(b.mean() - a.mean()),
                                     "n_pairs": len(pairs), "p_raw": p}
        padj = holm(praw)
        print(f"\n  FROZEN / {algo.upper()}  ({len(pairs)} paired agents)")
        for f in sorted(feats, key=lambda x: praw[x]):
            e = explor[f"{algo}|{f}"]
            e["p_holm"] = padj[f]
            star = "  <-- survives Holm" if padj[f] < 0.05 else ""
            print(f"    {f:<22} calm {e['calm']*100:>5.1f}%  volatile {e['volatile']*100:>5.1f}%"
                  f"  delta {e['delta']*100:+5.1f}pp   p={e['p_raw']:.4f}  "
                  f"Holm p={padj[f]:.4f}{star}")
    summary["exploratory_regime_by_feature"] = explor

    # ---------------- policy reduction ----------------
    print("\n" + "=" * 96)
    print("POLICY REDUCTION -- how much of the behaviour is one constant?")
    print("=" * 96)
    print("AUDIT-VALID AGENTS ONLY. alpha/beta/r/mean_diff are cost quantities, and an agent")
    print("that failed the behaviour audit did not finish the parent order, so its cost is not")
    print("comparable with TWAP's. Counts of what was dropped are printed per cell.")
    red = {}
    for env, runs in (("reacting", "runs_primary_v3"), ("injected", "runs_signal_phaseD")):
        valid = audit_valid(runs)
        for algo in ("ppo", "dqn"):
            allrows = [r for r in load_reduction(runs, algo) if is_base(r["run"])]
            rows = [r for r in allrows if valid.get(r["run"], True)]
            n_drop = len(allrows) - len(rows)
            if n_drop:
                print(f"  [{env}/{algo}] {n_drop} of {len(allrows)} base agents dropped: "
                      f"behaviour-audit invalid")
            if not rows:
                print(f"  [{env}/{algo}] NO audit-valid base agents -- cell reported as such")
                red[f"{env}|{algo}|ALL"] = {"n": 0, "n_dropped_audit_invalid": n_drop,
                                            "note": "no audit-valid agents in this cell"}
                continue
            for regime in ("calm", "volatile", "ALL"):
                sub = rows if regime == "ALL" else [r for r in rows if r["regime"] == regime]
                if not sub:
                    continue
                r_abs = mean_sd([abs(x["corr_with_probe"]) for x in sub])
                red[f"{env}|{algo}|{regime}"] = {
                    "n": len(sub),
                    "n_dropped_audit_invalid": n_drop if regime == "ALL" else None,
                    "mean_abs_r": r_abs,
                    "mean_r": mean_sd([x["corr_with_probe"] for x in sub]),
                    "beta": mean_sd([x["beta_frontload_dose"] for x in sub]),
                    "alpha_bps": mean_sd([x["alpha_bps"] for x in sub]),
                    "mean_diff_bps": mean_sd([x["mean_diff_bps"] for x in sub]),
                    "top_action_share": mean_sd([x["top_action_share"] for x in sub]),
                    "mean_state_entropy": mean_sd([x["mean_state_entropy"] for x in sub]),
                    "entropy_uniform": sub[0]["entropy_uniform"],
                }
                if regime == "ALL":
                    d = red[f"{env}|{algo}|ALL"]
                    # DQN is a deterministic argmax and has no action distribution, so
                    # per-state entropy does not exist for it. Say so; do not print NaN
                    # and do not impute zero.
                    h = ("n/a (deterministic argmax)" if d["mean_state_entropy"][2] == 0
                         else f"{d['mean_state_entropy'][0]:.2f}/{d['entropy_uniform']:.2f}")
                    print(f"  {env:<9} {algo.upper():<4} n={len(sub):<3} "
                          f"|r| {d['mean_abs_r'][0]:.3f}±{d['mean_abs_r'][1]:.3f}   "
                          f"beta {d['beta'][0]:+.3f}±{d['beta'][1]:.3f}   "
                          f"alpha {d['alpha_bps'][0]:+.4f}±{d['alpha_bps'][1]:.4f} bps   "
                          f"top-action {d['top_action_share'][0]*100:.0f}%   "
                          f"per-state H {h}")

    # Frozen-replay reduction: ALL THREE datasets, validation split, from
    # rq3_reduction_frozen.py. The inversion study's stage 3 measured the same thing on
    # runs_10s alone; that file is still read below purely as a cross-check of the two
    # independent implementations, not as a second source of quoted numbers.
    frozen = json.loads((RQ3 / "rq3_reduction_frozen_all.json").read_text())
    for algo in ("ppo", "dqn", "ALL"):
        sub = frozen if algo == "ALL" else [r for r in frozen if r["algo"] == algo]
        red[f"frozen|{algo}|validation"] = {
            "n": len(sub),
            "mean_abs_r": mean_sd([abs(r["corr_with_probe"]) for r in sub]),
            "beta": mean_sd([r["beta_frontload_dose"] for r in sub]),
            "alpha_bps": mean_sd([r["alpha_bps"] for r in sub]),
            "mean_diff_bps": mean_sd([r["mean_diff_bps"] for r in sub]),
            "datasets": sorted({r["dataset"] for r in sub}),
        }
        d = red[f"frozen|{algo}|validation"]
        print(f"  frozen    {algo.upper():<4} n={d['n']:<3} |r| {d['mean_abs_r'][0]:.3f}"
              f"±{d['mean_abs_r'][1]:.3f}   beta {d['beta'][0]:+.3f}±{d['beta'][1]:.3f}   "
              f"alpha {d['alpha_bps'][0]:+.4f}±{d['alpha_bps'][1]:.4f} bps   "
              f"[{len(d['datasets'])} datasets, validation]")

    st3 = json.loads((S / "l2_test_results" / "l2_inversion_stage3.json").read_text())
    mine = {r["run"]: r for r in frozen if r["dataset"] == "runs_10s"}
    dd = [abs(mine[n]["corr_with_probe"] - v["corr_with_probe"])
          for n, v in st3["validation"].items() if n in mine]
    print(f"  cross-check vs the inversion study's independent implementation on its one "
          f"dataset: {len(dd)} agents, max |delta r| = {max(dd):.2e}"
          f"{'' if max(dd) < 1e-6 else '   !! DISAGREEMENT -- do not quote'}")
    summary["frozen_reduction_crosscheck"] = {"n": len(dd), "max_abs_delta_r": float(max(dd))}
    summary["reduction"] = red

    # ---------------- action concentration ----------------
    print("\n--- action concentration and per-state entropy (base agents) ---")
    for env, runs in (("reacting", "runs_primary_v3"), ("injected", "runs_signal_phaseD")):
        for algo in ("ppo", "dqn"):
            rows = [r for r in load_reduction(runs, algo) if is_base(r["run"])]
            if not rows:
                continue
            for regime in ("calm", "volatile"):
                sub = [r for r in rows if r["regime"] == regime]
                if not sub:
                    continue
                ts = mean_sd([r["top_action_share"] for r in sub])
                se = mean_sd([r["mean_state_entropy"] for r in sub])
                mg = mean_sd([r["marginal_entropy"] for r in sub])
                acts = sorted({r["top_action"] for r in sub})
                se_txt = ("undefined (deterministic argmax)" if se[2] == 0
                          else f"{se[0]:.2f}/{sub[0]['entropy_uniform']:.2f}")
                print(f"  {env:<9} {algo.upper():<4} {regime:<9} "
                      f"top-action share {ts[0]*100:>5.1f}%  modal action(s) {acts}  "
                      f"per-state H {se_txt}  marginal H {mg[0]:.2f}")

    # ---------------- does the residual alpha differ from zero? --------------------------
    print("\n" + "=" * 96)
    print("ALPHA vs ZERO -- what the agent achieves that a CONSTANT front-loading dose does not")
    print("This is the quantity that decides whether state-dependence is worth anything.")
    print("=" * 96)
    from scipy.stats import wilcoxon as _wz
    alpha_tests = {}
    for env, runs in (("reacting", "runs_primary_v3"), ("injected", "runs_signal_phaseD")):
        valid = audit_valid(runs)
        for algo in ("ppo", "dqn"):
            rows = [r for r in load_reduction(runs, algo)
                    if is_base(r["run"]) and valid.get(r["run"], True)]
            if len(rows) < 3:
                print(f"  {env}|{algo:<4} only {len(rows)} audit-valid agents -- not tested")
                continue
            a = np.array([r["alpha_bps"] for r in rows], float)
            try:
                _, p = _wz(a)
            except Exception:
                p = float("nan")
            alpha_tests[f"{env}|{algo}"] = {"n": len(a), "mean": float(a.mean()),
                                            "sd": float(a.std(ddof=1)), "p_raw": float(p)}
    padj = holm({k: v["p_raw"] for k, v in alpha_tests.items()})
    for k, v in alpha_tests.items():
        v["p_holm"] = padj[k]
        verdict = ("indistinguishable from zero" if padj[k] >= 0.05
                   else "DIFFERENT from zero -- inspect before quoting")
        print(f"  {k:<16} n={v['n']:<3} alpha {v['mean']:+.4f} ± {v['sd']:.4f} bps   "
              f"p={v['p_raw']:.4f}  Holm p={padj[k]:.4f}   {verdict}")
    summary["alpha_vs_zero"] = alpha_tests

    # ---------------- does attending to the paid signal predict performance? -------------
    print("\n" + "=" * 96)
    print("DOES ATTENTION PAY? attribution share on the paid signal vs the agent's own edge")
    print("Exploratory, n=10 per cell. Attribution and cost are measured on DIFFERENT episode")
    print("seeds (9.0e6 vs 9.5e6) for the same agent, so this is a cross-agent association,")
    print("not a within-episode one. A positive slope would mean the agents that looked at the")
    print("signal more did better; the absence of one is what the null predicts.")
    print("=" * 96)
    from scipy.stats import spearmanr
    pays = {}
    valid = audit_valid("runs_signal_phaseD")
    for algo in ("ppo", "dqn"):
        attr = {r["run"]: r["group_share"].get("injected signal", np.nan)
                for r in load_reactive_attr("runs_signal_phaseD", algo) if is_base(r["run"])}
        red_rows = [r for r in load_reduction("runs_signal_phaseD", algo)
                    if is_base(r["run"]) and valid.get(r["run"], True)]
        x = np.array([attr.get(r["run"], np.nan) for r in red_rows], float)
        y = np.array([r["mean_diff_bps"] for r in red_rows], float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 4:
            rho, p = spearmanr(x[ok], y[ok])
            # DIRECTION, stated explicitly because it is easy to report backwards.
            # y is (agent - TWAP) cost, so NEGATIVE y = cheaper = better. A NEGATIVE rho
            # therefore means more attention on the signal goes with CHEAPER execution --
            # the direction in which attention would be paying off.
            direction = ("more attention <-> cheaper execution" if rho < 0
                         else "more attention <-> more expensive execution")
            pays[algo] = {"n": int(ok.sum()), "spearman_rho": float(rho), "p": float(p),
                          "direction": direction}
            print(f"  injected/{algo.upper():<4} n={ok.sum():<3} Spearman rho={rho:+.3f}  "
                  f"p={p:.3f}   {direction}"
                  f"{'' if p < 0.05 else '  -- NOT significant'}")
    summary["attention_vs_edge"] = pays

    JSON_OUT.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {JSON_OUT}")


if __name__ == "__main__":
    main()
