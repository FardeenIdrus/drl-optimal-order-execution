"""Results-section TABLES for the LaTeX report -> booktabs .tex files.

Emits ready-to-\\input LaTeX (needs \\usepackage{booktabs} in the preamble). Every number is
read from the source-of-record JSONs (never hand-typed). Currently builds the tables whose
data already exists; grid (T4), descriptive-stats (T6), env-validation (T5) and L2 (T7) are
added as their sources land.

Run:  .venv/bin/python reports/tables/make_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent


def esc(s: str) -> str:
    """Escape LaTeX-special chars in a text cell."""
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _load(dirname: str):
    j = json.load(open(S / dirname / "judgement.json"))
    a = {e["run"]: e for e in json.load(open(S / dirname / "behaviour_audit.json"))}
    return j, a


def _tag_of(run: str, seed: int) -> str:
    m = re.search(rf"_s{seed}(_.+)?$", run)
    return (m.group(1) or "") if m else ""


def write(name: str, body: str):
    (OUT / name).write_text(body)
    print(f"wrote {name}")


# ---------------------------------------------------------------- T1
def t1_primary():
    j, a = _load("step5_v3")
    rows = sorted(j["per_run"], key=lambda r: (r["algo"], r["regime"], r["seed"]))
    lines = [
        r"% Auto-generated from step5_v3/judgement.json + behaviour_audit.json. Do not edit by hand.",
        r"\begin{tabular}{llrrrrc}",
        r"\toprule",
        r"algo & regime & seed & vs fixed-TWAP & $p_{\mathrm{fix}}$ & vs adaptive-TWAP & valid \\",
        r" & & & (bps) & & (bps) & \\",
        r"\midrule",
    ]
    for r in rows:
        valid = "yes" if a[r["run"]]["valid"] else r"\textbf{NO}"
        lines.append(
            f"{r['algo']} & {r['regime']} & {r['seed']} & "
            f"{r['mean_vs_fixed_bps']:+.4f} & {r['p_fixed']:.3f} & "
            f"{r['mean_vs_adaptive_bps']:+.4f} & {valid} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t1_primary_campaign.tex", "\n".join(lines))


# ---------------------------------------------------------------- T2
def t2_tuning():
    j, a = _load("step5_selection_v3")
    purpose = {"": "base config", "_v1a": "network 64x64", "_v1b": "network 128x128",
               "_v2": "reward x100", "_v3a": "learning rate 1e-3", "_v3b": "learning rate 1e-4",
               "_v4a": "ent\\_coef 0.0", "_v4b": "ent\\_coef 0.05", "_v5": "n\\_steps 8192",
               "_v6": "reward x100 @ 10M", "_v6b": "lr 1e-4 @ 10M",
               "_w3a": "net128 + reward x100", "_w3b": "lr 1e-4 + reward x100",
               "_d1": "DQN eps 0.05+anneal", "_d2": "DQN reward x100+net64"}
    from collections import defaultdict
    groups = defaultdict(list)
    for r in j["per_run"]:
        groups[(r["algo"], r["regime"], _tag_of(r["run"], r["seed"]))].append(r)
    order = ["_v3a", "_v3b", "_v2", "_v1b", "_v5", "_v4a", "_v4b", "_v1a",
             "_v6", "_v6b", "_w3a", "_w3b", "_d1", "_d2"]
    lines = [
        r"% Auto-generated from step5_selection_v3. Do not edit by hand.",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"variant & change & regime & valid/total & pooled vs adaptive & across-seed \\",
        r" & & & seeds & (bps) & $p$ \\",
        r"\midrule",
    ]
    for regime in ["volatile", "calm"]:
        for tag in order:
            algo = "dqn" if tag.startswith("_d") else "ppo"
            key = (algo, regime, tag)
            if key not in groups:
                continue
            grp = groups[key]
            valid = [r for r in grp if a[r["run"]]["valid"]]
            ada = np.array([r["mean_vs_adaptive_bps"] for r in valid])
            if len(ada) >= 2:
                from scipy.stats import ttest_1samp
                p = ttest_1samp(ada, 0.0, alternative="less").pvalue
                p_s = f"{p:.3f}"
            else:
                p_s = "--"
            pooled = f"{ada.mean():+.4f}" if len(ada) else "--"
            lines.append(
                f"{esc(tag or 'base')} & {purpose.get(tag, '?')} & {regime} & "
                f"{len(valid)}/{len(grp)} & {pooled} & {p_s} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t2_tuning_selection.tex", "\n".join(lines))


# ---------------------------------------------------------------- T3
def t3_confirmations():
    lines = [
        r"% Auto-generated from step5_confirm_v3a (9e6) + step5_confirm_v1b (13e6).",
        r"\begin{tabular}{lllrrrc}",
        r"\toprule",
        r"config & sealed block & seed & vs fixed & vs adaptive & $p$ (adaptive) & valid \\",
        r"\midrule",
    ]
    for dirn, label, blk in [("step5_confirm_v3a", "faster-learning (v3a)", "9{,}000{,}000"),
                             ("step5_confirm_v1b", "bigger-network (v1b)", "13{,}000{,}000")]:
        j, a = _load(dirn)
        rows = sorted(j["per_run"], key=lambda r: r["seed"])
        for i, r in enumerate(rows):
            valid = "yes" if a[r["run"]]["valid"] else r"\textbf{NO}"
            cfg = esc(label) if i == 0 else ""
            bl = blk if i == 0 else ""
            lines.append(
                f"{cfg} & {bl} & {r['seed']} & {r['mean_vs_fixed_bps']:+.4f} & "
                f"{r['mean_vs_adaptive_bps']:+.4f} & {r['p_adaptive']:.3f} & {valid} \\\\")
        v = list(j["verdicts"].values())[0]
        lines.append(r"\cmidrule(lr){2-7}")
        lines.append(
            f" & \\multicolumn{{6}}{{l}}{{pooled vs adaptive = {v['pooled_vs_adaptive_bps']:+.4f} bps; "
            f"across-seed $p$ = {v['across_seed_t_p_onesided']:.3f}; "
            f"cheaper in {v['n_cheaper_of_valid']}/{v['n_valid_seeds']}; "
            f"\\textbf{{PASS = {str(v['PASS']).upper()}}}}} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t3_sealed_confirmations.tex", "\n".join(lines))


# ---------------------------------------------------------------- T8
def t8_hyperparams():
    lines = [
        r"% Auto-generated. Base config + each pre-registered variant's single change.",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"code & agent / variant & change from base \\",
        r"\midrule",
        r"base (PPO) & PPO, discrete actions & net 30$\times$5, lr 3e-4, reward $\times$1, n\_steps 2048, ent 0.01, 2M steps \\",
        r"base (DQN) & DQN, discrete actions & net 30$\times$5, lr 1e-4, buffer 1M, $\varepsilon$ 1.0$\to$0.01 over 25\% \\",
        r"\midrule",
        r"v1a & bigger network & net\_arch [64, 64] \\",
        r"v1b & bigger network & net\_arch [128, 128] \\",
        r"v2 & stronger feedback & reward scale $\times$100 \\",
        r"v3a & faster learning & learning rate 1e-3 \\",
        r"v3b & slower learning & learning rate 1e-4 \\",
        r"v4a & no exploration bonus & ent\_coef 0.0 \\",
        r"v4b & more exploration & ent\_coef 0.05 \\",
        r"v5 & longer rollout & n\_steps 8192 \\",
        r"v6 & v2 trained longer & reward $\times$100, 10M steps \\",
        r"v6b & v3b trained longer & lr 1e-4, 10M steps \\",
        r"w3a & combination & net [128,128] + reward $\times$100 \\",
        r"w3b & combination & lr 1e-4 + reward $\times$100 \\",
        r"d1 & DQN collapse fix & exploration\_final\_eps 0.05, anneal 0.5 \\",
        r"d2 & DQN collapse fix & reward $\times$100 + net [64,64] \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("t8_hyperparameters.tex", "\n".join(lines))


# ---------------------------------------------------------------- T4
def t4_grid():
    """Robustness grid: all 16 size x deadline cells per regime (11 judged 2026-07-15
    + 5 previously tested), pooled vs adaptive, across-seed p, §7.5 trigger flag."""
    from scipy.stats import ttest_1samp
    sources = [
        (5.0, 2.5, "step5_grid_b5h150", "_gS5H150"),
        (12.5, 2.5, "step5_grid_b12h150", "_gS12H150"),
        (25.0, 2.5, "step5_grid_b25h150", "_gS25H150"),
        (50.0, 2.5, "step5_grid_b50h150", "_gS50H150"),
        (5.0, 5.0, "step5_sweep_b5", "_v3aB5"),
        (12.5, 5.0, "step5_sweep_b12", "_v3aB12"),
        (25.0, 5.0, "step5_selection_v3", "_v3a"),
        (50.0, 5.0, "step5_sweep_b50", "_v3aB50"),
        (5.0, 10.0, "step5_grid_b5h600", "_gS5H600"),
        (12.5, 10.0, "step5_grid_b12h600", "_gS12H600"),
        (25.0, 10.0, "step5_sweep_h600", "_v3aH600"),
        (50.0, 10.0, "step5_grid_b50h600", "_gS50H600"),
        (5.0, 20.0, "step5_grid_b5h1200", "_gS5H1200"),
        (12.5, 20.0, "step5_grid_b12h1200", "_gS12H1200"),
        (25.0, 20.0, "step5_grid_b25h1200", "_gS25H1200"),
        (50.0, 20.0, "step5_grid_b50h1200", "_gS50H1200"),
    ]
    lines = [
        r"% Auto-generated from step5_grid_* + step5_sweep_* + step5_selection_v3. Do not edit.",
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        r"regime & deadline & size & valid/ & cheaper & pooled vs & across-seed & follow-up \\",
        r" & (min) & (BTC) & total & & adaptive (bps) & $p$ & trigger \\",
        r"\midrule",
    ]
    for regime in ["calm", "volatile"]:
        for size, hz, dirname, tag in sorted(sources, key=lambda x: (x[1], x[0])):
            j, a = _load(dirname)
            vals, n_total = [], 0
            for r in j["per_run"]:
                if r["algo"] != "ppo" or r["regime"] != regime:
                    continue
                if _tag_of(r["run"], r["seed"]) != tag:
                    continue
                n_total += 1
                if a[r["run"]]["valid"]:
                    vals.append(r["mean_vs_adaptive_bps"])
            vals = np.array(vals)
            pooled = vals.mean()
            p = ttest_1samp(vals, 0.0, alternative="less").pvalue if len(vals) >= 2 else np.nan
            trig = (len(vals) == n_total and pooled <= -0.02 and p < 0.05)
            # centre volatile cell: numeric condition met, but its dev signal already
            # FAILED both sealed confirmations (criteria 6.8/6.11) -> resolved, not live
            if (size, hz) == (25.0, 5.0) and regime == "volatile":
                trig_s = "closed (failed both sealed tests)"
            else:
                trig_s = r"\textbf{YES}" if trig else "no"
            reg = regime if (hz, size) == (2.5, 5.0) else ""
            lines.append(
                f"{reg} & {hz:g} & {size:g} & {len(vals)}/{n_total} & "
                f"{int((vals < 0).sum())}/{len(vals)} & {pooled:+.4f} & {p:.4f} & "
                f"{trig_s} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t4_robustness_grid.tex", "\n".join(lines))


L2DIR = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
L2_PANELS = [
    ("1-min bars, 30-min", "runs",
     [("ppo", "96.57"), ("ppo", "193.13"), ("dqn", "96.57"), ("dqn", "193.13")]),
    ("10-s bars, 30-min", "runs_10s",
     [("ppo", "96.57"), ("ppo", "193.13"), ("ppo", "386.27"),
      ("dqn", "96.57"), ("dqn", "193.13"), ("dqn", "386.27")]),
    ("10-s bars, 10-min", "runs_10s_10min",
     [("ppo", "96.57"), ("ppo", "193.13"), ("dqn", "96.57"), ("dqn", "193.13")]),
]


def _l2_arm(dirname, algo, size):
    vals, resid = [], []
    for s in range(5):
        m = json.load(open(L2DIR / dirname / f"{algo}_size{size}_seed{s}" / "meta.json"))
        vals.append(m["val_vs_twap_final"]); resid.append(m["val_residual_freq_final"])
    return np.array(vals), np.array(resid)


# ---------------------------------------------------------------- T5
def t5_env_validation():
    g = json.load(open(S / "step4_gates_v3.json"))
    fair = {r: json.load(open(S / "step3g" / f"fairness_verdict_{r}.json"))
            for r in ("calm", "volatile")}
    rows = []
    for reg in ("calm", "volatile"):
        g1 = g["G1_reaction_lever_rev1"]["regimes"][reg]
        g2 = g["G2_cost_vs_size_rev1"]["regimes"][reg]
        g3 = g["G3_benchmark_sanity_rev1"]["regimes"][reg]
        f = fair[reg]
        grad_max_t = max(abs(p["t"]) for p in f["pace_gradient"])
        rows += [
            (reg, "own impact is real and persists (2nd immediate dump / 1st, cost ratio)", f"{g1['self_impact_ratio_primary']:.2f}",
             r"$\geq 1.25$", g1["pass"]),
            (reg, "dump cost increases with order size (Spearman $\\rho$)", f"{g1['spearman_rho']:.2f}",
             r"$> 0$", g1["pass"]),
            (reg, "book refills after impact (probe cost $+30$s vs $+1$s, bps)",
             f"{g1['probe_bps_t30']:.2f} vs {g1['probe_bps_t1']:.2f}", "lower at $+30$s", g1["pass"]),
            (reg, "cost-vs-size growth matches the real book (sim vs real ratio)",
             f"{g2['growth_ratio_sim']:.2f} vs {g2['growth_ratio_real']:.2f}",
             "within band", g2["pass"]),
            (reg, "fixed-TWAP completes every episode",
             f"{g3['twap_completion_rate']:.0%}".replace("%", r"\%"),
             r"$\geq 99\%$", g3["pass"]),
            (reg, "dumping costs more than scheduling (dump / TWAP, drift-free, bps)",
             f"{g3['driftfree_true_dump_mean_bps']:.2f} / {g3['driftfree_twap_mean_bps']:.2f}",
             "dump $\\geq$ TWAP", g3["pass"]),
            (reg, "no residual background drift (ticks/episode; $t$)",
             f"{f['background_drift_ticks_per_ep']:.2f} ($t$={f['background_drift_t']:.2f})",
             "$t$ n.s.", f["drift_pass"]),
            (reg, "no constant-pace policy beats TWAP (max $|t|$ across paces)",
             f"{grad_max_t:.2f}", "none significant", f["gradient_pass"]),
        ]
    lines = [
        r"% Auto-generated from step4_gates_v3.json + step3g/fairness_verdict_*.json.",
        r"\begin{tabular}{llllc}", r"\toprule",
        r"regime & check & measured & pass band & verdict \\", r"\midrule",
    ]
    for reg, check, meas, band, ok in rows:
        lines.append(f"{reg} & {check} & {meas} & {band} & "
                     f"{'PASS' if ok else r'\textbf{FAIL}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t5_env_validation.tex", "\n".join(lines))


# ---------------------------------------------------------------- T7
def t7_l2_summary():
    from scipy.stats import ttest_1samp
    lines = [
        r"% Auto-generated from the 70 L2 agents' meta.json files (VALIDATION data).",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"panel & arm & pooled vs TWAP & cheaper & flagged & across-seed \\",
        r" & & (bps) & seeds & seeds & $p$ \\", r"\midrule",
    ]
    for label, dirname, arms in L2_PANELS:
        for algo, size in arms:
            vals, resid = _l2_arm(dirname, algo, size)
            p = ttest_1samp(vals, 0.0, alternative="less").pvalue
            lines.append(f"{esc(label)} & {algo.upper()} {size} & {vals.mean():+.4f} & "
                         f"{int((vals < 0).sum())}/5 & {int((resid > 0.10).sum())}/5 & {p:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t7_l2_summary.tex", "\n".join(lines))


# ---------------------------------------------------------------- T9
def t9_l2_perrun():
    lines = [
        r"% Auto-generated: all 70 L2 agents, per seed (VALIDATION data; test columns follow",
        r"% the sealed exam). resid = share of episodes finished by the forced deadline buy.",
        r"% LONGTABLE: flows across pages; include directly (no float, no resizebox).",
        r"\begin{longtable}{llrrr}", r"\toprule",
        r"panel & arm & seed & vs TWAP (bps) & resid \\", r"\midrule", r"\endhead",
    ]
    for label, dirname, arms in L2_PANELS:
        for algo, size in arms:
            vals, resid = _l2_arm(dirname, algo, size)
            for s in range(5):
                flag = r" \textbf{DL}" if resid[s] > 0.10 else ""
                lines.append(f"{esc(label)} & {algo.upper()} {size} & {s} & "
                             f"{vals[s]:+.4f} & {resid[s]:.1%}{flag} \\\\".replace("%", r"\%"))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{longtable}")
    write("t9_l2_perrun.tex", "\n".join(lines))


# ---------------------------------------------------------------- T10
def t10_tuning_perrun():
    j, a = _load("step5_selection_v3")
    rows = sorted(j["per_run"], key=lambda r: (r["algo"], r["regime"],
                                               _tag_of(r["run"], r["seed"]), r["seed"]))
    lines = [
        r"% Auto-generated from step5_selection_v3 (all 98 tuning-campaign runs, per seed).",
        r"% LONGTABLE: flows across pages; include directly (no float, no resizebox).",
        r"\begin{longtable}{lllrrrrc}", r"\toprule",
        r"algo & regime & variant & seed & vs fixed & $p$ & vs adaptive & valid \\", r"\midrule", r"\endhead",
    ]
    for r in rows:
        tag = _tag_of(r["run"], r["seed"]) or "base"
        valid = "yes" if a[r["run"]]["valid"] else r"\textbf{NO}"
        lines.append(f"{r['algo']} & {r['regime']} & {esc(tag)} & {r['seed']} & "
                     f"{r['mean_vs_fixed_bps']:+.4f} & {r['p_fixed']:.3f} & "
                     f"{r['mean_vs_adaptive_bps']:+.4f} & {valid} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write("t10_tuning_perrun.tex", "\n".join(lines))


# ---------------------------------------------------------------- T11
def t11_dqn_probe():
    lines = [
        r"% Auto-generated from step5_dqnprobe_* (criteria Part D, 18 runs).",
        r"\begin{tabular}{llrrrrrc}", r"\toprule",
        r"cell & regime & seed & do-nothing & forced-buy & vs adaptive & $p$ & audit \\",
        r" & & & share & share & (bps) & & \\", r"\midrule",
    ]
    for cell, label in [("b5h150", "5 BTC / 2.5-min"), ("b25h150", "25 BTC / 2.5-min"),
                        ("b25h1200", "25 BTC / 20-min")]:
        j = json.load(open(S / f"step5_dqnprobe_{cell}" / "judgement.json"))
        aud = {e["run"]: e for e in json.load(open(S / f"step5_dqnprobe_{cell}" / "behaviour_audit.json"))}
        for r in sorted(j["per_run"], key=lambda r: (r["regime"], r["seed"])):
            e = aud[r["run"]]
            verdict = "valid" if e["valid"] else r"\textbf{COLL.}"
            lines.append(f"{label} & {r['regime']} & {r['seed']} & "
                         f"{e['action_shares'][0]:.0%} & {e['deadline_residual_frac']:.0%} & "
                         f"{r['mean_vs_adaptive_bps']:+.4f} & {r['p_adaptive']:.3f} & {verdict} \\\\"
                         .replace("%", r"\%"))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t11_dqn_probe.tex", "\n".join(lines))


# ---------------------------------------------------------------- T6
def t6_descriptive():
    audit = {e["run"]: e["valid"] for e in
             json.load(open(S / "step5_v3" / "behaviour_audit.json"))}
    lines = [
        r"% Auto-generated from per_episode_v3/*.npz (deterministic replay; integrity",
        r"% exact vs step5_v3/judgement.json for all 20 runs).",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"regime & policy & mean & sd & median & 5\% & 95\% & $n$ episodes \\", r"\midrule",
    ]
    for regime in ("calm", "volatile"):
        d = np.load(S / "per_episode_v3" / f"{regime}.npz")
        entries = [("fixed TWAP", d["fixed"]), ("adaptive TWAP", d["adaptive"])]
        for algo in ("ppo", "dqn"):
            arrs = [d[k] for k in d.files if k.startswith(f"{algo}_") and audit[k]]
            entries.append((f"{algo.upper()} (valid seeds)", np.concatenate(arrs)))
        for name, a in entries:
            lines.append(f"{regime} & {name} & {a.mean():+.4f} & {a.std():.4f} & "
                         f"{np.median(a):+.4f} & {np.percentile(a,5):+.4f} & "
                         f"{np.percentile(a,95):+.4f} & {len(a):,} \\\\".replace(",", "{,}"))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t6_descriptive_stats.tex", "\n".join(lines))


# ---------------------------------------------------------------- T12
def t12_ladder():
    lines = [
        r"% Auto-generated from step5_esc_* (development block, 5 seeds) and step5_xblock_*",
        r"% (reserve block, first use). The two grid trigger groups, calm regime.",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"group & seed & development (bps) & reserve (bps) & change \\", r"\midrule",
    ]
    for cell, label in [("b50h600", "50 BTC / 10-min"), ("b25h1200", "25 BTC / 20-min")]:
        dev = {r["run"]: r["mean_vs_adaptive_bps"]
               for r in json.load(open(S / f"step5_esc_{cell}" / "judgement.json"))["per_run"]
               if r["regime"] == "calm"}
        res = {r["run"]: r["mean_vs_adaptive_bps"]
               for r in json.load(open(S / f"step5_xblock_{cell}" / "judgement.json"))["per_run"]
               if r["regime"] == "calm"}
        for i, run in enumerate(sorted(dev)):
            seed = run.split("_s")[1][0]
            lines.append(f"{label if i == 0 else ''} & {seed} & {dev[run]:+.4f} & "
                         f"{res[run]:+.4f} & {res[run] - dev[run]:+.4f} \\\\")
        dm, rm = np.mean(list(dev.values())), np.mean(list(res.values()))
        lines.append(f" & pooled & {dm:+.4f} & {rm:+.4f} & {rm - dm:+.4f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t12_ladder.tex", "\n".join(lines))


if __name__ == "__main__":
    t1_primary()
    t2_tuning()
    t3_confirmations()
    t4_grid()
    t5_env_validation()
    t6_descriptive()
    t7_l2_summary()
    t8_hyperparams()
    t9_l2_perrun()
    t10_tuning_perrun()
    t11_dqn_probe()
    t12_ladder()
    print("ALL TABLES DONE ->", OUT)
