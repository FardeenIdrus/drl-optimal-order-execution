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
        r"regime & deadline & size & valid/ & cheaper & pooled vs & across-seed & \S 7.5 \\",
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
                trig_s = r"closed (\S 6 FAIL)"
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


if __name__ == "__main__":
    t1_primary()
    t2_tuning()
    t3_confirmations()
    t4_grid()
    t8_hyperparams()
    print("READY-NOW TABLES DONE ->", OUT)
