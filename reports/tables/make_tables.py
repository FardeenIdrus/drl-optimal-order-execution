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


def _perrun_row(r: dict, e: dict, lead: str) -> str:
    """One appendix row: both benchmarks, then the two behaviour-audit quantities.

    Percentages are written as `\\%` directly rather than formatted with `{:.0%}` and then
    string-replaced. The replace approach was used first and was wrong twice over: it also
    escaped the `%` inside the format spec, and it was applied to the return value of
    `list.append`, which is None."""
    return (f"{lead} & {esc(r['run'])} & {r['mean_vs_fixed_bps']:+.4f} & "
            f"{r['p_fixed']:.3f} & {r['mean_vs_adaptive_bps']:+.4f} & "
            f"{e['top_share'] * 100:.0f}" + r"\% & "
            f"{e['deadline_residual_frac'] * 100:.0f}" + r"\% & "
            f"{'valid' if e['valid'] else chr(92) + 'textbf{no}'} " + r"\\")


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
        r"config & confirmation block & seed & vs fixed & vs adaptive & $p$ (adaptive) & valid \\",
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
            # A single \multicolumn{6}{l} cannot wrap, so a long summary string silently
            # overflows the text block (it did: 109pt). Kept compact deliberately.
            f" & \\multicolumn{{6}}{{l}}{{pooled {v['pooled_vs_adaptive_bps']:+.4f} bps; "
            f"$p$ = {v['across_seed_t_p_onesided']:.3f}; "
            f"{v['n_cheaper_of_valid']}/{v['n_valid_seeds']} cheaper; "
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
                trig_s = "closed (failed both confirmations)"
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


# --------------------------------------------------------- sealed-exam (TEST) columns
L2TEST = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results")
_L2_TEST_CACHE: dict = {}


def _l2_test():
    """Per-agent sealed-exam rows, keyed (runs_dir, run). One shot, block spent 2026-07-30."""
    if not _L2_TEST_CACHE:
        for f in sorted(L2TEST.glob("test_*.json")):
            for r in json.loads(f.read_text())["runs_flat"]:
                _L2_TEST_CACHE[(r["runs_dir"], r["run"])] = r
    return _L2_TEST_CACHE


def _l2_arm_test(dirname, algo, size):
    """Sealed-exam values for one arm, per seed, with the deadline-audit flag.

    The stored `arm_summary` pooled column averages ALL five seeds including agents the
    deadline audit rejects, which contradicts the audit-before-cost rule applied in every
    other campaign. Pooling here is over VALID seeds only; no PASS verdict moves (all four
    nominally-passing arms have zero flagged seeds) but four DQN control arms shift, every
    one of them toward cheaper. See the addendum in reports/l2_test_protocol.md."""
    t = _l2_test()
    vals, flags = [], []
    for s in range(5):
        r = t.get((dirname, f"{algo}_size{size}_seed{s}"))
        if r is None:
            continue
        vals.append(r["mean_paired_diff_bps"])
        flags.append(bool(r["dl_flag"]))
    return np.array(vals), np.array(flags, dtype=bool)


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
            (reg, "a purchase moves the price, and the move persists (second immediate purchase / first, cost ratio)", f"{g1['self_impact_ratio_primary']:.2f}",
             r"$\geq 1.25$", g1["pass"]),
            (reg, "cost of an immediate purchase rises with its size (Spearman $\\rho$)", f"{g1['spearman_rho']:.2f}",
             r"$> 0$", g1["pass"]),
            (reg, "the book refills after a purchase (probe cost 30\\,s later vs 1\\,s later, bps)",
             f"{g1['probe_bps_t30']:.2f} vs {g1['probe_bps_t1']:.2f}", "lower at 30\\,s", g1["pass"]),
            (reg, "cost rises with size as it does on Hyperliquid (simulator vs Hyperliquid ratio)",
             f"{g2['growth_ratio_sim']:.2f} vs {g2['growth_ratio_real']:.2f}",
             "within 25\\%", g2["pass"]),
            (reg, "fixed TWAP completes every episode",
             f"{g3['twap_completion_rate']:.0%}".replace("%", r"\%"),
             r"$\geq 99\%$", g3["pass"]),
            (reg, "buying all at once costs more than scheduling (immediate / TWAP, bps)",
             f"{g3['driftfree_true_dump_mean_bps']:.2f} / {g3['driftfree_twap_mean_bps']:.2f}",
             "immediate $\\geq$ TWAP", g3["pass"]),
            (reg, "no background price drift (ticks per episode; $t$)",
             f"{f['background_drift_ticks_per_ep']:.2f} ($t$={f['background_drift_t']:.2f})",
             "$t$ not significant", f["drift_pass"]),
            (reg, "no constant pace beats TWAP (largest $|t|$ across the six paces)",
             f"{grad_max_t:.2f}", "none significant", f["gradient_pass"]),
        ]
    lines = [
        r"% Auto-generated from step4_gates_v3.json + step3g/fairness_verdict_*.json.",
        # The check column is a sentence, not a token: it needs a p{} column or it
        # overflows the text block (it did: 107pt).
        r"\begin{tabular}{l p{6.1cm} l l c}", r"\toprule",
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
        r"% Auto-generated. VALIDATION columns from the 70 agents' meta.json; SEALED-EXAM",
        r"% columns from l2_test_results/test_*.json (one shot, 2026-07-30, block spent).",
        r"% Sealed columns pool BEHAVIOUR-VALID seeds only (audit before cost), unlike the",
        r"% stored arm_summary -- see the addendum in reports/l2_test_protocol.md.",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"& & \multicolumn{3}{c}{validation} & \multicolumn{3}{c}{held-out test period} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"panel & configuration (size in BTC) & pooled & cheaper & $p$ & pooled & cheaper & $p$ \\",
        r" & & (bps) & seeds & & (bps) & seeds & \\", r"\midrule",
    ]
    for label, dirname, arms in L2_PANELS:
        for algo, size in arms:
            vals, resid = _l2_arm(dirname, algo, size)
            p = ttest_1samp(vals, 0.0, alternative="less").pvalue
            tv, tf = _l2_arm_test(dirname, algo, size)
            ok = tv[~tf] if len(tv) else tv
            clears = False
            if len(ok) >= 2:
                tp = ttest_1samp(ok, 0.0, alternative="less").pvalue
                # clears the pass rule once the behaviour audit is applied first (Sec 4.6)
                clears = bool(ok.mean() < 0 and tp < 0.05)
                bf = (lambda x: r"\textbf{" + x + "}") if clears else (lambda x: x)
                tcell = (f"{bf(f'{ok.mean():+.4f}')} & {bf(f'{int((ok < 0).sum())}/{len(ok)}')}"
                         f" & {bf(f'{tp:.3f}')}")
            elif len(ok) == 1:
                bf = lambda x: x
                tcell = f"{ok.mean():+.4f} & {int((ok < 0).sum())}/1 & --"
            else:
                bf = lambda x: x
                tcell = "-- & -- & --"
            lines.append(f"{esc(label)} & {bf(algo.upper() + ' ' + str(size))} & "
                         f"{vals.mean():+.4f} & "
                         f"{int((vals < 0).sum())}/5 & {p:.3f} & {tcell} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t7_l2_summary.tex", "\n".join(lines))


# ---------------------------------------------------------------- T9
def t9_l2_perrun():
    lines = [
        r"% Auto-generated: all 70 L2 agents, per seed. VALIDATION columns from meta.json;",
        r"% SEALED-EXAM column from l2_test_results/test_*.json (one shot, 2026-07-30).",
        r"% resid = share of episodes finished by the forced deadline buy; DL = audit-rejected.",
        r"% LONGTABLE: flows across pages; include directly (no float, no resizebox).",
        r"\begin{longtable}{llrrrrr}", r"\toprule",
        r"panel & configuration (size in BTC) & seed & \multicolumn{2}{c}{validation} & "
        r"\multicolumn{2}{c}{test period} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & & & vs TWAP (bps) & resid & vs TWAP (bps) & audit \\",
        r"\midrule", r"\endhead",
    ]
    for label, dirname, arms in L2_PANELS:
        for algo, size in arms:
            vals, resid = _l2_arm(dirname, algo, size)
            tv, tf = _l2_arm_test(dirname, algo, size)
            for s in range(5):
                flag = r" \textbf{DL}" if resid[s] > 0.10 else ""
                if s < len(tv):
                    tcell = (f"{tv[s]:+.4f} & "
                             + (r"\textbf{DL}" if tf[s] else "ok"))
                else:
                    tcell = "-- & --"
                lines.append(f"{esc(label)} & {algo.upper()} {size} & {s} & "
                             f"{vals[s]:+.4f} & {resid[s]:.1%}{flag} & {tcell} \\\\"
                             .replace("%", r"\%"))
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


# ---------------------------------------------------------------- T13
def t13_primary_observation():
    """Amendment A4.3: the observation specification on the PRIMARY track. Upper panel is the
    mechanism (critic explained variance, measured by reports/diagnostics/diag_learning_primary.py);
    lower panel is the verdict (cost, from the two campaigns' judgement files).

    DQN is absent from the upper panel by construction: explained variance of a critic is a
    policy-gradient quantity and DQN has no separate value head fitted against returns. It IS in
    the lower panel, where its 10/10 audit failure is the finding."""
    diag = json.load(open(S / "diagnostics_primary" / "diag_learning_primary.json"))
    dm = {(d["arm"], d["run"]): d for d in diag}
    jo, ao = _load("step5_v3")
    jn, an = _load("step5_primary_v3_obsfix")

    lines = [
        r"% Auto-generated from diagnostics_primary/diag_learning_primary.json and",
        r"% step5_v3 + step5_primary_v3_obsfix judgement/behaviour_audit. Do not edit by hand.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"& \multicolumn{2}{c}{explained variance} & \multicolumn{2}{c}{corr($V$, return)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"run & without & with & without & with \\", r"\midrule",
    ]
    for regime in ("calm", "volatile"):
        for seed in range(5):
            run = f"ppo_{regime}_s{seed}"
            o, n = dm[("original", run)]["critic"], dm[("obsfix", run)]["critic"]
            lines.append(
                f"{esc(run)} & {o['explained_variance']:+.4f} & {n['explained_variance']:+.4f} & "
                f"{o['corr_V_return']:+.3f} & {n['corr_V_return']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.6em}", "",
              r"\begin{tabular}{llrrrrc}", r"\toprule",
              r"& & \multicolumn{2}{c}{without the arrival price} & "
              r"\multicolumn{2}{c}{with the arrival price} & \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
              r"algo & regime & vs TWAP (bps) & valid & vs TWAP (bps) & valid & edge \\",
              r"\midrule"]
    for algo in ("ppo", "dqn"):
        for regime in ("calm", "volatile"):
            g = f"{algo}_{regime}"
            o = [r["mean_vs_adaptive_bps"] for r in jo["per_run"]
                 if r["algo"] == algo and r["regime"] == regime
                 and _tag_of(r["run"], r["seed"]) == "" and ao[r["run"]]["valid"]]
            n = [r["mean_vs_adaptive_bps"] for r in jn["per_run"]
                 if r["algo"] == algo and r["regime"] == regime and an[r["run"]]["valid"]]
            fo = f"{np.mean(o):+.4f}" if o else "---"
            fn = f"{np.mean(n):+.4f}" if n else "---"
            edge = "no" if not (jo["verdicts"][g]["EDGE"] or jn["verdicts"][g]["EDGE"]) else "YES"
            lines.append(f"{algo} & {regime} & {fo} & {len(o)}/5 & {fn} & {len(n)}/5 & "
                         f"{edge} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t13_primary_observation.tex", "\n".join(lines))


# ---------------------------------------------------------------- T14
def t14_primary_obsfix_perrun():
    """APPENDIX. Every run of Amendment A4.3 -- the primary market with the arrival price added
    to the agent's inputs. T13 reports this campaign as four group means; every other campaign
    in the appendix is published run by run, so twenty evaluations had no per-run disclosure.

    All ten DQN rows are behaviour-audit invalid. Their costs describe policies that never
    finished the order; the audit column is what those rows carry, not the cost."""
    j, a = _load("step5_primary_v3_obsfix")
    lines = [
        r"% Auto-generated from step5_primary_v3_obsfix. Do not edit by hand.",
        r"\begin{tabular}{llrrlrrl}", r"\toprule",
        r"algo & run & vs fixed & $p$ & vs adaptive & top & deadline & audit \\",
        r" & & TWAP (bps) & & TWAP (bps) & action & residual & \\", r"\midrule",
    ]
    for r in sorted(j["per_run"], key=lambda r: (r["algo"], r["regime"], r["seed"])):
        lines.append(_perrun_row(r, a[r["run"]], r["algo"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t14_primary_obsfix_perrun.tex", "\n".join(lines))


# ---------------------------------------------------------------- T15
def t15_a5_sealed_perrun():
    """APPENDIX. The one-shot sealed test on block 25,000,000, both arms, every evaluation.

    ARM B trained five FRESH seeds (20-24). ARM A re-scored the SAME twenty A4.3 agents,
    weights untouched, on the same new block -- so arm A isolates the block from the seeds.
    Both arms are reported; the escalated candidate failed in both."""
    lines = [
        r"% Auto-generated from step5_a5_armB_freshseeds + step5_a5_armA_sameagents.",
        r"% Block 25,000,000 -- SPENT. One shot, no re-runs.",
        r"\begin{tabular}{llrrlrrl}", r"\toprule",
        r"arm & run & vs fixed & $p$ & vs adaptive & top & deadline & audit \\",
        r" & & TWAP (bps) & & TWAP (bps) & action & residual & \\", r"\midrule",
    ]
    for tag, d in [("B (fresh seeds)", "step5_a5_armB_freshseeds"),
                   ("A (same agents)", "step5_a5_armA_sameagents")]:
        j, a = _load(d)
        for r in sorted(j["per_run"], key=lambda r: (r["algo"], r["regime"], r["seed"])):
            lines.append(_perrun_row(r, a[r["run"]], tag))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("t15_a5_sealed_perrun.tex", "\n".join(lines))


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
    t13_primary_observation()
    t14_primary_obsfix_perrun()
    t15_a5_sealed_perrun()
    print("ALL TABLES DONE ->", OUT)
