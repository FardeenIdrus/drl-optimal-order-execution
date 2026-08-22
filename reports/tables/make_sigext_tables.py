"""Measured-signal extension tables (Phase F) -> LaTeX fragments in reports/tables/.

Companion to reports/figures/sigext/make_sigext_figures.py. Reads the FROZEN result
JSONs only; every number traces to a source-of-record file (live doc addenda G/H/I/J).

TIERING (mirrors the figure suite):
  MAIN      ts2_dev_verdicts, ts4_exploiter_ceiling
  SUPPORT   ts5_base_env_diagnostic
  APPENDIX  ts1_certification_gates, ts3_dev_per_run
  (ts6_sealed_exhibit is written by populate_sealed() once the 17e6 run lands.)

Run:  PYTHONPATH=src .venv/bin/python reports/tables/make_sigext_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent
DEV = S / "step5_signal_dev"
DIAG = DEV / "diagnostics_postnull"
SEALED = S / "step5_signal_sealed"


def _w(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print(f"  wrote {name}.tex")


def _p(p: float) -> str:
    """p-value in a form a reader can scan; no false precision."""
    if p < 1e-15:
        return r"$<10^{-15}$"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times 10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def _tag_of(run: str, seed: int) -> str:
    m = re.search(rf"_s{seed}(_.+)?$", run)
    return (m.group(1) or "") if m else ""


def table_ts1_certification() -> None:
    """APPENDIX. Phase C environment certification: fidelity, fairness, exploitability."""
    g = json.loads((S / "signal" / "gates" / "sigext_gates_v4c_PASS.json").read_text())
    m = g["injection_matching"]
    rows = []
    for regime in ("calm", "volatile"):
        hs = m["regimes"][regime]["horizons"]
        for h in sorted(hs, key=lambda z: float(z)):
            r = hs[h]
            if not r["gated"]:
                continue
            rows.append(f"{regime} & {h} & {r['real_calibrate_slope']:.4f} & "
                        f"{r['sim_total_slope']:.4f} & {100*r['rel_gap']:.1f}\\% & "
                        f"{'PASS' if r['pass'] else 'FAIL'} \\\\")
    fair = []
    for regime in ("calm", "volatile"):
        f = g["fairness"][regime]
        fair.append(f"{regime} & {f['background_drift_ticks_per_ep']:+.2f} & "
                    f"{f['background_drift_t']:+.2f} & "
                    f"{'PASS' if f['drift_pass'] else 'FAIL'} & "
                    f"{'PASS' if f['gradient_pass'] else 'FAIL'} \\\\")
    body = r"""% Phase C certification of the injected environment (APPENDIX).
% Source: signal/gates/sigext_gates_v4c_PASS.json  (all_pass = true)
\begin{table}[htbp]\centering
\caption{Certification of the injected environment. Upper panel: injected
predictability matched against the venue measurement at every gated horizon
(registered acceptance band $\pm$20\%). Lower panel: fairness, comprising background
drift and the pace-gradient exploitability test that no constant-pace schedule can
profit from residual drift.}
\label{tab:sigext-certification}
\small
\begin{tabular}{llrrrl}
\toprule
regime & horizon (s) & measured slope & simulated slope & relative gap & verdict \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{lrrll}
\toprule
regime & drift (ticks/episode) & $t$ & drift verdict & exploitability \\
\midrule
""" + "\n".join(fair) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts1_sigext_certification", body)


def table_ts2_dev_verdicts() -> None:
    """MAIN. Per-group verdicts under the frozen decision rule."""
    j = json.loads((DEV / "judgement.json").read_text())
    label = {"dqn_calm": "DQN base", "dqn_volatile": "DQN base",
             "ppo_calm": "PPO base", "ppo_volatile": "PPO base",
             "ppo_calm_v1a": "PPO net[64,64]", "ppo_volatile_v1a": "PPO net[64,64]",
             "ppo_calm_v1b": "PPO net[128,128]", "ppo_volatile_v1b": "PPO net[128,128]",
             "ppo_calm_v2": r"PPO reward $\times$100", "ppo_volatile_v2": r"PPO reward $\times$100"}
    rows = []
    for regime in ("calm", "volatile"):
        for key, v in j["verdicts"].items():
            if not key.endswith(regime) and f"_{regime}_" not in key + "_":
                continue
            if regime not in key:
                continue
            pooled = v["pooled_vs_adaptive_bps"]
            pooled_s = "---" if (isinstance(pooled, float) and np.isnan(pooled)) else f"{pooled:+.4f}"
            flag = v.get("EDGE", v.get("ESCALATE"))
            rows.append(f"{regime} & {label.get(key, key)} & {v['n_valid_seeds']} & "
                        f"{pooled_s} & {v['n_negative_both']} & {v['n_significant_both']} & "
                        f"{'yes' if flag else 'no'} \\\\")
    body = r"""% Development-block verdicts, injected environment (MAIN).
% Source: step5_signal_dev/judgement.json (frozen rule; n=2000/agent; block 18e6)
\begin{table}[htbp]\centering
\caption{Development-block outcome for every agent configuration in the injected
environment. The decision rule, fixed before training, required a negative mean
against both benchmarks, significance at $p<0.01$, agreement across seeds, and at
least 0.05\,bps of materiality. No configuration met it; every pooled mean is on the
costlier side of adaptive TWAP.}
\label{tab:sigext-dev-verdicts}
\small
\begin{tabular}{llrrrrl}
\toprule
regime & configuration & valid seeds & pooled vs TWAP (bps) & seeds cheaper & seeds significant & edge \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts2_sigext_dev_verdicts", body)


def table_ts4_exploiter() -> None:
    """MAIN. The exploiter ceiling: what a signal-reading rule captures."""
    f = json.loads((DIAG / "diag_signal_follower.json").read_text())
    name = {"follower": "signal-follower (registered benchmark)",
            "half": "half-strength tilt (diagnostic)",
            "bangbang": "threshold rule (diagnostic)"}
    rows = []
    for regime in ("calm", "volatile"):
        ex = f["regimes"][regime]["exploiters"]
        for k in ("follower", "half", "bangbang"):
            e = ex[k]
            rows.append(f"{regime} & {name[k]} & {e['mean_diff_bps']:+.4f} & "
                        f"{e['se']:.4f} & {_p(e['wilcoxon_p'])} & "
                        f"{100*e['executed_frac']:.1f}\\% \\\\")
    body = r"""% Exploiter ceiling on the development block (MAIN).
% Source: step5_signal_dev/diagnostics_postnull/diag_signal_follower.json
\begin{table}[htbp]\centering
\caption{What the injected signal was worth to a rule that simply reads it. All three
rules act inside the agents' own action grid and use only the observation feature the
agents received, evaluated on the same episodes with common random numbers. The
registered rule captures several times the materiality threshold in both regimes.}
\label{tab:sigext-exploiter}
\small
\begin{tabular}{llrrlr}
\toprule
regime & rule & mean vs TWAP (bps) & s.e. & $p$ & completed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts4_sigext_exploiter", body)


def table_ts5_base_env() -> None:
    """SUPPORT. Base-environment diagnostic (post-hoc; multiplicity disclosed)."""
    b = json.loads((DIAG / "diag_base_env.json").read_text())
    name = {"raw": r"$1+(S_2-\bar{S_2})$", "x3": r"$1+3(S_2-\bar{S_2})$",
            "ema8": r"$1+3(\mathrm{EMA}_{8s}-\bar{S_2})$"}
    rows = []
    for regime in ("calm", "volatile"):
        ex = b["regimes"][regime]["exploiters"]
        for k in ("raw", "x3", "ema8"):
            e = ex[k]
            both = (abs(e["mean_diff_bps"]) >= 0.05 and e["wilcoxon_p"] < 0.01
                    and e["mean_diff_bps"] < 0)
            rows.append(f"{regime} & {name[k]} & {e['mean_diff_bps']:+.4f} & "
                        f"{e['se']:.4f} & {_p(e['wilcoxon_p'])} & "
                        f"{'yes' if both else 'no'} \\\\")
    body = r"""% Base-environment imbalance-reader diagnostic (SUPPORTING).
% Source: step5_signal_dev/diagnostics_postnull/diag_base_env.json
\begin{table}[htbp]\centering
\caption{Post-hoc diagnostic asking whether the environment without injection already
contained a capturable edge. It did, in the volatile regime only, and modestly: one of
six tests clears both the materiality and significance thresholds, surviving a
Bonferroni correction across the six ($0.01/6=0.00167$) by a narrow margin. These rules
were not pre-registered and the tilt strengths were chosen by the analyst; the
multiplicity is disclosed here rather than absorbed. Note that the smoothed reader is
null in both regimes, indicating that the endogenous relationship is very short lived.}
\label{tab:sigext-base-env}
\small
\begin{tabular}{llrrll}
\toprule
regime & rule & mean vs TWAP (bps) & s.e. & $p$ & clears both thresholds \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts5_sigext_base_env", body)


def table_ts3_per_run() -> None:
    """APPENDIX. Every run, both benchmarks."""
    j = json.loads((DEV / "judgement.json").read_text())
    audit = {e["run"]: e for e in json.loads((DEV / "behaviour_audit.json").read_text())}
    rows = []
    for r in j["per_run"]:
        a = audit[r["run"]]
        rows.append(f"{r['run'].replace('_', chr(92)+'_')} & {r['mean_vs_fixed_bps']:+.4f} & "
                    f"{_p(r['p_fixed'])} & {r['mean_vs_adaptive_bps']:+.4f} & "
                    f"{_p(r['p_adaptive'])} & {'valid' if a['valid'] else 'invalid'} \\\\")
    body = r"""% Per-run development-block results (APPENDIX).
% Source: step5_signal_dev/judgement.json + behaviour_audit.json
\begin{table}[htbp]\centering
\caption{Every run in the injected-environment campaign, against both benchmarks
($n=2000$ paired episodes each). Runs marked invalid failed the behavioural audit,
which is applied before any cost comparison and independently of outcome.}
\label{tab:sigext-per-run}
\scriptsize
\begin{tabular}{lrlrll}
\toprule
run & vs fixed TWAP (bps) & $p$ & vs adaptive TWAP (bps) & $p$ & audit \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts3_sigext_per_run", body)


def populate_sealed() -> None:
    """MAIN, when the 17e6 exhibit lands (criteria section 8 Amendment A1)."""
    jf, ff = SEALED / "judgement.json", SEALED / "follower_context.json"
    if not jf.exists():
        print("  ts6_sealed_exhibit: PENDING (sealed run still in flight)")
        return
    j = json.loads(jf.read_text())
    rows = []
    for r in j["per_run"]:
        rows.append(f"{r['run'].replace('_', chr(92)+'_')} & {r['mean_vs_fixed_bps']:+.4f} & "
                    f"{_p(r['p_fixed'])} & {r['mean_vs_adaptive_bps']:+.4f} & "
                    f"{_p(r['p_adaptive'])} \\\\")
    fol = ""
    if ff.exists():
        f = json.loads(ff.read_text())
        fr = []
        for regime in ("calm", "volatile"):
            e = f["regimes"][regime]["vs_adaptive"]
            fr.append(f"{regime} & {e['mean_diff_bps']:+.4f} & {e['se']:.4f} & "
                      f"{_p(e['wilcoxon_p'])} \\\\")
        fol = (r"""
\vspace{0.6em}
\begin{tabular}{lrrl}
\toprule
regime & follower vs adaptive TWAP (bps) & s.e. & $p$ \\
\midrule
""" + "\n".join(fr) + r"""
\bottomrule
\end{tabular}
""")
    body = r"""% Sealed exhibit on block 17e6 (MAIN). Predictions registered before unsealing.
% Source: step5_signal_sealed/judgement.json (+ follower_context.json)
\begin{table}[htbp]\centering
\caption[Sealed-block replication]{Sealed-block replication. The predictions were recorded in the registration
ledger before the block was opened: no agent would pass, and the signal-reading rule
would remain profitable beyond the materiality threshold in both regimes.}
\label{tab:sigext-sealed}
\small
\begin{tabular}{lrlrl}
\toprule
run & vs fixed TWAP (bps) & $p$ & vs adaptive TWAP (bps) & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + fol + r"""\end{table}
"""
    _w("ts6_sigext_sealed", body)


def table_ts10_obsfix_per_run() -> None:
    """APPENDIX. Every run in the three injected-market campaigns trained WITH the arrival
    price among the agent's inputs: A4 (10 base PPO), A4.1 (18 PPO variants), A4.2 (10 DQN).

    WHY THIS EXISTS. TS9 reports these campaigns as group means only, while every other
    campaign in the appendix is published run by run (TS3, T1, T10, T11, T12, T9). Thirty-eight
    evaluations had no per-run disclosure. The three folder names differ by a few characters and
    are three DIFFERENT campaigns -- they are labelled in the table and must never be pooled.

    The DQN rows are ALL behaviour-audit invalid. Their costs describe policies that did not
    finish the order, so the audit column is what those rows are for -- not the cost."""
    src = [("A4", "step5_signal_obsfix"), ("A4.1", "step5_signal_obsfix_var"),
           ("A4.2", "step5_signal_obsfix_dqn")]
    rows = []
    for tag, d in src:
        j = json.loads((S / d / "judgement.json").read_text())
        aud = {e["run"]: e for e in json.loads((S / d / "behaviour_audit.json").read_text())}
        for r in sorted(j["per_run"], key=lambda r: (r.get("algo", ""), r["regime"], r["seed"],
                                                     r["run"])):
            a = aud[r["run"]]
            rows.append(
                f"{tag} & {r['run'].replace('_', chr(92)+'_')} & "
                f"{r['mean_vs_fixed_bps']:+.4f} & {_p(r['p_fixed'])} & "
                f"{r['mean_vs_adaptive_bps']:+.4f} & {_p(r['p_adaptive'])} & "
                f"{a['top_share']:.0%} & {a['deadline_residual_frac']:.0%} & "
                f"{'valid' if a['valid'] else chr(92)+'textbf{no}'} \\\\".replace("%", r"\%"))
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = r"""% Per-run results for the three arrival-price campaigns, injected market (APPENDIX).
% Source: step5_signal_obsfix / _var / _dqn -- judgement.json + behaviour_audit.json
\begin{table}[htbp]\centering
\caption{Every run in the injected market trained with the arrival price among the agent's
inputs, against both benchmarks ($n=2000$ paired episodes each), on the development block.
A4 is the base configuration, A4.1 three pre-planned variants, A4.2 the value-based algorithm;
these are three separate campaigns and are not pooled. Runs marked \textbf{no} failed the
behavioural audit, which is applied before any cost comparison and independently of outcome, so
their costs describe policies that did not finish the order.}
\label{tab:sigext-obsfix-per-run}
\scriptsize
\begin{tabular}{llrlrlrrl}
\toprule
 & run & vs fixed TWAP & $p$ & vs adaptive TWAP & $p$ & top & deadline & audit \\
 & & (bps) & & (bps) & & action & residual & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts10_sigext_obsfix_per_run", body)


def main() -> None:
    print("building measured-signal extension tables ->", OUT)
    print(" MAIN:")
    table_ts2_dev_verdicts()
    table_ts4_exploiter()
    table_ts7_ceiling()
    table_ts9_a4_observation()
    table_ts10_obsfix_per_run()
    populate_sealed()
    print(" SUPPORTING:")
    table_ts5_base_env()
    table_ts8_comparators()
    print(" APPENDIX:")
    table_ts1_certification()
    table_ts3_per_run()
    print("done.")




def table_ts7_ceiling() -> None:
    """MAIN. Amendment A2 Test 2: tuning grid on the 18e6 development block + one-shot
    confirmed ceiling on the 21e6 confirmation block + captured fraction vs the agents'
    results on the 17e6 confirmation block. Block kinds follow Table D3."""
    tune = {r: json.loads((DIAG / f"tune_follower_{r}.json").read_text())
            for r in ("calm", "volatile")}
    conf = json.loads((S / "step5_signal_ceiling21e6" / "ceiling_confirmation.json").read_text())
    sealed = json.loads((SEALED / "judgement.json").read_text())
    pooled = {}
    for reg in ("calm", "volatile"):
        v = [r["mean_vs_adaptive_bps"] for r in sealed["per_run"] if r["regime"] == reg]
        pooled[reg] = float(np.mean(v))
    grid_rows = []
    for c in tune["calm"]["grid"]:
        cs = str(c)
        grid_rows.append(f"{c:g} & {tune['calm']['rows'][cs]['mean_diff_bps']:+.4f} & "
                         f"{tune['volatile']['rows'][cs]['mean_diff_bps']:+.4f} \\\\")
    conf_rows = []
    for reg in ("calm", "volatile"):
        e = conf["regimes"][reg]["vs_adaptive"]
        captured = 100.0 * pooled[reg] / e['mean_diff_bps']
        conf_rows.append(f"{reg} & {conf['regimes'][reg]['c_star']:g} & "
                         f"{e['mean_diff_bps']:+.4f} & {e['se']:.4f} & {_p(e['wilcoxon_p'])} & "
                         f"{pooled[reg]:+.4f} & {captured:+.1f}\\% \\\\")
    body = r"""% Amendment A2 Test 2: tuned attainable-edge ceiling (MAIN).
% Sources: diagnostics_postnull/tune_follower_*.json, step5_signal_ceiling21e6/,
% step5_signal_sealed/judgement.json. Registration: criteria section 8 Amendment A2.
\begin{table}[htbp]\centering
\caption[The attainable-edge ceiling]{\textbf{The attainable-edge ceiling and the fraction
captured.} Upper panel: mean paired saving against adaptive TWAP for each coefficient in the
registered grid, on the development block. Lower panel: the tuned rule confirmed once on a
held-out confirmation block, against the agents' results on their own sealed block, and the
fraction of the ceiling captured.}
\label{tab:sigext-ceiling}
\small
\begin{tabular}{lrr}
\toprule
tilt coefficient $c$ & calm (bps) & volatile (bps) \\
\midrule
""" + "\n".join(grid_rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{llrrlrl}
\toprule
regime & $c^*$ & confirmed ceiling (bps) & s.e. & $p$ & agents (sealed) & captured \\
\midrule
""" + "\n".join(conf_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts7_sigext_ceiling", body)




def table_ts8_comparators() -> None:
    """SUPPORT. A3 comparator columns in both environments + the frontier verdict."""
    inj = json.loads((S / "step5_comparators" / "injected_dev.json").read_text())
    base = json.loads((S / "step5_comparators" / "base_5e6.json").read_text())
    fr = json.loads((S / "step5_comparators" / "frontier_summary.json").read_text())
    lab = {"adaptive": "adaptive TWAP", "fixed": "fixed TWAP",
           "ac_kT0": r"AC $\kappa T$=0", "ac_kT1": r"AC $\kappa T$=1",
           "ac_kT2": r"AC $\kappa T$=2", "ac_kT4": r"AC $\kappa T$=4",
           "vwap_oracle": "oracle VWAP (infeasible)"}
    rows = []
    for k in ("adaptive", "fixed", "ac_kT0", "ac_kT1", "ac_kT2", "ac_kT4", "vwap_oracle"):
        cells = []
        for d in (base, inj):
            for reg in ("calm", "volatile"):
                v = d["regimes"][reg][k]
                cells.append(f"{v['mean_vs_adaptive_bps']:+.4f}")
                cells.append(f"{v['std_cost_bps']:.2f}")
        rows.append(f"{lab[k]} & " + " & ".join(cells) + r" \\")
    fr_rows = []
    for reg in ("calm", "volatile"):
        f = fr[reg]
        ex = f["mean_excess_bps"]
        fr_rows.append(f"{reg} & {f['n_dominated']}/5 & "
                       + (f"{ex:+.4f}" if ex is not None else "---")
                       + " & " + ", ".join(lab.get(x["policy"], x["policy"])
                                               for x in f["frontier"]) + r" \\")
    body = r"""% A3 comparators + frontier (SUPPORTING). Sources: step5_comparators/*.json
\begin{table}[htbp]\centering
\caption[Comparator policies in both environments]{\textbf{Comparator policies in both
environments.} Mean difference against adaptive TWAP and per-episode cost standard deviation,
$n=2000$ paired episodes per cell. Lower panel, injected environment: how many trained agents
are strictly dominated, meaning a feasible benchmark offers both lower cost and lower risk.}
\label{tab:sigext-comparators}
% NINE columns: \small overflows the portrait text block by ~26pt. \scriptsize fits without
% resizebox (which would scale the font non-uniformly against surrounding text).
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{4}{c}{base environment} & \multicolumn{4}{c}{injected environment} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
& \multicolumn{2}{c}{calm} & \multicolumn{2}{c}{volatile} & \multicolumn{2}{c}{calm} & \multicolumn{2}{c}{volatile} \\
policy & vs TWAP & s.d. & vs TWAP & s.d. & vs TWAP & s.d. & vs TWAP & s.d. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{llrl}
\toprule
regime & agents dominated & mean excess cost (bps) & efficient set \\
\midrule
""" + "\n".join(fr_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts8_sigext_comparators", body)




def table_ts9_a4_observation() -> None:
    """MAIN. Amendment A4: the observation variant. Upper panel = the mechanism (critic
    explained variance before/after); lower = the verdict (performance unchanged)."""
    orig = json.loads((DIAG / "diag_learning.json").read_text())
    a4 = json.loads((DIAG / "diag_learning_a4.json").read_text())
    j4 = json.loads((S / "step5_signal_obsfix" / "judgement.json").read_text())
    aud = {x["run"]: x for x in json.loads(
        (S / "step5_signal_obsfix" / "behaviour_audit.json").read_text())}
    jd = json.loads((DEV / "judgement.json").read_text())
    ao = {x["run"]: x for x in json.loads((DEV / "behaviour_audit.json").read_text())}
    om = {r["run"]: r for r in orig}
    am = {r["run"]: r for r in a4}
    ev_rows = []
    for run in sorted(om):
        ev_rows.append(f"{run.replace('_', chr(92)+'_')} & {om[run]['critic']['explained_variance']:+.4f} & "
                       f"{am[run]['ev']:+.4f} & {om[run]['critic']['corr_V_return']:+.3f} & "
                       f"{am[run]['corr']:+.3f} \\\\")
    ver = []
    for reg in ("calm", "volatile"):
        o = [r["mean_vs_adaptive_bps"] for r in jd["per_run"]
             if r["algo"] == "ppo" and r["regime"] == reg
             and r["run"].endswith(f"s{r['seed']}") and ao[r["run"]]["valid"]]
        n = [r["mean_vs_adaptive_bps"] for r in j4["per_run"]
             if r["regime"] == reg and aud[r["run"]]["valid"]]
        ver.append(f"{reg} & {np.mean(o):+.4f} & {len(o)}/5 & {np.mean(n):+.4f} & "
                   f"{len(n)}/5 & no \\\\")
    body = r"""% Amendment A4: observation variant (MAIN).
% Sources: diagnostics_postnull/diag_learning{,_a4}.json, step5_signal_obsfix/
\begin{table}[htbp]\centering
\caption[The observation variant]{\textbf{The observation variant.} The agents' state omitted
the price relative to arrival, against which their cost is measured. Upper panel: adding that
one feature, with training, seeds and market otherwise identical, makes the value function
learnable in every run. Lower panel: it does not change cost. Valid-seed counts are reported
alongside each figure, since fewer runs pass the behavioural audit under the variant.}
\label{tab:sigext-a4}
\small
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{explained variance} & \multicolumn{2}{c}{corr($V$, return)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
run & without & with & without & with \\
\midrule
""" + "\n".join(ev_rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{lrrrrl}
\toprule
& \multicolumn{2}{c}{without the arrival price} & \multicolumn{2}{c}{with the arrival price} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
regime & vs TWAP (bps) & valid & vs TWAP (bps) & valid & meets edge rule \\
\midrule
""" + "\n".join(ver) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _w("ts9_sigext_a4_observation", body)


if __name__ == "__main__":
    main()
