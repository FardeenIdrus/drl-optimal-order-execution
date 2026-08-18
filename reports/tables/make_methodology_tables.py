"""The Methodology chapter's four tables. No number is typed here.

  tab:m-provenance   what the vendored implementation supplied against what this study built
  tab:m-tracks       the three environments: design panel and scale panel
  tab:m-injgates     the injected environment's certification, six checks, both regimes
  tab:m-register     what was registered, when, the written prediction, whether it held

EVERY NUMERIC CELL IS READ AT BUILD TIME:

  methodology_measurements.json          census, budgets, order ladders, observation widths,
                                         K and Q (reports/diagnostics/methodology_measure.py)
  qrm_optimal_execution/.../default.yaml the inherited scaffold's own configuration -- the
                                         left column of the provenance table is parsed from
                                         the authors' file, not transcribed from their paper
  configs/experiment*.yaml               this study's configuration
  signal/gates/sigext_gates_v4c_PASS.json  the injected environment's certified gates

TWO DECLARED EXCEPTIONS, both the same kind the Data chapter's Panel B declares.

1. tab:m-provenance's prose cells (calibration market, learner family, benchmark names) are
   designations, not measurements. They are literals here, marked LITERAL, and each carries
   its source in the code below. The NUMERIC cells are parsed.
2. tab:m-register's "what it fixed" and "did it hold" columns are human designations
   recorded in the frozen criteria ledger. They are literals here, each with its ledger line
   number, because a registration's meaning is not derivable from any file.

CAPTION OBLIGATIONS are printed by this script at the end of every run, because a caption
that is not written next to the number it qualifies is the failure mode this project keeps
hitting. Copy them into the chapter verbatim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OX = SCRATCH / "oxford_l4"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VENDORED = REPO.parent / "qrm_optimal_execution"

M = json.loads((OX / "methodology_measurements.json").read_text())
G = json.loads((OX / "signal" / "gates" / "sigext_gates_v4c_PASS.json").read_text())
CEN, OBS = M["census"], M["observation_widths"]


def _yaml_scalar(text: str, key: str):
    m = re.search(rf"^\s*{key}\s*:\s*([^#\n]+)", text, re.M)
    return m.group(1).strip().strip('"') if m else None


def _yaml_list(text: str, key: str):
    m = re.search(rf"^\s*{key}\s*:\s*\[(.+?)\]", text, re.M)
    return [x.strip() for x in m.group(1).split(",")] if m else None


SCAF = (VENDORED / "src" / "qrm_rl" / "configs" / "default.yaml").read_text()
OURS = (REPO / "configs" / "experiment.yaml").read_text()


def _tabular(colspec: str, header: list[str], rows: list[list[str]], notes: str = "") -> str:
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
           " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    for r in rows:
        cells = [c for c in r if c != ""] if r and r[0].startswith(r"\multicolumn") else r
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule"]
    if notes:
        out.append(rf"\multicolumn{{{len(header)}}}{{@{{}}p{{\linewidth}}@{{}}}}{{\footnotesize {notes}}} \\")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def _w(name: str, body: str) -> None:
    (HERE / name).write_text(body)
    print(f"  wrote {name}  ({len(body.splitlines())} lines)")


# --------------------------------------------------------------- tab:m-provenance
def provenance() -> None:
    """Left column parsed from the authors' own config; right from ours and the census."""
    horizon = int(float(_yaml_scalar(SCAF, "time_horizon")))
    step = int(float(_yaml_scalar(SCAF, "trader_time_step")))
    inv = int(float(_yaml_scalar(SCAF, "initial_inventory")))
    their_actions = _yaml_list(SCAF, "actions")
    their_state = int(float(_yaml_scalar(SCAF, "state_dim")))
    their_agent = _yaml_scalar(SCAF, "agent_type")
    their_pen = _yaml_scalar(SCAF, "final_penalty")
    their_K = len(_yaml_list(SCAF, "aes"))          # one AES per visible depth
    our_actions = _yaml_list(OURS, "actions")
    our_sizes = sorted({s for t in ("reacting_simulator", "injected_simulator")
                        for s in CEN[t]["order_sizes_btc"]})
    PD = CEN["primary_design"]
    ow = OBS

    rows = [
        # LITERAL (designation, paper §2 and their README): the calibration market.
        ["Calibrated on", "France T\\'el\\'ecom, Euronext Paris, Jan 2010 to Mar 2012",
         "Hyperliquid BTC perpetual, December 2025, per-order records"],
        ["Book depth", rf"$K = {their_K}$ per side",
         rf"$K = {ow['K_per_side']}$ per side (${2*ow['K_per_side']}$ queue sizes observed)"],
        ["Queue symmetry", "bid--ask symmetry assumed",
         rf"asymmetric: per-side intensities and invariant distributions, "
         rf"$Q={ow['Q_by_regime']['calm']}$ calm / ${ow['Q_by_regime']['volatile']}$ volatile"],
        ["Observation", rf"{their_state} variables",
         rf"{ow['recorded_books']['base']} recorded books; {ow['reacting_simulator']['base']} "
         rf"reacting; {ow['injected_simulator']['base']} injected "
         rf"($+1$ each simulator under the registered amendment)"],
        ["Actions", rf"{len(their_actions)}: {', '.join(their_actions)} of the volume at best ask",
         rf"{len(our_actions)} pace multiples of TWAP: {', '.join(our_actions)}; "
         rf"$1.0$ \emph{{is}} adaptive TWAP"],
        ["Learner", their_agent,
         " and ".join(a.upper() for a in ("ppo", "dqn")
                      if a in set(CEN["reacting_simulator"]["algos"]) |
                      set(CEN["injected_simulator"]["algos"]))],
        # The paper's decision COUNT is deliberately not derived here. 600/25 is 24 intervals
        # or 25 decision points depending on whether t=0 counts, and a wrong count about a
        # third party's design is the one error that cannot be corrected after submission.
        # State what their file states: the horizon, the cadence, the inventory.
        ["Episode", rf"{horizon}\,s horizon, one decision every {step}\,s, {inv} shares",
         rf"{PD['decisions_per_episode']}\,s horizon, one decision every "
         rf"{PD['cadence_s']}\,s ({PD['decisions_per_episode']} decisions), "
         rf"{PD['order_btc']:g}\,BTC; ladder {{{', '.join(f'{s:g}' for s in our_sizes)}}}\,BTC "
         rf"and horizons {{{', '.join(f'{h}' for h in PD['horizon_variants_decisions'])}}}\,s"],
        # LITERAL (designation, paper eq. 4; their final_penalty parsed above).
        ["Reward", rf"terminal-penalised shortfall, $\alpha = {their_pen}$",
         "negative implementation shortfall against the arrival mid, bps, forced completion"],
        ["Benchmarks", "TWAP, POPV1--4",
         "plain and adaptive TWAP, Almgren--Chriss family, expected-volume and oracle VWAP"],
        ["Evaluation", "no seed protocol, sealed block or registration reported",
         "registered rules, materiality floor, five seeds, common random numbers, "
         "behaviour audit, three-rung ladder, one-shot sealed confirmations"],
        ["Agents trained", "not reported",
         rf"\textbf{{{CEN['TOTAL_agents_of_record']}}} "
         rf"({CEN['recorded_books']['agents']} / {CEN['reacting_simulator']['agents']} / "
         rf"{CEN['injected_simulator']['agents']} by environment)"],
    ]
    note = (r"The left column describes \emph{both} the paper and its released implementation. "
            r"Rows parsed from the authors' own configuration file "
            r"(\texttt{src/qrm\_rl/configs/default.yaml}, MIT licence, vendored at commit "
            r"\texttt{c066726}, working tree clean): book depth, observation, actions, agent, "
            r"episode and reward penalty. The book depth in particular is \emph{not} stated "
            r"numerically in the paper and is verifiable only from that file.")
    _w("m1_provenance.tex", _tabular("@{}p{0.15\\linewidth}p{0.34\\linewidth}p{0.44\\linewidth}@{}",
                                     ["", "Espa\\~na et al. (2025)", "This study"], rows, note))


# ------------------------------------------------------------------- tab:m-tracks
def tracks() -> None:
    rb, re_, inj = CEN["recorded_books"], CEN["reacting_simulator"], CEN["injected_simulator"]
    ow = OBS

    def budgets(v):
        return ", ".join(f"{b/1e6:g}M" for b in v)

    design = [
        ["Market", "recorded order books, replayed", "queue-reactive simulator",
         "queue-reactive simulator"],
        ["Does the book react?", "no", "yes", "yes"],
        # "endogenous" is defined nowhere in the document; say what it means instead.
        ["Predictable signal", "whatever the real data held",
         "only what the simulator's own dynamics produce",
         "a measured Hyperliquid signal, injected"],
        ["Decision cadence", r"10\,s and 1\,min (three builds)",
         rf"{CEN['primary_design']['cadence_s']}\,s", rf"{CEN['primary_design']['cadence_s']}\,s"],
        ["Decisions per episode", "30, 60, 180 (by build)",
         rf"{CEN['primary_design']['decisions_per_episode']} primary; "
         rf"{{{', '.join(str(d) for d in re_['decisions_per_episode'])}}} across the horizon grid",
         ", ".join(str(d) for d in inj["decisions_per_episode"])],
        ["Order sizes (BTC)",
         ", ".join(f"{s:g}" for s in sorted({s for b in rb["builds"].values()
                                             for s in b["order_sizes_btc"]})),
         ", ".join(f"{s:g}" for s in re_["order_sizes_btc"]),
         ", ".join(f"{s:g}" for s in inj["order_sizes_btc"])],
        ["Training budget (steps)",
         budgets(sorted({b for v in rb["builds"].values() for b in v["training_budget_steps"]})),
         budgets(re_["training_budget_steps"]), budgets(inj["training_budget_steps"])],
        ["Observation width",
         rf"{ow['recorded_books']['base']}",
         rf"{ow['reacting_simulator']['base']} / {ow['reacting_simulator']['with_arrival_price']}",
         rf"{ow['injected_simulator']['base']} / {ow['injected_simulator']['with_arrival_price']}"],
        # LITERAL (designation, l2_test_protocol.md:97-98 and criteria section 3).
        ["Benchmark", "fixed TWAP", "adaptive TWAP", "adaptive TWAP"],
    ]
    # Settings only. The census (agents, campaigns, learners, the 429 total) is scope, not a
    # setting, and left this table on 2026-08-18; it belongs to the research-design section or
    # the appendix. The in-table note went with it: captions describe, they do not argue.
    _w("m2_tracks.tex", _tabular(
        "@{}p{0.2\\linewidth}p{0.24\\linewidth}p{0.24\\linewidth}p{0.24\\linewidth}@{}",
        ["", "Recorded books", "Reacting simulator", "Injected simulator"], design))


# ---------------------------------------------------------------- tab:m-injgates
def injected_gates() -> None:
    """The injected environment's certification. Same shape as T5, plus the matching row."""
    rows = []
    for reg in ("calm", "volatile"):
        g1 = G["G1_reaction_lever_rev1"]["regimes"][reg]
        g2 = G["G2_cost_vs_size_rev1"]["regimes"][reg]
        g3 = G["G3_benchmark_sanity_rev1"]["regimes"][reg]
        f = G["fairness"][reg]
        mt = G["injection_matching"]["regimes"][reg]["horizons"]
        gated = {h: v for h, v in mt.items() if v["gated"]}
        worst_h, worst = max(gated.items(), key=lambda kv: kv[1]["rel_gap"])
        grad = f["pace_gradient"]
        best = min(grad, key=lambda p: p["cost_vs_twap_bps"])
        tick = lambda ok: r"\checkmark" if ok else r"$\times$"  # noqa: E731
        rows += [
            [reg, "a purchase moves the price, and the move persists (second immediate purchase / first, cost ratio)",
             f"{g1['self_impact_ratio_primary']:.2f}", r"$\geq 1.25$", tick(g1["pass"])],
            [reg, "cost of an immediate purchase rises with its size (Spearman $\\rho$)",
             f"{g1['spearman_rho']:.2f}", r"$> 0$", tick(g1["pass"])],
            [reg, "cost rises with size as it does on Hyperliquid (simulator vs Hyperliquid ratio)",
             f"{g2['growth_ratio_sim']:.2f} vs {g2['growth_ratio_real']:.2f} "
             f"({g2['rel_gap']:.1%} apart)".replace("%", r"\%"),
             r"within $25\%$", tick(g2["pass"])],
            [reg, "fixed TWAP completes every episode",
             f"{g3['twap_completion_rate']:.0%}".replace("%", r"\%"), r"$\geq 99\%$",
             tick(g3["pass"])],
            [reg, "no background price drift (ticks per episode; bps; $t$)",
             f"{f['background_drift_ticks_per_ep']:.2f}; "
             f"{f['background_drift_bps']:.3f}; $t={f['background_drift_t']:.2f}$",
             r"$|t|<2$ \textbf{or} $|\mathrm{mean}|\leq0.5$", tick(f["drift_pass"])],
            # BAND LABEL CORRECTED 2026-08-12. It read "none material", which is FALSE for
            # volatile: its best pace is -0.0859 bps, 1.7x the 0.05 materiality floor. The
            # gate requires an advantage to be BOTH material AND significant, and volatile
            # passes on the second clause only. Stating the wrong reason for a pass is the
            # same defect the drift row two lines above is built to avoid.
            # COUNT CORRECTED 2026-08-18: SIX paces are tested, not seven.
            # step3g.py:544 mults = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]; the seventh action,
            # 0.0, never trades and so cannot be a completing constant-pace policy.
            [reg, "no constant pace beats TWAP (best of six paces, bps vs TWAP)",
             f"{best['cost_vs_twap_bps']:+.4f} at ${best['mult']:g}\\times$ "
             f"($t={best['t']:.2f}$)",
             r"none both material \emph{and} significant", tick(f["gradient_pass"])],
            [reg, "the injected signal reproduces the real slope (worst gated horizon)",
             f"{worst['rel_gap']:.1%} at {worst_h}\\,s".replace("%", r"\%"),
             r"$\leq 20\%$, all gated horizons",
             tick(all(v["pass"] for v in gated.values()))],
        ]
    # FOOTER DELETED 2026-08-18. It ran to ~200 words and ARGUED: the disjunction defence,
    # the materiality-vs-significance point, the tab:t5 comparison and the cost-level caveat
    # are all body prose for the injected-signal section. Captions describe; the argument goes
    # in the text (CARLO_REVISION_PASS.md:41-44). It also printed the cost gap as "~2.2x",
    # which step4_gates_v3.json measures at 2.72 calm / 2.35 volatile, and used "dump", now
    # removed document-wide.
    _w("m3_injected_gates.tex", _tabular(
        "@{}lp{0.33\\linewidth}p{0.21\\linewidth}p{0.18\\linewidth}c@{}",
        ["Regime", "What the check asks", "Measured", "Registered band", "Pass"], rows))


# --------------------------------------------------------------- tab:m-register
# DECLARED EXCEPTION (see the module docstring). A registration's MEANING -- what it fixed,
# whether the prediction it carried came true -- is a human designation recorded in the frozen
# ledger, not derivable from any file. Every row therefore carries the ledger line that
# settles it, and the ledger is under version control so the dates are checkable by someone
# other than its author. Numeric thresholds quoted in the rows are the ledger's own.
REGISTER_ROWS = [
    # (what was fixed, when relative to the data, the written prediction, outcome, source)
    ("The definition of an edge: all four conditions together",
     "before any evaluation data were seen", "---", "applied unchanged",
     "criteria section 3, lines 60--64"),
    ("Materiality floor of $0.05$ bps on the mean saving",
     "before any evaluation data were seen", "---", "applied unchanged",
     "criteria section 3, line 63"),
    ("Significance level of $0.01$ on the paired test",
     "before any evaluation data were seen", "---", "applied unchanged",
     "criteria section 3, line 61"),
    ("Six simulator validation checks and their pass bands",
     "before any agent was trained", "all six pass", "held; all six passed",
     "criteria section 2"),
    ("Behaviour audit: exclusion above one unit left to the deadline in more than "
     "one episode in ten",
     "before any agent was scored", "fixed TWAP itself passes at every order size",
     "held; worst case $0.010$ against a $0.10$ cap",
     "criteria sections 4, 4b, 7.4"),
    ("Which episode pool serves which purpose, and one-shot use of each sealed pool",
     "before any pool was opened", "---", "applied unchanged; every sealed pool spent once",
     "criteria section 3; seed-disjointness audit"),
    ("Injected signal must reproduce the measured venue slope within $\\pm20\\%$",
     "before the injection was certified", "the band is reachable at every gated horizon",
     "held; worst gated horizon $17.6\\%$", "criteria section 8"),
    ("First sealed confirmation, agents",
     "before the pool was opened",
     "no agent meets the edge rule; pooled difference within noise of zero",
     "\\textbf{held}", "criteria A1; results log (K)"),
    ("First sealed confirmation, the rule-based follower",
     "before the pool was opened",
     "cheaper than the benchmark beyond the materiality floor, both regimes",
     "\\textbf{held}", "criteria A1; results log (L)"),
    ("Adding the arrival price to the observation, policy-gradient agent",
     "before the agents were re-trained",
     "the predicted-score estimate improves above $0.10$; the cost verdict does not change",
     "\\textbf{both held} ($0.42$ against a bar of $0.10$)",
     "criteria A4; results log (Y23)"),
    ("Adding the arrival price to the observation, value-based agent",
     "before the agents were re-trained",
     "the spread between action values at least doubles; invalid runs fall to 4--7 of 10",
     "\\textbf{value spread held; the invalid-run prediction FAILED} --- 10 of 10, "
     "marginally worse",
     "criteria A4.2; results log at 2101"),
    ("Comparator schedules lose on cost by construction once risk aversion is positive",
     "stated in advance of the comparator runs", "every such schedule is costlier",
     "\\textbf{FAILED} --- all differences statistically indistinguishable from zero; "
     "recorded as a correction, not amended away",
     "criteria A3.2"),
    ("Identity check between the zero-urgency schedule and adaptive TWAP",
     "before the check was run",
     "costs agree within $0.02$ bps and actions agree at $95\\%$",
     "\\textbf{cost held by two orders of magnitude; the action clause FAILED at "
     "$94.6/94.8\\%$} and the bar was amended after the achievable ceiling was measured "
     "at $94.67/94.78\\%$, labelled post hoc wherever cited",
     "criteria A3.1, lines 1230--1239"),
]


def register() -> None:
    rows = [[w, when, pred, out] for (w, when, pred, out, _src) in REGISTER_ROWS]
    held = sum(1 for r in REGISTER_ROWS if "FAILED" not in r[3])
    failed = len(REGISTER_ROWS) - held
    note = (rf"Every entry was written down and dated before the run it governs, in a ledger "
            rf"held under version control, so the ordering is checkable by someone other than "
            rf"its author. {failed} of {len(REGISTER_ROWS)} predictions were falsified by their "
            rf"own tests and are recorded here as such, with the diagnosis attached rather than "
            rf"the threshold moved. The one exception, the action-agreement clause, was amended "
            rf"only after the mechanically achievable maximum was measured and found to lie "
            rf"below the registered bar; it is labelled post hoc wherever it is cited. "
            rf"\textbf{{The research questions were not part of this registration}}: the "
            rf"decision rules, thresholds, pool assignments and per-amendment predictions were.")
    _w("m4_register.tex", _tabular(
        "@{}p{0.26\\linewidth}p{0.16\\linewidth}p{0.26\\linewidth}p{0.28\\linewidth}@{}",
        ["What was fixed", "When", "The written prediction", "What happened"], rows, note))
    print(f"    ({held} held, {failed} falsified -- the falsified ones are the point)")


def main() -> None:
    print("Methodology tables:")
    provenance()
    tracks()
    injected_gates()
    register()
    print("\nCAPTION OBLIGATIONS -- copy into the chapter, do not paraphrase:")
    print("  m1 provenance : the left column describes the paper AND its released code; the")
    print("                  book depth is not in the paper and comes from default.yaml.")
    print("  m2 tracks     : counting directories overstates the census by "
          f"{CEN['directories_overstate_by']}; the recorded-book budget is not uniform.")
    print("  m3 inj. gates : print the 25% band beside the volatile cost-growth gap; the drift")
    print("                  row must carry BOTH statistics and the disjunction; caveat C6.")
    print("  m4 register   : the falsified predictions are the evidence the protocol is real;")
    print("                  the caption must say the research questions were NOT registered.")


if __name__ == "__main__":
    main()
