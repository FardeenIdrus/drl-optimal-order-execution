"""Table A1 -- what each environment's test could actually have detected.

WHY THIS TABLE EXISTS. Every headline verdict in the Results chapter is a null. A null is only
informative if the instrument could have seen the effect had it been present, and until now
the chapter asserted that for one environment and said nothing about the other two. This table
states it for all three, per cell, and states plainly where the answer is unflattering.

THREE CAPTION OBLIGATIONS, ALL CARRIED BELOW AND NONE OPTIONAL.
  1. The minimum detectable effect is estimated from five seeds, so the INTERVAL is the claim
     and the point estimate is not. A bare point estimate from five seeds would assert a
     precision five seeds cannot carry.
  2. Cells with fewer than three audit-valid seeds are NOT QUOTABLE and are shown as such, not
     silently dropped: their spread rests on one degree of freedom and their interval runs to
     absurd values. The seeds were excluded by the behaviour audit BEFORE any cost comparison,
     because an agent that never finished the parent order has no comparable cost.
  3. The dispersion these figures rest on was measured on more than one block per environment,
     and the table reports that spread. This is what licenses applying a sensitivity computed
     on one block to a verdict issued on another -- in a chapter whose Part C finding is that
     blocks differ, that transfer cannot be assumed, and here it is not.

Source: oxford_l4/power_analysis_multiblock.json (written by
        reports/diagnostics/power_analysis_multiblock.py)
"""
from __future__ import annotations

import json
from pathlib import Path

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent / "ta1_power.tex"
SRC = S / "power_analysis_multiblock.json"

ENV_LABEL = {"frozen": "Recorded order books", "reacting": "Reacting simulator",
             "injected": "Injected predictability"}
# The block a verdict of record was issued on, per environment: that is the row the reader
# needs. Other blocks exist only to establish that dispersion transfers, and are summarised
# in the transfer column rather than given their own rows.
VERDICT_BLOCK = {"frozen": "test", "reacting": "dev_5e6", "injected": "dev_18e6"}
CELL_LABEL = {"calm": "calm", "volatile": "volatile",
              "size96.57": "96.6 BTC", "size193.13": "193 BTC", "size386.27": "386 BTC"}


def fmt(x, nd=3):
    return "---" if x is None else f"{x:.{nd}f}"


def main() -> None:
    doc = json.loads(SRC.read_text())
    cells = doc["cells"]
    xb = {(r["env"], r["algo"], r["group"]): r for r in doc["cross_block"]}

    rows = []
    for env in ("frozen", "reacting", "injected"):
        sel = [c for c in cells if c["env"] == env and c["block"] == VERDICT_BLOCK[env]]
        for c in sorted(sel, key=lambda c: (c["algo"], c["group"])):
            algo = c["algo"].upper().replace("DQN", "DQN").replace("PPO", "PPO")
            cell = CELL_LABEL.get(c["group"], c["group"])
            if c.get("n_valid", 0) < 2:
                rows.append((env, algo, cell, c["n_valid"], None, None, None, None,
                             None, None, "no test exists"))
                continue
            t = xb.get((env, c["algo"], c["group"]))
            rows.append((env, algo, cell, c["n_valid"], c["sd_episode_mean_bps"],
                         c["sd_across_seed_bps"], c["inter_seed_corr"], c["mde_80_bps"],
                         (c["mde_80_lo_bps"], c["mde_80_hi_bps"]),
                         c["power_at_materiality_0p05"],
                         None if c["quotable"] else "NOT QUOTABLE",
                         ) + ((t["sd_spread_pct"],) if t else (None,)))

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{llrrrrrrr}")
    L.append(r"\toprule")
    L.append(r"& & \multicolumn{1}{c}{seeds} & \multicolumn{1}{c}{$\sigma_{\text{ep}}$} "
             r"& \multicolumn{1}{c}{$\sigma_{\text{seed}}$} & \multicolumn{1}{c}{$\rho$} "
             r"& \multicolumn{1}{c}{MDE} & \multicolumn{1}{c}{95\% interval} "
             r"& \multicolumn{1}{c}{power} \\")
    L.append(r"& & \multicolumn{1}{c}{valid} & \multicolumn{1}{c}{(bps)} "
             r"& \multicolumn{1}{c}{(bps)} & & \multicolumn{1}{c}{(bps)} "
             r"& \multicolumn{1}{c}{on MDE} & \multicolumn{1}{c}{at 0.05} \\")
    L.append(r"\midrule")

    last_env = None
    for r in rows:
        env, algo, cell, n, sd_ep, sd_seed, rho, m80, ivl, pwr, flag, *rest = r
        spread = rest[0] if rest else None
        if env != last_env:
            if last_env is not None:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{ENV_LABEL[env]}}}}} \\")
            last_env = env
        if sd_ep is None:
            L.append(rf"\quad {algo} & {cell} & {n} & \multicolumn{{6}}{{l}}"
                     rf"{{\itshape fewer than two audit-valid seeds; no across-seed test "
                     rf"exists}} \\")
            continue
        ivl_s = rf"[{ivl[0]:.3f}, {ivl[1]:.3f}]"
        note = rf" \textsuperscript{{\dag}}" if flag else ""
        L.append(rf"\quad {algo} & {cell}{note} & {n} & {fmt(sd_ep)} & {fmt(sd_seed,4)} "
                 rf"& {fmt(rho,2)} & {fmt(m80)} & {ivl_s} & {fmt(pwr,3)} \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    # ---- the transfer evidence, as its own short block -----------------------------
    L.append(r"\vspace{4pt}")
    L.append(r"\begin{tabular}{llr}")
    L.append(r"\multicolumn{3}{l}{\footnotesize\textbf{Dispersion measured on more than one "
             r"block, per environment}} \\")
    L.append(r"\toprule")
    L.append(r"environment & blocks compared & spread in $\sigma_{\text{ep}}$ \\")
    L.append(r"\midrule")
    for env in ("frozen", "reacting", "injected"):
        sub = [r for k, r in xb.items() if k[0] == env]
        if not sub:
            continue
        nb = len(next(iter(sub))["sd_episode_by_block"])
        worst = max(s["sd_spread_pct"] for s in sub)
        L.append(rf"{ENV_LABEL[env]} & {nb} & $\leq {worst:.1f}\%$ \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    L.append(r"\caption[What each design could have detected]{\textbf{The smallest cost "
             r"difference each design could have detected, by environment and cell.} "
             r"Every headline verdict in this chapter is a null, and a null is informative "
             r"only if the instrument could have seen the effect had it been present. "
             r"\textbf{MDE} is the effect size the registered across-seed test would detect "
             r"with 80\% probability at $\alpha=0.05$, computed from the exact non-central "
             r"$t$ distribution rather than a normal approximation, which at five seeds "
             r"overstates power substantially. \textbf{The interval is the claim, not the "
             r"point estimate}: with five seeds the across-seed spread is itself estimated on "
             r"four degrees of freedom, and the interval shown propagates that uncertainty. "
             r"$\sigma_{\text{ep}}$ is per-episode cost dispersion, $\sigma_{\text{seed}}$ "
             r"the spread across independently trained seeds, and $\rho$ the correlation "
             r"between seeds' per-episode difference series induced by common random numbers. "
             r"Shared noise shifts every seed mean together and so contributes nothing to "
             r"their spread, which is why the test is powerful against differences between "
             r"policies and powerless against a shift common to all of them. "
             r"\textsuperscript{\dag}\,\textbf{Not quotable}: fewer than three seeds survived "
             r"the behaviour audit. Unfinished orders are forced to complete at the deadline, "
             r"and the audit removes agents that rely on that forced purchase \emph{before} "
             r"any cost is compared; below three survivors the spread rests on one degree of "
             r"freedom. The lower panel reports the one thing that licenses applying a "
             r"sensitivity measured on one block to a verdict issued on another: dispersion "
             r"was measured on two or three blocks per environment and agrees to within "
             r"7.3\% everywhere. Recorded order books are the least sensitive of the three "
             r"environments, and the cause is stated per cell rather than generalised. The "
             r"weakest remaining cell is the value-based learner at 193\,BTC on the sealed "
             r"split, with an MDE of 0.623\,bps and a 0.075 probability of detecting a "
             r"difference at the 0.05\,bps materiality threshold; its across-seed spread is "
             r"85\% policy-to-policy variation, so that cell is a convergence problem rather "
             r"than a sampling one. Other cells in the same environment resolve to "
             r"0.032\,bps. This limitation is stated here rather than left implicit.}")
    L.append(r"\label{tab:power}")
    L.append(r"\end{table}")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(rows)} cells)")
    for r in rows:
        flag = r[10] or ""
        print(f"  {r[0]:<9}{r[1]:<5}{r[2]:<10} n={r[3]} "
              f"MDE={fmt(r[7])} {flag}")


if __name__ == "__main__":
    main()
