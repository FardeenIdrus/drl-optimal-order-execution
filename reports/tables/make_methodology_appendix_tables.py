"""The Methodology chapter's appendix tables.

  tab:a-observation   every observation input, named, for all three environments

WHY IT EXISTS. The chapter states three observation widths (7, 27, 28, and the two
amended widths) and enumerates none of them. A referee reproducing the study cannot
recover the state vector from the document. The six-group prose in section 4.2 is the
right level for the body and is not a specification.

NO WIDTH IS TYPED HERE. Every count is read from methodology_measurements.json, which
reads them from the code. The definitions below describe transformations that are
verified against source, with the file and line recorded beside each block.

Output: reports/tables/a1_observation.tex  (then sync_to_dissertation.py)
"""
from __future__ import annotations

import json
from pathlib import Path

OX = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
HERE = Path(__file__).resolve().parent
M = json.loads((OX / "methodology_measurements.json").read_text())
OBS = M["observation_widths"]

RB = OBS["recorded_books"]["base"]              # 7
REA = OBS["reacting_simulator"]["base"]         # 27
REA_AMD = OBS["reacting_simulator"]["with_arrival_price"]
INJ = OBS["injected_simulator"]["base"]         # 28
INJ_AMD = OBS["injected_simulator"]["with_arrival_price"]
K = OBS["K_per_side"]                           # 10

# --- Panel A. real_data_env.py:278-289 (_obs), data/features.py:100-116 -------------
PANEL_A = [
    ("1", "Inventory remaining", r"$2\,(\text{remaining} / \text{initial}) - 1$"),
    ("2", "Time remaining", r"$2\,(\text{decisions left} / \text{total}) - 1$"),
    ("3", "Spread", r"mean spread $/$ mid-quote $\times 10^{4}$, in basis points"),
    ("4", "Queue imbalance", r"(bid $-$ ask size) $/$ (bid $+$ ask size), mean sizes over the top five levels"),
    ("5", "Recent return", r"log change in the mid-quote over the preceding five minutes"),
    ("6", "Rolling volatility", r"square root of the realised variance accumulated over the preceding thirty minutes"),
    ("7", "Ask depth", r"mean ask size over the top five levels"),
]

# --- Panel B. reactive_env.py:298-318 (_observe) --------------------------------------
PANEL_B = [
    ("1", "Inventory", "Quantity remaining", r"remaining $/$ order size"),
    ("2", "Time", "Time remaining", r"$1 - (\text{decisions taken} / \text{total})$"),
    (f"3--{2 + 2 * K}", "Book", "Queue sizes",
     rf"$\log(1 + \text{{size}})$ at each of {K} price levels per side, in queue units"),
    (f"{3 + 2 * K}", "Book", "Spread", rf"spread in ticks $/\,{K}$"),
    (f"{4 + 2 * K}", "Own fills", "Last fill", r"queue units bought at the previous decision"),
    (f"{5 + 2 * K}", "Own fills", "Cumulative fills", r"fraction of the parent order bought so far"),
    (f"{6 + 2 * K}", "Market flow", "Traded volume",
     r"$\log(1 + \text{units traded})$ over the preceding thirty seconds"),
    (f"{7 + 2 * K}", "Market flow", "Net buying pressure",
     r"signed log of net units bought less sold, same window"),
]

# 0.88 not 0.94: the three internal column boundaries cost 6\tabcolsep, which 0.94 does not
# leave room for. At 0.94 the table ran 8.7pt over the text block.
HDR = r"\begin{tabular}{@{}p{0.06\linewidth}p{0.12\linewidth}p{0.20\linewidth}p{0.50\linewidth}@{}}"


def build() -> str:
    L = [HDR, r"\toprule",
         rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Panel A. Recorded order books, {RB} inputs}}}} \\",
         r"\midrule",
         r"\# & \multicolumn{2}{l}{Input} & Definition \\", r"\midrule"]
    for n, name, defn in PANEL_A:
        L.append(rf"{n} & \multicolumn{{2}}{{l}}{{{name}}} & {defn} \\")
    L += [r"\multicolumn{4}{@{}p{\dimexpr0.88\linewidth+6\tabcolsep}@{}}{\footnotesize Inputs 3 to 7 are standardised by a "
          r"normaliser fitted on the training split alone. Window lengths are specified in clock time "
          r"and converted to a bar count for each dataset, so the same interval applies at both "
          r"resolutions.} \\",
          r"\addlinespace[0.6em]", r"\midrule",
          rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Panel B. Reacting simulator, {REA} inputs}}}} \\",
          r"\midrule",
          r"\# & Group & Input & Definition \\", r"\midrule"]
    for n, grp, name, defn in PANEL_B:
        L.append(rf"{n} & {grp} & {name} & {defn} \\")
    L += [r"\multicolumn{4}{@{}p{\dimexpr0.88\linewidth+6\tabcolsep}@{}}{\footnotesize A queue unit is the average "
          r"order size measured at the level concerned, so its size in BTC differs by level. "
          r"The accumulation threshold in \Cref{sec:meth-problem} is one unit at the best price.} \\",
          r"\addlinespace[0.6em]", r"\midrule",
          rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Panel C. Injected simulator, {INJ} inputs}}}} \\",
          r"\midrule",
          rf"\multicolumn{{4}}{{@{{}}p{{\dimexpr0.88\linewidth+6\tabcolsep}}@{{}}}}{{Inputs 1 to {REA} exactly as Panel B, plus:}} \\",
          rf"{INJ} & Signal & Injected signal & value at the current step, in basis points \\",
          r"\bottomrule",
          r"\multicolumn{4}{@{}p{\dimexpr0.88\linewidth+6\tabcolsep}@{}}{\footnotesize The registered amendment adds one input "
          rf"to each simulator, giving {REA_AMD} and {INJ_AMD}: the price relative to arrival, "
          r"(mid-quote $-$ arrival mid-quote) $/$ arrival mid-quote $\times 10^{4}$, in basis points. "
          rf"The recorded-book environment was never re-trained under it and remains at {RB}.}} \\",
          r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> None:
    out = HERE / "a1_observation.tex"
    out.write_text(build())
    print(f"wrote {out.name}")
    print(f"  recorded books {RB} | reacting {REA} ({REA_AMD} amended) | "
          f"injected {INJ} ({INJ_AMD} amended) | K={K}")
    print("  no width typed: all read from methodology_measurements.json")


if __name__ == "__main__":
    main()
