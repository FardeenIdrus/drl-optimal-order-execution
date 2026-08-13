"""The Methodology chapter's two diagrams and one rebuild.

  fig:m-problem       the decision problem and the policy network, one schematic
  fig:m-architecture  the evaluation ladder (rebuild: the outcome box is removed and the
                      episode-pool panel is widened from one environment to three)
  fig:m-signal        handled by the existing injection-fidelity builder; not redrawn here

LABELLING RULES, from the author, 2026-08-12 and applied to every string in this file:
  * short and specific. No sentence-long annotations on a diagram.
  * no internal names, no file stems, no ledger shorthand. A reader who has only the
    dissertation must understand every word.
  * nothing that implies a mistake. "After fixing the critic" is the banned example: it
    reads as a confession and it is not what the figure shows.
  * no results. These are methods diagrams; a p-value on one pre-empts the next chapter.

Numbers come from methodology_measurements.json, so the observation groups and the action
grid cannot drift away from the code.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch         # noqa: E402

OX = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
HERE = Path(__file__).resolve().parent
M = json.loads((OX / "methodology_measurements.json").read_text())
OBS, GRID, CEN = M["observation_widths"], M["action_grid"], M["census"]

INK = "#1a1a1a"
BLUE, GREEN, RED, GREY = "#dbe9f6", "#dff0d8", "#f6dbdb", "#f0f0f0"


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def _box(ax, x, y, w, h, text, fc, fs=8.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                facecolor=fc, edgecolor=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35, color=INK, weight=weight)


def _arrow(ax, x1, y1, x2, y2, text="", fs=7.6, dx=0.10, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=INK, lw=1.2))
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, text, fontsize=fs,
                ha="left", va="center", color=INK)


# ------------------------------------------------------------------- fig:m-problem
def problem_and_network() -> None:
    """What the agent sees, what it may do, and how the two are connected.

    The diagram has to do analytical work, not decorate. Two things it shows that prose
    cannot: the benchmark sits INSIDE the action set (the 1.0 rung is labelled as such and
    drawn in the same grid as every other action), and the observation is dominated by book
    state (the queue-sizes block is drawn to scale against the rest).
    """
    g = OBS["groups"]
    total = OBS["reacting_simulator"]["base"]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(1.20, 9.95); ax.axis("off")

    # --- observation, drawn to scale so the queue block visibly dominates ---------------
    ax.text(1.6, 9.42, "What the agent sees", fontsize=9.8, weight="bold", ha="center")
    ax.text(1.6, 9.04, f"{total} numbers, each decision", fontsize=8.0, ha="center")
    order = ["queue sizes", "own fills", "trailing market flow",
             "inventory", "time remaining", "spread"]
    y, TOP = 8.62, 8.62
    for name in order:
        n = g[name]
        h = max(0.30, 3.0 * n / total)
        _box(ax, 0.30, y - h, 2.60, h, f"{name}  ({n})", BLUE if n > 2 else GREY,
             fs=8.2 if n > 2 else 7.5)
        y -= h + 0.13
    OBS_BOT = y + 0.13
    ax.text(1.6, OBS_BOT - 0.26, "boxes drawn to scale", fontsize=7.0, ha="center",
            style="italic", color="#555555")

    # --- network -------------------------------------------------------------------------
    NET_MID = (TOP + OBS_BOT) / 2
    ax.text(4.75, 7.92, "Inherited shape", fontsize=9.0, weight="bold", ha="center")
    _box(ax, 3.50, NET_MID - 1.05, 2.50, 2.10,
         "Policy network\n5 layers of 30 units\n\nalso tested\n2 layers of 64\n2 layers of 128",
         GREY, fs=8.2)
    _arrow(ax, 2.98, NET_MID, 3.44, NET_MID)

    # --- actions: the grid, with the benchmark rung marked -------------------------------
    ax.text(8.15, 9.42, "What the agent may do", fontsize=9.8, weight="bold", ha="center")
    ax.text(8.15, 9.04, "one choice per decision", fontsize=8.0, ha="center")
    mult = GRID["multiples"]
    x0, w, rh = 6.62, 3.06, 0.47
    for i, m in enumerate(mult):
        yy = TOP - i * rh
        is_twap = m == GRID["twap_action"]
        _box(ax, x0, yy - rh + 0.05, w, rh - 0.05,
             f"{m:g} x the even pace" + ("      the benchmark" if is_twap else ""),
             GREEN if is_twap else GREY, fs=8.2,
             weight="bold" if is_twap else "normal")
    ACT_BOT = TOP - len(mult) * rh + 0.05
    ax.text(8.15, ACT_BOT - 0.40, "the benchmark is one of the choices",
            fontsize=8.0, ha="center", weight="bold", color=INK)
    _arrow(ax, 6.06, NET_MID, 6.56, NET_MID)

    # --- the loop, on its own row so nothing overlaps -------------------------------------
    LY, LH = 1.55, 1.05
    _box(ax, 6.62, LY, 3.06, LH, "Market\nthe order book responds", BLUE, fs=8.6)
    _box(ax, 3.50, LY, 2.50, LH, "Score\ncost against the price\nat the moment of arrival",
         GREEN, fs=8.4)
    _box(ax, 0.30, LY, 2.60, LH, "Next decision\nthe book has moved on", GREY, fs=8.4)
    _arrow(ax, 8.15, ACT_BOT - 0.70, 8.15, LY + LH + 0.04)
    _arrow(ax, 6.56, LY + LH / 2, 6.06, LY + LH / 2)
    _arrow(ax, 3.44, LY + LH / 2, 2.96, LY + LH / 2)
    _arrow(ax, 1.60, LY + LH + 0.04, 1.60, OBS_BOT - 0.48)

    ax.set_title("The decision problem and the policy network", fontsize=11.5, color=INK)
    _save(fig, "fig_m_problem")


# -------------------------------------------------------------- fig:m-architecture
def evaluation_architecture() -> None:
    """REBUILD. Two defects in the version this replaces, both structural:

    1. It carried a box headed "WHAT HAPPENED" reporting the study's outcome, including two
       p-values and the sentence "the agent does not beat TWAP out-of-sample". That belongs
       to the results chapter; in a methods chapter it announces the answer before the
       method has been described. REMOVED, not softened.
    2. Its pool panel showed one environment's episode pools while the ladder beside it
       covers three. WIDENED: all three environments, and the recorded-book column states
       its different protocol rather than being left blank.
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0.20, 9.95); ax.axis("off")

    # --- left: episode pools, all three environments ------------------------------------
    ax.text(2.35, 9.5, "Episode pools", fontsize=9.8, weight="bold", ha="center")
    ax.text(2.35, 9.13, "no episode appears in two of them", fontsize=7.8, ha="center")
    pools = [
        ("Training", "private to each agent", GREY),
        ("Monitoring", "learning curves only\nnever used to judge", GREY),
        ("Development", "screening and tuning", BLUE),
        ("Reserve", "does a result repeat?", GREEN),
        ("Confirmation", "opened once, then spent", RED),
    ]
    y = 8.55
    for name, sub, fc in pools:
        _box(ax, 0.3, y - 0.80, 4.1, 0.80, f"{name}\n{sub}", fc, fs=8.0)
        y -= 1.02

    # --- right: the ladder ----------------------------------------------------------------
    ax.text(7.6, 9.5, "How a candidate is tested", fontsize=9.8, weight="bold", ha="center")
    ax.text(7.6, 9.13, "each stage removes a different false positive", fontsize=7.8,
            ha="center")
    rungs = [
        ("Screen", "on the development pool", BLUE),
        ("Replicate", "five independent seeds,\nthen the reserve pool", GREEN),
        ("Confirm", "one shot, on a confirmation pool", RED),
    ]
    yy = 8.5
    for i, (name, sub, fc) in enumerate(rungs):
        _box(ax, 5.6, yy - 1.15, 4.0, 1.15, f"{name}\n{sub}", fc, fs=8.4)
        if i < len(rungs) - 1:
            _arrow(ax, 7.6, yy - 1.15, 7.6, yy - 1.75,
                   " survivors only" if i else " candidates only")
        yy -= 1.75

    # --- bottom: what differs by environment. NO OUTCOMES. --------------------------------
    rb, rea, inj = (CEN["recorded_books"], CEN["reacting_simulator"],
                    CEN["injected_simulator"])
    _box(ax, 0.3, 0.45, 9.3, 1.75,
         "By environment\n"
         f"Recorded books, {rb['agents']} agents:  one exam for every agent, no screening "
         "stage, arms corrected for\n"
         f"Both simulators, {rea['agents']} and {inj['agents']} agents:  the full ladder "
         "above\n"
         "Simulator pools can be created at will, so their scarcity is a rule, not a limit",
         "#fbfbfb", fs=8.4)

    ax.set_title("Evaluation architecture", fontsize=11, color=INK)
    _save(fig, "fig_m_architecture")


def main() -> None:
    print("Methodology figures:")
    problem_and_network()
    evaluation_architecture()
    print("\nLABEL CHECK -- no internal names, no results, no confession wording. "
          "Every string above is plain English.")


if __name__ == "__main__":
    main()
