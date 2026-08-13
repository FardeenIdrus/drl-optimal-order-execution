"""Two measurements the Methodology chapter needs, neither of which existed.

1. WHY THE SIMULATOR USES TEN LEVELS EACH SIDE WHEN THE DATA CARRIES TWENTY.
   The chapter states both numbers, so it has to say why the second ten were not modelled.
   MEASURED AND THE OBVIOUS ANSWER IS WRONG. The first hypothesis was that depth beyond ten
   is thin and intermittent, so per-level rates could not be estimated there. It is false:
   no level in the top twenty is ever empty, and levels 11-20 hold MORE resting size than
   3-10. The honest justification is the one measured below instead -- the top ten levels
   already contain far more size than the largest quantity this study ever asks the book to
   absorb, so the unmodelled region is one the agent cannot reach. Read straight off the
   December half-second book, streamed one day at a time.

2. WHAT THE MATERIALITY FLOOR IS WORTH IN REAL TERMS.
   The registered floor is 0.05 bps. The record justifies needing *a* floor (pairing across
   thousands of episodes makes trivial differences significant) but never says why this one.
   Measured here: the floor as a fraction of what the trade actually costs, using the
   already-certified simulator gate measurements of real book-walking cost at the traded
   sizes, and as a fraction of the venue's own half-spread.

RAM: the book files are 1.3 GB across 32 days at 85 columns. Only the size columns are
read, one day at a time, and accumulated into running sums. Nothing is concatenated.

Output: scratch_hyperliquid/oxford_l4/depth_and_materiality.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OX = SCRATCH / "oxford_l4"
BOOK = OX / "book_05s_v2"
GATES = OX / "signal" / "gates" / "sigext_gates_v4c_PASS.json"
TICK = OX / "tick_class" / "tick_class_measurement.json"
OUT = OX / "depth_and_materiality.json"

LEVELS = 20
SIM_K = 10


def depth_by_level() -> dict:
    """Running totals of resting size and empty-level counts, per level, per side."""
    cols = [f"{s}_sz_{i}" for i in range(1, LEVELS + 1) for s in ("bid", "ask")]
    tot = {c: 0.0 for c in cols}
    empty = {c: 0 for c in cols}
    rows = 0
    days = sorted(BOOK.glob("*.parquet"))
    if not days:
        raise SystemExit(f"no book files under {BOOK}")
    for d in days:
        t = pq.read_table(d, columns=cols)
        rows += t.num_rows
        for c in cols:
            a = t[c].to_numpy(zero_copy_only=False).astype("float64")
            a = np.nan_to_num(a, nan=0.0)
            tot[c] += float(a.sum())
            empty[c] += int((a <= 0).sum())
        del t

    per_level = {}
    for i in range(1, LEVELS + 1):
        b, a = f"bid_sz_{i}", f"ask_sz_{i}"
        per_level[i] = {
            "total_size": tot[b] + tot[a],
            "empty_frac": (empty[b] + empty[a]) / (2 * rows),
        }
    grand = sum(v["total_size"] for v in per_level.values())
    for i, v in per_level.items():
        v["share_of_visible_size"] = v["total_size"] / grand

    top = sum(per_level[i]["share_of_visible_size"] for i in range(1, SIM_K + 1))
    mean_btc_per_level = {i: per_level[i]["total_size"] / (2 * rows) for i in per_level}
    cum_top_k = sum(mean_btc_per_level[i] for i in range(1, SIM_K + 1))
    empty_top = max(per_level[i]["empty_frac"] for i in range(1, SIM_K + 1))
    empty_bot = min(per_level[i]["empty_frac"] for i in range(SIM_K + 1, LEVELS + 1))
    return {
        "source": str(BOOK), "n_grid_rows": rows, "n_days": len(days),
        "levels_in_data": LEVELS, "levels_in_simulator": SIM_K,
        "per_level": per_level,
        "top_k_share_of_visible_size": top,
        "mean_btc_per_level_one_side": mean_btc_per_level,
        "mean_cumulative_btc_through_top_k_one_side": cum_top_k,
        "worst_empty_frac_in_top_k": empty_top,
        "best_empty_frac_below_k": empty_bot,
    }


def reachability(depth: dict, sizes: list[float], decisions: int, max_pace: float) -> dict:
    """Can the agent ever reach level eleven? Compare the top-ten depth to what it trades."""
    cum = depth["mean_cumulative_btc_through_top_k_one_side"]
    largest_order = max(sizes)
    # The largest quantity a single decision can demand: the fastest pace on the largest
    # order at the primary horizon. This is what one step asks the visible book to absorb.
    largest_slice = max_pace * largest_order / decisions
    return {
        "mean_depth_through_top_ten_one_side_btc": cum,
        "largest_order_btc": largest_order,
        "largest_single_decision_btc": largest_slice,
        "depth_covers_largest_single_decision_times": cum / largest_slice,
        "depth_covers_whole_largest_order_times": cum / largest_order,
        "note": ("the unmodelled levels 11-20 sit beyond a region the traded quantities do "
                 "not exhaust; the simulator models the part of the book the agent reaches"),
    }


def materiality() -> dict:
    """0.05 bps against what a trade at these sizes actually costs, and against the spread."""
    g = json.loads(GATES.read_text())
    floor = 0.05
    out = {"registered_floor_bps": floor, "source_gates": str(GATES)}
    for reg in ("calm", "volatile"):
        g2 = g["G2_cost_vs_size_rev1"]["regimes"][reg]
        g3 = g["G3_benchmark_sanity_rev1"]["regimes"][reg]
        real = {int(k): v for k, v in g2["real_bps"].items()}
        largest = real[max(real)]
        out[reg] = {
            "real_book_walk_cost_bps_by_units": real,
            "real_cost_at_largest_probe_bps": largest,
            "floor_as_frac_of_real_cost_at_largest_probe": floor / largest,
            "twap_cost_in_sim_bps": g3["driftfree_twap_mean_bps"],
        }
    if TICK.exists():
        tc = json.loads(TICK.read_text())
        sp = tc["spread"]["full_month"]["mean_spread_bps"]
        if sp:
            out["venue_mean_spread_bps"] = sp
            out["floor_as_frac_of_half_spread"] = floor / (sp / 2)
    return out


def main() -> None:
    d = depth_by_level()
    mm = json.loads((OX / "methodology_measurements.json").read_text())
    pd_ = mm["census"]["primary_design"]
    sizes = mm["census"]["reacting_simulator"]["order_sizes_btc"]
    max_pace = max(mm["action_grid"]["multiples"])
    r = reachability(d, sizes, pd_["decisions_per_episode"], max_pace)
    m = materiality()
    OUT.write_text(json.dumps({"depth": d, "reachability": r, "materiality": m}, indent=1))

    print(f"DEPTH  {d['n_grid_rows']:,} half-second rows over {d['n_days']} days")
    print(f"  top {SIM_K} levels hold {d['top_k_share_of_visible_size']:.1%} of visible resting size")
    print(f"  worst empty fraction inside the top {SIM_K}: {d['worst_empty_frac_in_top_k']:.1%}")
    print(f"  best empty fraction below it:              {d['best_empty_frac_below_k']:.1%}")
    print("  per level (share of size | empty):")
    for i in range(1, LEVELS + 1):
        v = d["per_level"][i]
        mark = "  <- simulator stops here" if i == SIM_K else ""
        print(f"    L{i:2d}  {v['share_of_visible_size']:6.2%}  {v['empty_frac']:6.2%}{mark}")

    print(f"\nREACHABILITY  mean depth through the top ten, one side: "
          f"{r['mean_depth_through_top_ten_one_side_btc']:.1f} BTC")
    print(f"  largest whole order {r['largest_order_btc']:g} BTC "
          f"-> depth covers it {r['depth_covers_whole_largest_order_times']:.1f}x")
    print(f"  largest single decision {r['largest_single_decision_btc']:.3f} BTC "
          f"-> depth covers it {r['depth_covers_largest_single_decision_times']:,.0f}x")

    print(f"\nMATERIALITY  floor {m['registered_floor_bps']} bps")
    for reg in ("calm", "volatile"):
        r = m[reg]
        print(f"  {reg}: real book-walk cost {r['real_cost_at_largest_probe_bps']:.3f} bps at the "
              f"largest probe -> floor is {r['floor_as_frac_of_real_cost_at_largest_probe']:.1%} of it")
    if "floor_as_frac_of_half_spread" in m:
        print(f"  half-spread: floor is {m['floor_as_frac_of_half_spread']:.1%} of it "
              f"(mean spread {m['venue_mean_spread_bps']:.3f} bps)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
