#################################################################################################################
#
# @description : Post-processing at the UNCHANGED stock threshold, as w_off varies.
#
# The three-line fig_classweight pairs post-processing with prevalence matching, so it
# cannot say which of the two did the work. This figure holds the threshold at the
# stock appliance value and turns only the duration rules on and off.
#
# Panel A is the mean. Panel B is the per-appliance CHANGE, because the mean hides a
# sign flip: at a badly placed threshold the duration rules can make things WORSE --
# bridging merges a low-threshold haze into long blocks that min_on then protects.
# Dishwasher (floor 10 W) is the case in point.
#
# PROVENANCE: values come from figures/fig_classweight.csv, which
# make_classweight_figure.py writes while recomputing them from the stored
# predictions. The `stock_pp` rows are collected there but deliberately not drawn in
# that figure, so its three published lines stay unchanged.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_postprocess_lines_figure
#
#################################################################################################################

import argparse
import collections
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for _noisy in ("fontTools", "fontTools.subset", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "black",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11,
    "axes.linewidth": 1.0, "lines.linewidth": 1.4,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
})

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID = "#e4e3df"

#: Panel A: the two decision rules, categorical slots 1-2 in order.
LINES = [("stock",    "Stock threshold",                    C_BLUE,   "o"),
         ("stock_pp", "Stock threshold + post-processing",  C_ORANGE, "s")]
#: Panel B: the appliances, same hues and markers the other figures give them.
APP_STYLE = [("WashingMachine", "Washing m.", C_BLUE,   "o"),
             ("Dishwasher",     "Dishwasher", C_ORANGE, "s"),
             ("Microwave",      "Microwave",  C_AQUA,   "^"),
             ("Fridge",         "Fridge",     C_YELLOW, "D")]


def _decorate(ax, xs):
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([("%g" % x) for x in xs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"OFF-class weight  $w_{off}$   (log scale)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_edgecolor(INK_MUTED)
    ax.tick_params(colors=INK_2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("figures/fig_classweight.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_postprocess_lines"))
    a = p.parse_args()
    if not a.csv.exists():
        raise SystemExit(f"{a.csv} not found -- run scripts.make_classweight_figure first")

    tab = collections.defaultdict(dict)       # (scoring, appliance) -> {w_off: f1}
    with open(a.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            tab[(r["scoring"], r["appliance"])][float(r["w_off"])] = float(r["f1"])
    if not any(k[0] == "stock_pp" for k in tab):
        raise SystemExit(f"{a.csv} has no 'stock_pp' rows -- regenerate it with the "
                         f"current scripts.make_classweight_figure")

    apps = [x for x, _, _, _ in APP_STYLE]
    xs = sorted(tab[("stock", apps[0])])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True,
                                   facecolor="white")

    rows = []
    for key, label, colour, marker in LINES:
        ys = [float(np.mean([tab[(key, x)][w] for x in apps])) for w in xs]
        axA.plot(xs, ys, color=colour, marker=marker, markersize=8, linewidth=2,
                 markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        rows += [dict(panel="mean", series=key, appliance="MEAN", w_off=w,
                      value=round(y, 2)) for w, y in zip(xs, ys)]
        print(f"{label:<36} " + "  ".join(f"{w:g}:{y:5.2f}" for w, y in zip(xs, ys)))
    axA.set_title(f"Mean F1 over {len(apps)} appliances", color=INK, pad=10)
    axA.set_ylabel("F1 (%)")
    axA.set_xlim(0.88, 11.5)
    _decorate(axA, xs)
    axA.legend(frameon=False, loc="lower left", labelcolor=INK_2)

    print("\nchange from post-processing at the stock threshold:")
    for app, label, colour, marker in APP_STYLE:
        ys = [tab[("stock_pp", app)][w] - tab[("stock", app)][w] for w in xs]
        axB.plot(xs, ys, color=colour, marker=marker, markersize=8, linewidth=2,
                 markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        rows += [dict(panel="delta", series=app, appliance=app, w_off=w,
                      value=round(y, 2)) for w, y in zip(xs, ys)]
        print(f"  {label:<14} " + "  ".join(f"{w:g}:{y:+6.2f}" for w, y in zip(xs, ys)))
    axB.axhline(0.0, color=INK_MUTED, linewidth=1.2, zorder=2)
    axB.set_title("Change from post-processing, per appliance", color=INK, pad=10)
    axB.set_ylabel(r"$\Delta$ F1 (percentage points)")
    axB.set_xlim(0.88, 11.5)
    _decorate(axB, xs)
    # Lower-left is where the Dishwasher line lives (it is the only negative series),
    # so the legend goes in the empty mid-band between it and WashingMachine.
    axB.legend(frameon=False, loc="center left", labelcolor=INK_2)

    fig.suptitle("Duration rules at the unchanged stock threshold",
                 fontsize=16, color=INK)
    fig.text(0.5, -0.03,
             "No threshold is re-tuned: only the duration rules are switched on. "
             "Values above the zero line in panel B are appliances post-processing "
             "helps; below it, smoothing at a badly placed threshold merges noise into "
             "long runs that min_on then protects.",
             ha="center", va="top", fontsize=9.5, color=INK_2)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["panel", "series", "appliance",
                                            "w_off", "value"])
        wr.writeheader(); wr.writerows(rows)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=240 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print(f"\nwrote {a.out}.png / .pdf / .csv  (source: {a.csv})")


if __name__ == "__main__":
    main()
