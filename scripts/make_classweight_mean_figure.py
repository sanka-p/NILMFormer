#################################################################################################################
#
# @description : The class-weighting result on its own -- mean F1 vs w_off, three scoring rules.
#
# This is panel A of fig_classweight as a standalone figure, for use where the
# per-appliance breakdown is not wanted.
#
# PROVENANCE: the values are read from figures/fig_classweight.csv, which
# make_classweight_figure.py writes while recomputing them from the stored predictions.
# This script therefore does NOT touch the .pt files -- it renders the same numbers the
# two-panel figure was built from, so the two cannot disagree. Regenerate the CSV first
# if the underlying runs change. The per-cell appliance count is asserted below so a
# truncated or partial CSV fails loudly instead of plotting a silently different mean.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_classweight_mean_figure
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

# Same style block as the other paper figures so they read as a set.
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

#: Categorical slots 1-3 of the reference palette, unmodified and in order; the same
#: hues and markers the two-panel figure gives these three rules, so a reader moving
#: between the figures keeps their bearings. Identity is carried by the legend and a
#: distinct marker shape, never by colour alone, and the values are written as CSV.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID = "#e4e3df"

SCORINGS = [
    ("stock",      "Stock threshold",          C_BLUE,   "o"),
    ("prevalence", "Prevalence-matched",       C_ORANGE, "s"),
    ("prev_pp",    "Prevalence + post-proc.",  C_AQUA,   "^"),
]
N_APPLIANCES = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("figures/fig_classweight.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_classweight_mean"))
    a = p.parse_args()

    if not a.csv.exists():
        raise SystemExit(f"{a.csv} not found -- run scripts.make_classweight_figure first")
    cells = collections.defaultdict(list)
    with open(a.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            cells[(r["scoring"], float(r["w_off"]))].append(float(r["f1"]))

    xs = sorted({w for _, w in cells})
    for (key, w), v in sorted(cells.items()):
        if len(v) != N_APPLIANCES:
            raise SystemExit(f"{key} @ w_off={w}: {len(v)} appliances, expected "
                             f"{N_APPLIANCES} -- the CSV is partial, refusing to plot "
                             f"a mean over a different set")

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True,
                           facecolor="white")
    rows = []
    for key, label, colour, marker in SCORINGS:
        ys = [float(np.mean(cells[(key, w)])) for w in xs]
        ax.plot(xs, ys, color=colour, marker=marker, markersize=8, linewidth=2,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        rows += [dict(scoring=key, w_off=w, mean_f1=round(y, 2))
                 for w, y in zip(xs, ys)]
        print(f"{label:<26} " + "  ".join(f"{w:g}:{y:5.2f}" for w, y in zip(xs, ys)))

    ax.set_title(f"Mean F1 over {N_APPLIANCES} appliances", color=INK, pad=10)
    ax.set_ylabel("F1 (%)")
    ax.set_xlabel(r"OFF-class weight  $w_{off}$   (log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.88, 11.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([("%g" % x) for x in xs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_edgecolor(INK_MUTED)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_2)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["scoring", "w_off", "mean_f1"])
        wr.writeheader(); wr.writerows(rows)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=240 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print(f"\nwrote {a.out}.png / .pdf / .csv  (source: {a.csv})")


if __name__ == "__main__":
    main()
