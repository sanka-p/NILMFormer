#################################################################################################################
#
# @description : Class weighting vs temporal post-processing, as a 2x2.
#
# Two interventions were applied to the same detector and it was never shown what each
# one is worth on its own:
#
#   class weighting   a TRAINING change -- weighted MSE with w_off > w_on.
#   post-processing   a SCORING change  -- Kelly's duration rules on the predicted state.
#
# The threshold rule is held FIXED at prevalence-matched-on-validation in all four
# cells, so the figure isolates these two factors and nothing else. (Moving off the
# stock threshold is itself worth ~6.7 F1 -- 52.8 -> 59.5 -- and is held constant here
# rather than being folded into one of the bars.)
#
# w_off = 1.5 is used as the class-weighting cell because it is the BEST setting
# measured, under both scorings. That is selection on the outcome and therefore
# generous to class weighting: the bar is an upper bound on what it buys.
#
# PROVENANCE: values come from figures/fig_classweight.csv, which
# make_classweight_figure.py writes while recomputing them from the stored
# predictions. This script does not touch the .pt files, so the two figures cannot
# disagree. Regenerate that CSV first if the underlying runs change.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_intervention_figure
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

#: Baseline wears muted ink because it is the reference, not a series; the three
#: interventions take categorical slots 1-3 unmodified and in order. Bars are also
#: value-labelled and written out as CSV, so identity never rests on colour.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID, C_BASE = "#e4e3df", "#b9b8b3"

BASE_W, CW_W = 1.0, 1.5
#: (key, label, scoring, w_off, colour)
CELLS = [
    ("neither", "Neither",            "prevalence", BASE_W, C_BASE),
    ("cw",      "Class weighting",    "prevalence", CW_W,   C_BLUE),
    ("pp",      "Post-processing",    "prev_pp",    BASE_W, C_ORANGE),
    ("both",    "Both",               "prev_pp",    CW_W,   C_AQUA),
]
APPS = ["Fridge", "Microwave", "Dishwasher", "WashingMachine"]
SHORT = {"Fridge": "Fridge", "Microwave": "Microwave",
         "Dishwasher": "Dishwasher", "WashingMachine": "Washing m."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("figures/fig_classweight.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/fig_interventions"))
    a = p.parse_args()
    if not a.csv.exists():
        raise SystemExit(f"{a.csv} not found -- run scripts.make_classweight_figure first")

    tab = {}
    with open(a.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            tab[(r["scoring"], r["appliance"], float(r["w_off"]))] = float(r["f1"])

    vals, per_app = {}, collections.defaultdict(dict)
    for key, _, scoring, w, _c in CELLS:
        got = []
        for app in APPS:
            k = (scoring, app, w)
            if k not in tab:
                raise SystemExit(f"missing {k} in {a.csv} -- regenerate it")
            per_app[key][app] = tab[k]
            got.append(tab[k])
        vals[key] = float(np.mean(got))

    d_cw, d_pp, d_both = (vals["cw"] - vals["neither"],
                          vals["pp"] - vals["neither"],
                          vals["both"] - vals["neither"])
    inter = d_both - d_cw - d_pp
    for key, label, _, _, _ in CELLS:
        print(f"{label:<18}{vals[key]:6.2f}")
    print(f"\nclass weighting  {d_cw:+.2f}\npost-processing  {d_pp:+.2f}"
          f"\nboth             {d_both:+.2f}\ninteraction      {inter:+.2f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.45]),
                                   constrained_layout=True, facecolor="white")

    # -- Panel A: the mean, one bar per cell ---------------------------------------
    xs = np.arange(len(CELLS))
    for i, (key, label, _, _, colour) in enumerate(CELLS):
        axA.bar(i, vals[key], width=0.68, color=colour, zorder=3,
                edgecolor="white", linewidth=1.6)
        axA.text(i, vals[key] + 0.7, f"{vals[key]:.1f}", ha="center", va="bottom",
                 fontsize=12.5, color=INK)
        if key != "neither":
            axA.text(i, vals[key] - 2.6, f"{vals[key]-vals['neither']:+.1f}",
                     ha="center", va="top", fontsize=11.5, color="white")
    axA.set_xticks(xs)
    axA.set_xticklabels([c[1] for c in CELLS], rotation=18, ha="right")
    axA.set_ylabel("F1 (%)")
    axA.set_ylim(0, 78)
    axA.set_title(f"Mean over {len(APPS)} appliances", color=INK, pad=10)

    # -- Panel B: per appliance ----------------------------------------------------
    w = 0.20
    base = np.arange(len(APPS))
    for i, (key, label, _, _, colour) in enumerate(CELLS):
        axB.bar(base + (i - 1.5) * w, [per_app[key][x] for x in APPS], width=w * 0.92,
                color=colour, label=label, zorder=3, edgecolor="white", linewidth=1.0)
    axB.set_xticks(base)
    axB.set_xticklabels([SHORT[x] for x in APPS])
    axB.set_ylabel("F1 (%)")
    axB.set_ylim(0, 100)
    axB.set_title("Per appliance", color=INK, pad=10)
    axB.legend(frameon=False, loc="upper center", ncol=4, labelcolor=INK_2,
               columnspacing=1.1, handlelength=1.4)

    for ax in (axA, axB):
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_edgecolor(INK_MUTED)
        ax.tick_params(colors=INK_2)

    fig.suptitle("Post-processing is worth an order of magnitude more than "
                 "class weighting", fontsize=16, color=INK)
    fig.text(0.5, -0.02,
             "Threshold rule held fixed at prevalence-matched-on-validation in all four "
             f"cells. Class weighting uses w_off = {CW_W:g}, its best measured setting, so "
             f"that bar is an upper bound. Interaction term {inter:+.1f} F1.",
             ha="center", va="top", fontsize=9.5, color=INK_2)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cell", "scoring", "w_off", "appliance", "f1"])
        for key, label, scoring, wo, _c in CELLS:
            for app in APPS:
                wr.writerow([label, scoring, wo, app, f"{per_app[key][app]:.2f}"])
            wr.writerow([label, scoring, wo, "MEAN", f"{vals[key]:.2f}"])
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=240 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print(f"\nwrote {a.out}.png / .pdf / .csv  (source: {a.csv})")


if __name__ == "__main__":
    main()
