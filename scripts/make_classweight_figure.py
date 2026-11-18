#################################################################################################################
#
# @description : What increasing the OFF-class weight actually does to detection F1.
#
# The weighted-MSE arms use `[w_off, w_on]` with w_off > w_on -- the OPPOSITE of
# inverse-frequency weighting -- because the failure being treated is over-firing
# (predicted-ON / true-ON > 1), not missed detections.
#
# The figure's point is that the answer DEPENDS ON THE SCORING RULE, so plotting one
# curve would misrepresent it. Panel A draws the same four models under three decision
# rules; panel B shows why the mean in panel A collapses at w_off = 10.
#
# Numbers are recomputed here from the stored predictions rather than transcribed, so
# the figure cannot drift from the tables. Prevalence-matched thresholds are taken on
# VALIDATION (label-free -- see result-redd-tod/README.md), never on test.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_classweight_figure --root result-redd-tod
#
#################################################################################################################

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for _noisy in ("fontTools", "fontTools.subset", "matplotlib", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.helpers.postprocess import durations_for, postprocess_state      # noqa: E402
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,      # noqa: E402
                                         ground_truth_states)
from scripts.score_postprocess import (SEEDS, _pick_threshold,            # noqa: E402
                                       _timestamp_metrics)

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

#: Arms in increasing OFF-class weight. w_off = 1.0 is the unweighted reference.
ARMS = [("TCN_KL_scratch", 1.0), ("TCN_KL_cw1_5", 1.5),
        ("TCN_KL_cw3", 3.0), ("TCN_KL_cw10", 10.0)]

#: Categorical slots 1-4 of the reference palette, used unmodified and in order.
#: Lines use the ADJACENT pairlist, which this order passes in both modes. Slots 3
#: and 4 sit below 3:1 on a light surface, so the relief rule applies: every series
#: carries a direct label and a distinct marker, and the plotted values are also
#: written out as CSV. Identity is never colour alone.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID = "#e4e3df"

SCORINGS = [
    ("stock",      "Stock threshold",          C_BLUE,   "o"),
    ("prevalence", "Prevalence-matched",       C_ORANGE, "s"),
    ("prev_pp",    "Prevalence + post-proc.",  C_AQUA,   "^"),
]
APP_STYLE = [("WashingMachine", "Washing m.", C_BLUE, "o"),
             ("Dishwasher",     "Dishwasher", C_ORANGE, "s"),
             ("Microwave",      "Microwave",  C_AQUA,  "^"),
             ("Fridge",         "Fridge",     C_YELLOW, "D")]


def collect(root):
    """{scoring: {appliance: {w_off: mean F1}}} recomputed from the stored runs."""
    out = {k: {} for k, _, _, _ in SCORINGS}
    for app, ws in DEFAULT_GRID.items():
        cfg = build_config("REDD", app, ws)
        y_valid, y_test, floor = ground_truth_states(cfg)
        dur, rate = durations_for(app), float(y_valid.mean())
        for k in out:
            out[k][app] = {}
        for arm, w_off in ARMS:
            acc = {k: [] for k, _, _, _ in SCORINGS}
            for seed in SEEDS:
                f = None
                for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
                    p = os.path.join(root, f"REDD_{key}_10s", str(ws), f"{arm}_{seed}.pt")
                    if os.path.exists(p + ".done"):
                        f = p
                        break
                if f is None:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                yh_v = np.asarray(log["valid_metrics_yhat"], dtype=np.float64).ravel()
                yh_t = np.asarray(log["test_metrics_yhat"], dtype=np.float64).ravel()
                if yh_v.size != y_valid.size or yh_t.size != y_test.size:
                    continue
                # Stock detector: the bare comparison the trainer itself uses.
                acc["stock"].append(
                    _timestamp_metrics(y_test, (yh_t > floor).astype(np.int8))["F1"])
                t1 = _pick_threshold(yh_v, rate, floor, None)
                acc["prevalence"].append(
                    _timestamp_metrics(y_test, (yh_t > t1).astype(np.int8))["F1"])
                t2 = _pick_threshold(yh_v, rate, floor, dur)
                s2 = postprocess_state((yh_t > t2).astype(np.int8), **dur)
                acc["prev_pp"].append(_timestamp_metrics(y_test, s2)["F1"])
            for k in out:
                if acc[k]:
                    out[k][app][w_off] = float(np.mean(acc[k]))
        print(f"  {app}: done", flush=True)
    return out


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
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--out", type=Path, default=Path("figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print("collecting (recomputed from stored predictions)...", flush=True)
    data = collect(a.root)
    xs = [w for _, w in ARMS]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True,
                                   facecolor="white")

    # -- Panel A: the mean, under three decision rules ------------------------------
    for key, label, colour, marker in SCORINGS:
        ys = [float(np.mean([data[key][app][w] for app in DEFAULT_GRID
                             if w in data[key][app]])) for w in xs]
        axA.plot(xs, ys, color=colour, marker=marker, markersize=8, linewidth=2,
                 markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        axA.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=(10, 0), va="center", fontsize=11, color=INK_2)
    axA.set_title("Mean F1 over four appliances", color=INK, pad=10)
    axA.set_ylabel("F1 (%)")
    axA.set_xlim(0.88, 26)
    _decorate(axA, xs)
    axA.legend(frameon=False, loc="lower left", labelcolor=INK_2)

    # -- Panel B: why the mean collapses --------------------------------------------
    for app, label, colour, marker in APP_STYLE:
        ys = [data["prev_pp"][app][w] for w in xs if w in data["prev_pp"][app]]
        axB.plot(xs[:len(ys)], ys, color=colour, marker=marker, markersize=8,
                 linewidth=2, markeredgecolor="white", markeredgewidth=1.2,
                 label=label, zorder=3)
        axB.annotate(label, (xs[len(ys) - 1], ys[-1]), textcoords="offset points",
                     xytext=(10, 0), va="center", fontsize=11, color=INK_2)
    axB.set_title("Per appliance, prevalence + post-processing", color=INK, pad=10)
    axB.set_ylabel("F1 (%)")
    axB.set_xlim(0.88, 26)
    axB.set_ylim(-4, 100)
    _decorate(axB, xs)
    axB.legend(frameon=False, loc="lower left", labelcolor=INK_2)
    mw = data["prev_pp"]["Microwave"]
    if 10.0 in mw:
        axB.annotate("silent detector:\nsignal falls below\nthe clipping floor",
                     (10.0, mw[10.0]), textcoords="offset points", xytext=(-14, 46),
                     ha="right", fontsize=10.5, color=INK_2,
                     arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.0))

    rows = []
    for key, label, _, _ in SCORINGS:
        for app in DEFAULT_GRID:
            for w in xs:
                if w in data[key][app]:
                    rows.append(dict(scoring=key, appliance=app, w_off=w,
                                     f1=round(data[key][app][w], 2)))
    with open(a.out / "fig_classweight.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["scoring", "appliance", "w_off", "f1"])
        wr.writeheader(); wr.writerows(rows)

    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"fig_classweight.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {a.out}/fig_classweight.png|pdf and fig_classweight.csv "
          f"({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
