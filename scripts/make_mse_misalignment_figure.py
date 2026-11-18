#################################################################################################################
#
# @description : Where a NILM regressor's squared error actually sits.
#
# This figure was built to show that MSE wastes its budget on amplitude error that
# detection does not care about. IT REFUTED THAT. Measured on REDD/TCN_KL_scratch,
# ~80% of the test squared error sits on FALSE POSITIVES -- the one thing F1 punishes
# hardest -- and only 15-33% on samples already classified correctly. MSE and F1 are
# largely ALIGNED; the dominant failure is confident, high-amplitude firing at the
# wrong time (FP RMSE 158 W on a fridge whose mean ON power is 192 W; 769 W on the
# microwave), not a cheap low-amplitude haze.
#
# The residual misalignment is narrower than first argued: MSE weights a false
# positive by amplitude^2 while F1 counts every one equally, so borderline firings
# stay under-penalised.
#
# Row 1  true vs predicted power, with the ON threshold drawn.
# Row 2  the per-sample squared error, filled.
# Shaded bands mark the samples F1 actually disagrees on. The message the eye gets for
# free is the measured one: the error mountains sit INSIDE the bands.
#
# Two appliances, because the story inverts with prevalence: Fridge (~23% ON) and
# Microwave (~1% ON).
#
# CAVEAT drawn on the figure: the trainer zeroes predictions below the appliance
# threshold before saving them (trainer.py:278), so sub-threshold model output is not
# recoverable here. That erases true-OFF error, which UNDERSTATES the already-correct
# share and OVERSTATES the false-positive share. The quoted correct-share is therefore
# a floor, not a point estimate; settling it needs export_raw_predictions.py.
#
# Numbers are recomputed from the stored predictions, never transcribed.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_mse_misalignment_figure \
#         --root result-redd-fix --arm TCN_KL_scratch
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.helpers.postprocess import runs                                  # noqa: E402
from scripts.calibrate_threshold import DEFAULT_GRID, build_config        # noqa: E402
from scripts.decompose_mse import truth_power_and_state, _find            # noqa: E402

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

#: Categorical slots 1-2 of the reference palette, used unmodified and in order; they
#: carry the two data series. The misclassification band is an ANNOTATION layer, not a
#: series, so it takes slot 4 at low alpha and is named in the legend. Squared error is
#: a single series and wears muted ink. Every series is also direct-labelled and the
#: plotted window is written out as CSV, so identity is never colour alone.
C_BLUE, C_ORANGE, C_YELLOW = "#2a78d6", "#eb6834", "#eda100"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8982"
FILL_ERR = "#c9c8c3"

#: Samples drawn per appliance. Fridge cycles slowly; the microwave fires in bursts, so
#: a shorter span keeps the activation from vanishing into a flat line.
SPAN = {"Fridge": 1800, "Microwave": 900, "Dishwasher": 2400, "WashingMachine": 2400}
SR_SECONDS = 10.0


def _pick_window(se, wrong, span):
    """Start index of the most illustrative window.

    Wanted: a stretch carrying a lot of squared error that ALSO contains at least one
    disagreement, so both halves of the comparison are on screen. Windows with no
    misclassified sample are excluded -- they would show the claim without its
    counter-example and overstate it.
    """
    step = max(span // 4, 1)
    best, best_score = None, -1.0
    for s in range(0, max(se.size - span, 1), step):
        sl = slice(s, s + span)
        if not wrong[sl].any():
            continue
        score = float(se[sl].sum())
        if score > best_score:
            best, best_score = s, score
    return 0 if best is None else best


def _bands(ax, t, wrong, label=None):
    """Shade every contiguous run of misclassified samples."""
    first = True
    for a, b in runs(wrong.astype(np.int8), value=1):
        ax.axvspan(t[a], t[min(b, t.size - 1)], color=C_YELLOW, alpha=0.20,
                   lw=0, zorder=0, label=(label if first else None))
        first = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arm", default="TCN_KL_scratch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apps", nargs="+", default=["Fridge", "Microwave"])
    p.add_argument("--start", type=int, default=None,
                   help="force the window start index (default: auto-pick)")
    p.add_argument("--out", default="figures/fig_mse_misalignment")
    a = p.parse_args()

    cols = []
    for app in a.apps:
        ws = DEFAULT_GRID[app]
        cfg = build_config("REDD", app, ws)
        y_pow, y_st, floor = truth_power_and_state(cfg)
        f = _find(a.root, app, ws, a.arm, a.seed)
        if f is None:
            raise SystemExit(f"no finished run for {a.arm} / {app} / ws{ws} / seed{a.seed}")
        yh = np.asarray(torch.load(f, map_location="cpu",
                                   weights_only=False)["test_metrics_yhat"]
                        ).ravel().astype(np.float64)
        if yh.size != y_pow.size:
            raise SystemExit(f"{app}: size mismatch {yh.size} vs {y_pow.size}")

        se = (yh - y_pow) ** 2
        pred_on = (yh > floor).astype(np.int64)
        wrong = pred_on != y_st
        # The headline number is GLOBAL, over the whole test split -- the drawn window
        # is only an illustration of it and must not be mistaken for the evidence.
        share = 100.0 * se[~wrong].sum() / se.sum()
        fp = (y_st == 0) & (pred_on == 1)
        fp_share = 100.0 * se[fp].sum() / se.sum()
        fp_rmse = float(np.sqrt(se[fp].mean())) if fp.any() else 0.0
        fp_frac = 100.0 * fp.mean()
        span = min(SPAN.get(app, 1800), se.size)
        s0 = a.start if a.start is not None else _pick_window(se, wrong, span)
        cols.append(dict(app=app, floor=floor, share=share, fp_share=fp_share,
                         fp_rmse=fp_rmse, fp_frac=fp_frac, s0=s0, span=span,
                         y_pow=y_pow, yh=yh, se=se, wrong=wrong, y_st=y_st,
                         on_rate=100.0 * y_st.mean()))
        print(f"{app:<16} {100.0*y_st.mean():>5.2f}% ON  floor {floor:>4.0f} W   "
              f"FP {fp_share:.1f}% of squared error (n={fp_frac:.2f}% of samples, "
              f"RMSE {fp_rmse:.0f} W); already-correct {share:.1f}%   "
              f"window [{s0}:{s0+span}]")

    fig, axes = plt.subplots(2, len(cols), figsize=(7.4 * len(cols), 7.4),
                             gridspec_kw=dict(height_ratios=[3.0, 2.0], hspace=0.22,
                                              wspace=0.20))
    axes = np.atleast_2d(axes.reshape(2, -1))

    for j, c in enumerate(cols):
        sl = slice(c["s0"], c["s0"] + c["span"])
        t = np.arange(c["span"]) * SR_SECONDS / 3600.0
        gt, pr, se, wrong = c["y_pow"][sl], c["yh"][sl], c["se"][sl], c["wrong"][sl]

        # ---- Row 1: the traces -------------------------------------------------
        ax = axes[0, j]
        _bands(ax, t, wrong, label="model and ground truth disagree")
        ax.plot(t, gt, color=C_BLUE, lw=1.9, label="ground truth", zorder=3)
        ax.plot(t, pr, color=C_ORANGE, lw=1.9, label="prediction", zorder=4)
        ax.axhline(c["floor"], color=INK_2, lw=1.2, ls=(0, (5, 4)), zorder=2,
                   label=f"ON threshold, {c['floor']:.0f} W")
        ax.set_title(f"{c['app']}  ({c['on_rate']:.1f}% of samples ON)",
                     color=INK, pad=10)
        ax.set_ylabel("power (W)", color=INK)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(colors=INK_2)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                  edgecolor=INK_3, labelcolor=INK)

        # ---- Row 2: where the squared error actually is ------------------------
        ax = axes[1, j]
        _bands(ax, t, wrong)
        ax.fill_between(t, 0.0, se, color=FILL_ERR, lw=0.0, zorder=2)
        ax.plot(t, se, color=INK_3, lw=0.9, zorder=3, label="squared error")
        ax.set_ylabel("squared error (W$^2$)", color=INK)
        ax.set_xlabel("time (hours)", color=INK)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0.0)
        ax.tick_params(colors=INK_2)
        ax.legend(handles=[Patch(facecolor=FILL_ERR, edgecolor=INK_3,
                                 label="squared error"),
                           Patch(facecolor=C_YELLOW, alpha=0.20,
                                 label="F1 disagrees here")],
                  loc="upper right", frameon=True, framealpha=0.95,
                  edgecolor=INK_3, labelcolor=INK)
        ax.text(0.015, 0.94,
                f"false positives carry {c['fp_share']:.0f}% of the squared error\n"
                f"from {c['fp_frac']:.1f}% of samples (RMSE {c['fp_rmse']:.0f} W)\n"
                f"only {c['share']:.0f}% lies where F1 is already satisfied",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                color=INK, bbox=dict(facecolor="white", edgecolor=INK_3,
                                     boxstyle="round,pad=0.38", alpha=0.95))

    fig.suptitle("Squared error concentrates on false positives, "
                 "where F1 already objects",
                 fontsize=16.5, color=INK, y=0.975)
    fig.text(0.5, 0.005,
             "Shares are over the full test split; the drawn windows are illustrations of it. "
             "Predictions below the ON threshold are zeroed before storage (trainer.py:278), "
             "which erases true-OFF error: the already-correct share is a FLOOR and the "
             "false-positive share a CEILING.",
             ha="center", va="bottom", fontsize=9.5, color=INK_2)
    fig.subplots_adjust(top=0.90, bottom=0.095, left=0.075, right=0.985)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=240 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    # Table view: the plotted windows, so the figure can be read without colour.
    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["appliance", "on_rate_pct", "floor_W", "share_on_correct_pct",
                    "sample", "hours", "true_W", "pred_W", "squared_error",
                    "true_state", "pred_state"])
        for c in cols:
            sl = slice(c["s0"], c["s0"] + c["span"])
            pred_on = (c["yh"][sl] > c["floor"]).astype(int)
            for i in range(c["span"]):
                w.writerow([c["app"], f"{c['on_rate']:.2f}", f"{c['floor']:.0f}",
                            f"{c['share']:.1f}", c["s0"] + i,
                            f"{i * SR_SECONDS / 3600.0:.4f}",
                            f"{c['y_pow'][sl][i]:.1f}", f"{c['yh'][sl][i]:.1f}",
                            f"{c['se'][sl][i]:.1f}", int(c["y_st"][sl][i]),
                            pred_on[i]])
    print(f"\nwrote {a.out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
