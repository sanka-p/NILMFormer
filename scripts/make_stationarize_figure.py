"""Schematic of NILMFormer-style stationarisation, as implemented in TCN_NILM.forward.

`stationarize` separates a window's SHAPE from its MAGNITUDE: the window is instance-
normalised, the TCN sees only the shape, and the two statistics are handed back as data
rather than as scale. The subtlety worth drawing is the ASYMMETRIC output path -- the
window's scale is restored onto the power channel only, never onto the ON/OFF logit,
because a detector's confidence must not depend on how loud the window happened to be.

Panels:
  A  the pipeline, with tensor shapes at each hop (kl_order 10, state_head on)
  B  two windows of identical shape and different magnitude, before normalisation
  C  the same two after normalisation -- the TCN cannot tell them apart
  D  the measured consequence: de-stationarisation rescales power 9.57x between a loud
     and a quiet window, and leaves the logit essentially untouched (1.19x)

Source: src/baselines/nilm/tcn.py:246-298. The 9.57 / 1.19 figures are from the
multi-task head verification run, check 3.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SHAPE = "#1E4FD8"     # the normalised signal path
MAG = "#C2410C"       # the statistics path
INK = "#16223C"
MUTED = "#5A6477"
BG = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#C9CCD4", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def box(ax, x, y, w, h, text, color, lw=1.6, fs=8.5, fc=None, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=color, facecolor=fc or "#FFFFFF",
        linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color, label=None, lw=1.6, rad=0.0, dy=0.03):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=7.4, color=color, zorder=4)


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    yS, yM, h = 0.60, 0.10, 0.25          # shape lane, magnitude lane, box height
    cS, cM = yS + h / 2, yM + h / 2

    box(ax, 0.010, yS, 0.115, h, "window $x$\n$(B,1,T)$\nscaled aggregate", INK, lw=1.4)
    arrow(ax, 0.127, cS, 0.163, cS, SHAPE)
    box(ax, 0.163, yS, 0.135, h,
        r"$x' = \dfrac{x-\mu}{\sigma}$" + "\nSHAPE only", SHAPE)
    arrow(ax, 0.300, cS, 0.336, cS, SHAPE)
    box(ax, 0.336, yS, 0.110, h, "KL filter\n$(B,11,T)$", SHAPE)
    arrow(ax, 0.448, cS, 0.484, cS, SHAPE)
    box(ax, 0.484, yS, 0.120, h, "concat\n$(B,13,T)$", INK, lw=1.4)
    arrow(ax, 0.606, cS, 0.642, cS, INK)
    box(ax, 0.642, yS, 0.085, h, "TCN", INK, lw=1.4)

    # magnitude lane: branch down, travel right, rejoin at concat
    arrow(ax, 0.0675, yS, 0.0675, yM + h, MAG)
    box(ax, 0.010, yM, 0.115, h,
        r"$\mu=\mathrm{mean}(x)$" + "\n" + r"$\sigma=\mathrm{std}(x)$"
        + "\n" + r"$\mathbf{.detach()}$", MAG, lw=1.6)
    arrow(ax, 0.127, cM, 0.544, cM, MAG,
          label=r"$\mu,\sigma$ broadcast over $T$ as 2 extra channels", dy=0.03)
    arrow(ax, 0.544, cM + 0.02, 0.544, yS, MAG)

    # output split
    ax.plot([0.740, 0.740], [cS - 0.170, cS + 0.170], color=INK, lw=1.3, zorder=2)
    ax.plot([0.727, 0.740], [cS, cS], color=INK, lw=1.3, zorder=2)
    arrow(ax, 0.740, cS + 0.170, 0.775, cS + 0.170, MAG)
    arrow(ax, 0.740, cS - 0.170, 0.775, cS - 0.170, SHAPE)

    box(ax, 0.775, cS + 0.105, 0.135, 0.130,
        r"power $\times\,\sigma' + \mu'$", MAG, lw=1.8, fs=8)
    box(ax, 0.775, cS - 0.235, 0.135, 0.130, "logit  UNCHANGED", SHAPE, lw=1.8, fs=8)

    ax.text(0.8425, cS + 0.255,
            r"$(\mu',\sigma')=\mathrm{Linear}(\mu,\sigma)$, init $[[0,0],[0,1]]$:"
            "\nstep 0 restores scale, injects no mean",
            ha="center", va="bottom", fontsize=7.0, color=MAG, linespacing=1.5)
    ax.text(0.8425, cS - 0.255,
            "never rescaled \u2014 a detector's confidence\n"
            "must not depend on how loud the window was",
            ha="center", va="top", fontsize=7.0, color=SHAPE, linespacing=1.5)

    ax.text(0.0, 0.99, "A", fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0.030, 0.985,
            "Stationarisation splits the window into shape and magnitude, then restores "
            "magnitude onto the power channel only",
            fontsize=9.5, color=INK, va="top")


def _demo():
    t = np.arange(360)
    base = np.zeros_like(t, dtype=float)
    base[60:150] = 1.0
    base[220:300] = 1.0
    base += 0.06 * np.sin(t / 7.0)
    return t, 140 * base + 60, 1320 * base + 410


def panel_bc(axB, axC):
    t, quiet, loud = _demo()
    specs = (
        (axB, quiet, loud, "B   Before: same shape, different magnitude", "watts"),
        (axC, (quiet - quiet.mean()) / quiet.std(),
         (loud - loud.mean()) / loud.std(),
         r"C   After $(x-\mu)/\sigma$: the two coincide exactly", "$z$"),
    )
    for ax, a, b, title, ylab in specs:
        ax.plot(t, b, color=MAG, lw=1.6, label="loud window")
        ax.plot(t, a, color=SHAPE, lw=1.6, ls="--", label="quiet window")
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=6)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlabel("sample", fontsize=8.5)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.6, loc="upper right", ncol=2,
                  handlelength=1.6, columnspacing=1.0)


def panel_d(ax):
    vals, cols = [9.57, 1.19], [MAG, SHAPE]
    ax.bar(["power\nchannel", "ON/OFF\nlogit"], vals, color=cols, width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.2f}\u00d7", ha="center", fontsize=9.5,
                color=cols[i], fontweight="bold")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.text(0.5, 2.1, "1.0 = no rescaling", fontsize=7.2, color=MUTED, ha="center")
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("output std ratio,\nloud window / quiet window", fontsize=8.5)
    ax.set_title("D   Measured: only power is rescaled", fontsize=9, color=INK,
                 loc="left", pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    fig = plt.figure(figsize=(13.0, 6.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92],
                          hspace=0.26, wspace=0.28,
                          left=0.055, right=0.975, top=0.96, bottom=0.085)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_bc(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    panel_d(fig.add_subplot(gs[1, 2]))

    fig.text(0.055, 0.015,
             "src/baselines/nilm/tcn.py:246-298  ·  shapes shown for kl_order 10 with "
             "state_head enabled  ·  9.57× / 1.19× measured in the multi-task head check",
             fontsize=7.2, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_stationarize.{ext}", dpi=200,
                    facecolor=BG, bbox_inches="tight")
    print("wrote figures/fig_stationarize.png / .pdf")


if __name__ == "__main__":
    main()
