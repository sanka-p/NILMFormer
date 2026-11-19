"""Real REDD waveforms: Microwave vs WashingMachine, in watts and after stationarising.

Motivation: the `stationarize` arms instance-normalise each window, deleting absolute
watts and keeping only shape. That is worth +14.7 F1 on Microwave (on top of activity
cropping -- the best Microwave score in the study) and costs WashingMachine. The tempting
explanation, "Microwave is amplitude-invariant and WashingMachine is amplitude-dependent",
is NOT what the data shows, and this figure is the refutation.

What it actually shows:
  left   four real segments per appliance, in watts, stacked -- Microwave is a short
         burst at a tightly repeated rated power; WashingMachine is a long heater duty
         cycle at ~3x the power.
  middle the same four z-scored per segment, i.e. exactly what a stationarised window
         presents to the TCN. Microwave keeps a clean two-level burst; WashingMachine
         becomes a dense square wave whose level no longer says how big the load is.
  right  ON-sample power pooled over every segment, which is the amplitude evidence the
         overlay cannot show. CV is over ALL segments, not only the four drawn.

NOTE on the pools: segments from results/tcn_aug_cache/ are multi-cycle stretches, not
isolated single activations, so a segment's max is the appliance's rated power rather
than one activation's amplitude. The traces are therefore cropped to CROP samples so the
individual duty cycles are legible.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kl_spectrum import CACHE, load_segments  # noqa: E402

# appliance, cache cell, colour, ON threshold (W, configs/datasets appliance_param)
# appliance, cache cell, colour, ON threshold (W), crop length (samples).
# Microwave segments are short (median 47 samples), so it gets a shorter crop; using one
# global crop silently dropped every Microwave segment.
CELLS = [("Microwave", "REDD_h3-5_10s_w128", "#1E4FD8", 200, 60),
         ("WashingMachine", "REDD_h3-5-6_10s_w768", "#C2410C", 20, 150)]
INK, MUTED = "#16223C", "#5A6477"
NSHOW = 4

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#C9CCD4", "xtick.color": MUTED,
                     "ytick.color": MUTED})


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 0.85],
                                          "hspace": 0.42, "wspace": 0.26})

    for r, (lbl, key, col, floor, CROP) in enumerate(CELLS):
        allsegs = [np.asarray(s, float)
                   for s in load_segments(CACHE / f"{key}_s0.pkl")[lbl]]
        # Statistics are over EVERY segment; the crop filter applies only to what is
        # drawn, so the quoted CVs do not silently describe a long-segment subset.
        on = np.concatenate([s[s > floor] for s in allsegs if (s > floor).any()])
        peaks = np.array([s.max() for s in allsegs])
        drawable = [s for s in allsegs if len(s) >= CROP]
        rng = np.random.default_rng(1)
        pick = [drawable[i] for i in
                rng.choice(len(drawable), min(NSHOW, len(drawable)), replace=False)]

        axA, axB, axC = axes[r]
        for i, s in enumerate(pick):
            t = np.arange(CROP) / 6.0
            axA.plot(t, s[:CROP] + i * 0, color=col, lw=1.1, alpha=0.85)
            z = (s - s.mean()) / (s.std() + 1e-9)
            axB.plot(t, z[:CROP] - i * 4.5, color=col, lw=1.1, alpha=0.85)

        axA.set_title(f"{lbl} — 4 real segments, watts", fontsize=9.5,
                      color=INK, loc="left", pad=6)
        axA.set_ylabel("watts", fontsize=8.5)
        axB.set_title(f"{lbl} — z-scored (what stationarise shows the TCN)",
                      fontsize=9.5, color=INK, loc="left", pad=6)
        axB.set_ylabel("$z$, traces offset", fontsize=8.5)
        axB.set_yticks([])
        for ax in (axA, axB):
            ax.set_xlabel("minutes from segment start", fontsize=8.5)
            ax.set_xlim(0, CROP / 6.0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

        axC.hist(on, bins=45, color=col, alpha=0.8, edgecolor="none")
        axC.axvline(np.median(on), color=INK, lw=1.4, ls="--")
        axC.set_title("ON-sample power, all segments", fontsize=9.5, color=INK,
                      loc="left", pad=6)
        axC.set_xlabel("watts", fontsize=8.5)
        axC.set_ylabel("samples", fontsize=8.5)
        # annotate on the side away from where the ON-power mass sits
        hi = np.median(on) > 0.5 * (on.min() + on.max())
        ax_x, ax_ha = (0.03, "left") if hi else (0.97, "right")
        axC.text(ax_x, 0.93,
                 f"median {np.median(on):,.0f} W\n"
                 f"CV of ON power {on.std()/on.mean():.2f}\n"
                 f"p10/p90 spread {np.percentile(on,90)/max(np.percentile(on,10),1):.1f}x",
                 transform=axC.transAxes, ha=ax_ha, va="top", fontsize=8,
                 color=col, linespacing=1.5)
        for sp in ("top", "right"):
            axC.spines[sp].set_visible(False)

    fig.suptitle("ON-power is bimodal in both loads \u2014 the washer-dryer's two modes are "
                 "far further apart, and stationarising removes what separates them",
                 fontsize=10.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.012,
             "REDD pools (results/tcn_aug_cache/), Microwave ws128, WashingMachine ws768 "
             "· 10 s sampling · segments are multi-cycle stretches, cropped to 60 "
             "(Microwave) / 150 (WashingMachine) samples for legibility · statistics "
             "over ALL segments, not only the four drawn",
             fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.915, bottom=0.095)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_activation_variability.{ext}", dpi=200,
                    facecolor="white", bbox_inches="tight")
    print("wrote figures/fig_activation_variability.png / .pdf")


if __name__ == "__main__":
    main()
