#################################################################################################################
#
# @description : Two figures arguing for the sampler and the multiplexer, from measured data.
#
# These answer a different question from results/fig_synth_generator.pdf. That figure shows
# HOW the generator works; these show WHY each piece is there, by measuring what happens
# without it. Both run off the same activation pool the model trains from.
#
#   fig_sampler_contrast      OffCutoffSampler vs naive cropping. A naive crop picks two
#                             uniform endpoints, so it cuts through an activation most of
#                             the time and the splice fabricates a step no meter produces.
#                             Measured: 83% of dishwasher crops and 91% of washing-machine
#                             crops are cut mid-activation, averaging ~1 kW of artificial
#                             discontinuity, against 2.4% / 1.6% for the sampler.
#
#   fig_multiplexer_diversity DeviceMultiplexer vs reusing one segment. The multiplexer
#                             draws from the whole pool, so the generated activations carry
#                             the measured ensemble's amplitude spread; without it the
#                             stream repeats a single amplitude (washing machine: pool
#                             sd 1006 W, multiplexer 1005 W, one segment 0 W).
#
# Usage:
#     PYTHONPATH=. ./env/bin/python -m scripts.make_sampler_figure
#     PYTHONPATH=. ./env/bin/python -m scripts.make_sampler_figure --app dishwasher
#
#################################################################################################################

import argparse
import itertools
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_synth_figure import (APPS, COLOR, CROP_COLOR, NICE, OFF_SHADE,
                                       check_mirror)
from src.helpers.preprocessing import UKDALE_DataBuilder
from src.helpers.tcn_ablation import (
    DeviceMultiplexer,
    OffCutoffSampler,
    _build_multiplexer,
    _calibrate_active_fraction,
    generate_stream,
)

CACHE = Path("results/tcn_aug_cache/UKDALE_h1-3-4-5_10s_w128_s0.pkl")
NAIVE_COLOR = "#CC79A7"

#: Appliances whose crops the cut-rate table reports, in the figure's order.
TABLE_APPS = ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"]


def naive_interval(segment, rng, min_len=8):
    """Two uniform endpoints -- cropping with no notion of where the activation is.

    This is the baseline the boundary-aware sampler replaces, not a strawman: it is what
    "take a random slice of a trace" means if the slice is not constrained to start and
    end in an OFF region.
    """
    for _ in range(200):
        a, b = sorted(rng.integers(0, segment.size, size=2))
        if b - a >= min_len:
            return int(a), int(b)
    return 0, segment.size


def cut_rate(segments, threshold, mux, rng, n=3000):
    """How often a crop's edge lands inside an activation, and how big the splice is.

    An edge above the ON threshold means the crop begins or ends mid-operation; dropped
    into a stream next to a zero gap it creates a step of that size, which is the
    artefact the sampler exists to avoid.
    """
    naive_cut = aware_cut = 0
    naive_jump, aware_jump = [], []
    for _ in range(n):
        s = segments[rng.integers(len(segments))]
        if s.size < 8:
            continue
        a, b = naive_interval(s, rng)
        c = s[a:b]
        if c.size == 0:
            continue
        naive_cut += int(c[0] >= threshold or c[-1] >= threshold)
        naive_jump.append(max(abs(float(c[0])), abs(float(c[-1]))))
    for _ in range(n):
        c = mux()
        if c.size == 0:
            continue
        aware_cut += int(c[0] >= threshold or c[-1] >= threshold)
        aware_jump.append(max(abs(float(c[0])), abs(float(c[-1]))))
    return {
        "naive_cut": 100 * naive_cut / max(1, len(naive_jump)),
        "aware_cut": 100 * aware_cut / max(1, len(aware_jump)),
        "naive_jump": float(np.mean(naive_jump)),
        "aware_jump": float(np.mean(aware_jump)),
    }


def naive_stream(segments, length, active_fraction, rng):
    """generate_stream's layout, fed by naive crops instead of boundary-aware ones.

    Only the crop source differs, so any visible difference between this and the real
    stream is attributable to the sampler and to nothing else.
    """
    target = int(length * active_fraction)
    target = max(1, min(target, length - 1))
    cur, samples = 0, []
    while cur < target:
        s = segments[rng.integers(len(segments))]
        if s.size < 8:
            continue
        a, b = naive_interval(s, rng)
        c = s[a:b]
        if c.size == 0:
            continue
        cur += c.size
        if cur > target:
            c = c[: -(cur - target)]
        samples.append(c)
    zeros = np.zeros(length - target)
    pts = np.sort(rng.integers(0, zeros.size, size=max(0, len(samples) - 1)))
    sectors = np.split(zeros, pts)
    spans, cursor = [], 0
    for z, c in itertools.zip_longest(sectors, samples):
        cursor += 0 if z is None else z.size
        if c is not None:
            spans.append((cursor, cursor + c.size))
            cursor += c.size
    sig = np.concatenate(list(itertools.chain.from_iterable(zip(sectors, samples))))[:length]
    return sig, [(a, b) for a, b in spans if a < length]


def splice_jumps(stream, spans, threshold):
    """Indices where a crop edge meets a gap with a step bigger than half the threshold."""
    out = []
    for a, b in spans:
        for i in (a, min(b, stream.size - 1)):
            if 0 < i < stream.size and abs(stream[i] - stream[i - 1]) > threshold / 2:
                out.append(i)
    return out


def draw_segment_crop(ax, segment, a, b, app, color, label, likelihood=None):
    """A segment with one crop interval marked, optionally over its OFF shading.

    `segment`, `a` and `b` must share one coordinate system. The caller passes the PADDED
    signal for both panels, because OffCutoffSampler returns padded indices while a naive
    crop is drawn on the raw segment; mixing the two puts panel (b)'s arrow off the axis.
    """
    x = np.arange(segment.size)
    if likelihood is not None:
        ax.fill_between(x, 0, 1, where=likelihood, transform=ax.get_xaxis_transform(),
                        color=OFF_SHADE, lw=0, zorder=0)
    ax.plot(x, segment, color=COLOR[app], lw=1.0, zorder=3)
    for v in (a, b):
        ax.axvline(v, color=color, lw=1.3, ls="--", zorder=4)
    top = float(segment.max()) * 1.30
    ax.set_xlim(0, segment.size - 1)
    ax.set_ylim(0, top)
    ax.annotate("", xy=(a, top * 0.88), xytext=(b, top * 0.88),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
    ax.set_yticks([]); ax.set_xticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(label, loc="left", fontsize=8.5, color=color, pad=4)


def draw_splice_zoom(ax, stream, at, color, label, half=30, ymax=None):
    """The neighbourhood of one crop boundary, where the splice artefact appears.

    `ymax` is shared between the two panels; autoscaling each would hide the fact that
    one step is a kilowatt and the other is a few watts.
    """
    lo = max(0, at - half)
    hi = min(stream.size, at + half)
    x = np.arange(lo, hi)
    ax.plot(x, stream[lo:hi], color=color, lw=1.2)
    ax.axvline(at, color="#777777", lw=0.8, ls=":", zorder=0)
    step = abs(float(stream[at]) - float(stream[at - 1])) if at > 0 else 0.0
    ax.set_xlim(lo, hi - 1)
    ax.set_ylim(0, (ymax if ymax else max(1.0, float(stream[lo:hi].max()))) * 1.30)
    ax.set_yticks([]); ax.set_xticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.annotate(f"{step:.0f} W step" if step >= 1 else f"{step:.1f} W step",
                xy=(at, stream[at]), xytext=(8, 10), textcoords="offset points",
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
    ax.set_title(label, loc="left", fontsize=8, color="#333333", pad=3)



def draw_peak_distribution(ax, pool, mux_peaks, single_peaks, app):
    """Activation-peak spread: the measured ensemble, the multiplexer, and one segment."""
    lo, hi = 0.0, float(np.percentile(np.concatenate([pool, mux_peaks]), 99.5)) * 1.05
    bins = np.linspace(lo, hi, 46)
    ax.hist(pool, bins=bins, density=True, color="#999999", alpha=0.55,
            label=f"measured pool (sd {pool.std():.0f} W)")
    ax.hist(mux_peaks, bins=bins, density=True, histtype="step", lw=1.4,
            color=COLOR[app], label=f"multiplexer (sd {mux_peaks.std():.0f} W)")
    ax.axvline(float(np.mean(single_peaks)), color=NAIVE_COLOR, lw=1.6, ls="--",
               label=f"one segment reused (sd {single_peaks.std():.0f} W)")
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel("activation peak power (W)", fontsize=7.5)
    ax.set_ylabel(NICE[app], rotation=0, ha="right", va="center", labelpad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")


def latex_table(stats):
    """The cut-rate table, as LaTeX, so the numbers in the paper are not transcribed."""
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Effect of boundary-aware cropping. Percentage of crops whose edge "
         r"falls inside an activation, and the mean power discontinuity that edge "
         r"introduces when the crop is spliced into a stream. 3000 crops per appliance "
         r"from the UK-DALE training pool.}",
         r"\label{table:sampler_cut_rate}",
         r"\begin{tabular}{l|cc|cc}", r"\hline",
         r"\multirow{2}{*}{Appliance} & \multicolumn{2}{c|}{Cut mid-activation} & "
         r"\multicolumn{2}{c}{Mean splice step (W)} \\ \cline{2-5}",
         r" & Naive & Boundary-aware & Naive & Boundary-aware \\ \hline"]
    for app in TABLE_APPS:
        s = stats[app]
        L.append(f" {NICE[app]} & {s['naive_cut']:.1f}\\% & "
                 f"\\textbf{{{s['aware_cut']:.1f}\\%}} & {s['naive_jump']:.0f} & "
                 f"\\textbf{{{s['aware_jump']:.1f}}} \\\\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ============================ computation ============================ #
#
# Kept apart from main() so scripts/synth_panels.py renders the same panels from the same
# numbers without copying any of it.


def load_pool(cache):
    """(segments, duty cycles, appliance_param) for the UK-DALE training pool."""
    with open(cache, "rb") as f:
        segments, on_fractions, _ = pickle.load(f)
    param = UKDALE_DataBuilder(data_path="data/UKDALE/", mask_app=APPS,
                               sampling_rate="10s", window_size=256).appliance_param
    return segments, on_fractions, param


def compute_cut_rates(segments, param, seed=0, n=3000, apps=None):
    """{appliance: cut rates and splice sizes} for naive vs boundary-aware cropping."""
    out = {}
    for app in (apps or TABLE_APPS):
        thr = float(param[app]["min_threshold"])
        mux = _build_multiplexer(app, segments[app], thr)
        out[app] = cut_rate(segments[app], thr, mux, np.random.default_rng(seed), n)
    return out


def compute_contrast(segments, on_fractions, param, app, seed=0, length=1536):
    """Everything the contrast figure draws, in one coordinate system.

    OffCutoffSampler pads its signal, so the crop indices it returns are in padded
    coordinates while a naive crop is in raw ones. Both are converted here, once.
    """
    thr = float(param[app]["min_threshold"])
    segs = segments[app]
    mux = _build_multiplexer(app, segs, thr)
    frac = _calibrate_active_fraction(mux, float(on_fractions[app]), thr)
    rng = np.random.default_rng(seed)

    # A segment the naive cut actually damages. At 93% of crops this is the typical case,
    # not a selected one, but it is searched for so the panel never shows a lucky miss.
    seg = naive_a = naive_b = None
    for _ in range(500):
        cand = segs[rng.integers(len(segs))]
        if cand.size < 40:
            continue
        ia, ib = naive_interval(cand, rng)
        if cand[ia] >= thr or cand[ib - 1] >= thr:
            seg, naive_a, naive_b = cand, ia, ib
            break
    if seg is None:
        raise ValueError(f"no naive crop cut a {app} activation; try another seed")

    median_len = float(np.median([x.size for x in segs]))
    smp = OffCutoffSampler(seg, thr, 1 if median_len < 100 else 100)
    if not smp.usable:
        smp = OffCutoffSampler(seg, thr, 1)
    aware = None
    for _ in range(500):
        u, v = sorted(rng.uniform(size=2))
        u, v = smp.sample_points(np.array([u, v]))
        if smp.unnormalized_on_cdf[v] == smp.unnormalized_on_cdf[u] or v - u < 8:
            continue
        aware = (int(u), int(v))
        break
    if aware is None:
        raise ValueError(f"no boundary-aware crop found for {app}; try another seed")

    n_stream, n_spans = naive_stream(segs, length, frac, np.random.default_rng(seed))
    w_stream, w_spans = check_mirror(mux, length, frac, seed)

    def worst_boundary(stream, spans):
        edges = [e for sp in spans for e in (sp[0], min(sp[1], stream.size - 1))
                 if 0 < e < stream.size]
        if not edges:
            return stream.size // 2
        return max(edges, key=lambda i: abs(float(stream[i]) - float(stream[i - 1])))

    pad = smp.off_window + 1
    return {
        "app": app, "threshold": thr,
        "padded": smp.signal, "likelihood": smp.likelihood,
        "naive": (naive_a + pad, naive_b + pad), "aware": aware,
        "naive_stream": n_stream, "naive_at": worst_boundary(n_stream, n_spans),
        "aware_stream": w_stream, "aware_at": worst_boundary(w_stream, w_spans),
        "zoom_max": max(float(n_stream.max()), float(w_stream.max())),
    }


def compute_diversity(segments, param, app, seed=0, n=2000):
    """(pool peaks, multiplexer peaks, one-segment peaks) for one appliance."""
    thr = float(param[app]["min_threshold"])
    segs = segments[app]
    mux = _build_multiplexer(app, segs, thr)
    pool = np.array([float(s.max()) for s in segs])
    np.random.seed(seed)
    mux_peaks = np.array([float(mux().max()) for _ in range(n)])

    rng = np.random.default_rng(seed)
    one = segs[int(rng.integers(len(segs)))]
    ow = 1 if float(np.median([x.size for x in segs])) < 100 else 100
    osmp = OffCutoffSampler(one, thr, ow)
    if not osmp.usable:
        osmp = OffCutoffSampler(one, thr, 1)
    single = DeviceMultiplexer([osmp.sample_random_interval])
    np.random.seed(seed)
    single_peaks = np.array([float(single().max()) for _ in range(n)])
    return pool, mux_peaks, single_peaks


def render_contrast(fig, data, stats):
    """The four panels of the sampler contrast onto a prepared figure."""
    app = data["app"]
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.14,
                          left=0.10, right=0.98, top=0.88, bottom=0.14)
    draw_segment_crop(fig.add_subplot(gs[0, 0]), data["padded"], *data["naive"], app,
                      NAIVE_COLOR, "(a)  naive crop: two uniform endpoints")
    draw_segment_crop(fig.add_subplot(gs[0, 1]), data["padded"], *data["aware"], app,
                      CROP_COLOR, "(b)  boundary-aware crop: endpoints in OFF regions",
                      likelihood=data["likelihood"])
    draw_splice_zoom(fig.add_subplot(gs[1, 0]), data["naive_stream"], data["naive_at"],
                     NAIVE_COLOR, "(c)  worst splice in the naive stream",
                     ymax=data["zoom_max"])
    draw_splice_zoom(fig.add_subplot(gs[1, 1]), data["aware_stream"], data["aware_at"],
                     CROP_COLOR, "(d)  worst splice, boundary-aware",
                     ymax=data["zoom_max"])
    s = stats[app]
    fig.text(0.10, 0.005,
             f"Over 3000 crops, {s['naive_cut']:.0f}% of naive crops cut the "
             f"{NICE[app].lower()} mid-activation, introducing a mean step of "
             f"{s['naive_jump']:.0f} W, against {s['aware_cut']:.1f}% and "
             f"{s['aware_jump']:.1f} W for the boundary-aware sampler.",
             fontsize=6.8, color="#555555", ha="left")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--app", default="washing_machine",
                   help="appliance for the contrast figure; one with a long cycle shows "
                        "the effect most clearly")
    p.add_argument("--diversity-apps", nargs="+", default=["washing_machine", "kettle"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=int, default=1536,
                   help="stream length the splice panels are drawn from")
    p.add_argument("--n-crops", type=int, default=3000)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--tex", type=Path, default=Path("results/table_sampler_cut_rate.tex"))
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
    segments, on_fractions, param = load_pool(a.cache)
    stats = compute_cut_rates(segments, param, a.seed, a.n_crops)

    data = compute_contrast(segments, on_fractions, param, a.app, a.seed, a.length)
    fig = plt.figure(figsize=(7.2, 3.8))
    render_contrast(fig, data, stats)
    for ext in ("pdf", "png"):
        fig.savefig(a.outdir / f"fig_sampler_contrast.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes = plt.subplots(len(a.diversity_apps), 1,
                              figsize=(7.2, 1.7 * len(a.diversity_apps)))
    axes = np.atleast_1d(axes)
    div_rows = []
    for ax_, dapp in zip(axes, a.diversity_apps):
        pool, mux_peaks, single_peaks = compute_diversity(segments, param, dapp, a.seed)
        draw_peak_distribution(ax_, pool, mux_peaks, single_peaks, dapp)
        div_rows.append((dapp, pool.std(), mux_peaks.std(), single_peaks.std()))
    fig2.suptitle("Activation amplitude spread: the multiplexer reproduces the measured "
                  "ensemble", fontsize=9, x=0.12, ha="left")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig2.savefig(a.outdir / f"fig_multiplexer_diversity.{ext}", dpi=300,
                     bbox_inches="tight")
    plt.close(fig2)

    a.tex.parent.mkdir(parents=True, exist_ok=True)
    a.tex.write_text(latex_table(stats) + "\n")

    print(f"wrote {a.outdir}/fig_sampler_contrast.pdf, "
          f"{a.outdir}/fig_multiplexer_diversity.pdf and {a.tex}\n")
    print(f"{'appliance':18s} {'naive cut':>10s} {'aware cut':>10s} "
          f"{'naive step':>11s} {'aware step':>11s}")
    for app_ in TABLE_APPS:
        s = stats[app_]
        print(f"{NICE[app_]:18s} {s['naive_cut']:9.1f}% {s['aware_cut']:9.1f}% "
              f"{s['naive_jump']:10.0f}W {s['aware_jump']:10.1f}W")
    print(f"\n{'appliance':18s} {'pool sd':>9s} {'mux sd':>9s} {'one-segment sd':>15s}")
    for dapp, ps, ms, ss in div_rows:
        print(f"{NICE[dapp]:18s} {ps:8.0f}W {ms:8.0f}W {ss:14.0f}W")
    return 0


if __name__ == "__main__":
    sys.exit(main())
