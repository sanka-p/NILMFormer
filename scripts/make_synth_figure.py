#################################################################################################################
#
# @description : Figure for the crop-and-remix synthetic aggregate used by the TCN+KL+Syn arm.
#
# Everything drawn here comes from the real generator under a fixed seed -- the activation
# pool is the same pickle scripts/run_one_expe.py trains from, and the streams are produced
# by src.helpers.tcn_ablation.generate_stream itself. Nothing is hand-drawn, so the duty
# cycles and active fractions printed on the figure are the ones the model was trained with.
#
# Panel (a) answers "where does one crop come from, and why does splicing it in leave no
# seam": the sampler draws an interval whose BOTH endpoints fall in a smoothed OFF region,
# so an activation is never cut mid-burst.
# Panel (b) answers "how does that become the model's input": one stream per appliance,
# summed into a noiseless aggregate -- the same operation `synth_aggregate_apps` applies to
# the real data (preprocessing.py:782), which is what makes train and test comparable.
#
# Usage:
#     PYTHONPATH=. ./env/bin/python -m scripts.make_synth_figure
#     PYTHONPATH=. ./env/bin/python -m scripts.make_synth_figure --seed 3 --panel-a-app dishwasher
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

from src.helpers.preprocessing import UKDALE_DataBuilder
from src.helpers.tcn_ablation import (
    DEFAULT_OFF_WINDOW,
    SHORT_BURST_OFF_WINDOW,
    OffCutoffSampler,
    _build_multiplexer,
    _calibrate_active_fraction,
    generate_stream,
)

CACHE = Path("results/tcn_aug_cache/UKDALE_h1-3-4-5_10s_w128_s0.pkl")
OUT_STEM = Path("results/fig_synth_generator")

#: Drawing order, coarsest duty cycle last so the aggregate panel reads top-to-bottom.
APPS = ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"]
NICE = {
    "kettle": "Kettle",
    "microwave": "Microwave",
    "dishwasher": "Dishwasher",
    "washing_machine": "Washing machine",
    "fridge": "Fridge",
}
#: Okabe-Ito. The matplotlib default cycle fails a colour-vision check on exactly the
#: pairs this figure needs: tab10 red/green separate by dE 4.1 under deuteranopia and
#: orange/green by 1.4 under protanopia (OKLab x100; >= 8 is the target). These five keep
#: every adjacent pair >= 9 and the worst non-adjacent pair at 7.9, which the per-strip
#: labels cover.
COLOR = {
    "kettle": "#D55E00",
    "microwave": "#CC79A7",
    "dishwasher": "#0072B2",
    "washing_machine": "#E69F00",
    "fridge": "#009E73",
}
AGG_COLOR = "#333333"
CROP_COLOR = "#D55E00"
OFF_SHADE = "#e2e7ec"

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def instrumented_stream(sample_generator, length, expected_active_percentage):
    """generate_stream, additionally returning where each crop landed.

    A mirror of src.helpers.tcn_ablation.generate_stream, kept here rather than in the
    library because only the figure needs the crop positions -- the values alone cannot
    recover them, since a crop's off padding is zero exactly like the gaps around it.
    `check_mirror` asserts the two produce identical output for the same seed, so this
    copy cannot drift from the real generator unnoticed.
    """
    target_active_length = int(length * expected_active_percentage)
    if target_active_length <= 0:
        return np.zeros(length), []
    if target_active_length >= length:
        target_active_length = length - 1

    current_length = 0
    samples = []
    while current_length < target_active_length:
        new_sample = sample_generator()
        if new_sample.size == 0:
            continue
        current_length += new_sample.size
        if current_length > target_active_length:
            new_sample = new_sample[: -(current_length - target_active_length)]
        samples.append(new_sample)

    n_off = length - target_active_length
    zeros = np.zeros(n_off)
    random_points = np.sort(np.random.randint(zeros.size, size=len(samples) - 1))
    zero_sectors = np.split(zeros, random_points)

    spans, cursor = [], 0
    for zsec, smp in itertools.zip_longest(zero_sectors, samples):
        cursor += 0 if zsec is None else zsec.size
        if smp is not None:
            spans.append((cursor, cursor + smp.size))
            cursor += smp.size

    signal = np.concatenate(
        list(itertools.chain.from_iterable(zip(zero_sectors, samples)))
    )[:length]
    return signal, [(a, b) for a, b in spans if a < length]


def check_mirror(mux, length, frac, seed):
    """Fail loudly if the instrumented copy ever stops matching the real generator."""
    np.random.seed(seed)
    ref = generate_stream(mux, length, frac)
    np.random.seed(seed)
    got, spans = instrumented_stream(mux, length, frac)
    if not np.array_equal(ref, got):
        raise AssertionError(
            "instrumented_stream has diverged from tcn_ablation.generate_stream; "
            "the figure would no longer depict the code that trains the model."
        )
    return got, spans


def sampler_for(segment, threshold, pool_median):
    """The OffCutoffSampler the generator would build for this segment.

    _build_multiplexer picks the boxcar width from the POOL's median activation length,
    then retries narrow per segment if the wide window found no OFF region. Both steps are
    mirrored here, or the shaded region would not be the one the sampler actually used --
    a 100-sample boxcar over a 47-sample kettle burst never clears 2000 W.
    """
    off_window = (
        SHORT_BURST_OFF_WINDOW if pool_median < DEFAULT_OFF_WINDOW else DEFAULT_OFF_WINDOW
    )
    smp = OffCutoffSampler(segment, threshold, off_window)
    if not smp.usable and off_window != SHORT_BURST_OFF_WINDOW:
        smp = OffCutoffSampler(segment, threshold, SHORT_BURST_OFF_WINDOW)
    return smp


def sample_interval_with_bounds(smp, rng, margin=0.10):
    """One crop, with the endpoints it was cut at -- sample_random_interval drops them.

    `margin` additionally demands that both endpoints sit away from the segment's own
    edges. The generator imposes no such thing, but a crop running edge to edge shows the
    reader nothing: the point of the panel is that the cut lands in an OFF region, which
    is only visible when there is OFF context left on each side of it.
    """
    n = smp.signal.size
    lo, hi = int(margin * n), int((1 - margin) * n)
    for _ in range(400):
        a, b = rng.uniform(size=2)
        a, b = min(a, b), max(a, b)
        a, b = smp.sample_points(np.array([a, b]))
        if smp.unnormalized_on_cdf[b] == smp.unnormalized_on_cdf[a]:
            continue
        if b - a < 8 or a < lo or b > hi:
            continue
        return int(a), int(b)
    return None


def pick_excerpt(streams, length, excerpt, stride=128):
    """Start index of the excerpt showing the most appliances at once.

    The streams have to be generated at the FULL training length, not at the width of the
    figure. generate_stream lays out `z1,c1,z2,c2,...,zK,cK`, so the last crop always ends
    at the final sample; over the 25k-sample streams the model trains on that edge effect
    is negligible, but at figure width it would put every rare appliance in the last
    column and misrepresent the generator.
    """
    best, best_key = 0, None
    for start in range(0, length - excerpt + 1, stride):
        seg = {a: v[start:start + excerpt] for a, v in streams.items()}
        n_active = sum(1 for v in seg.values() if v.max() > 0)
        energy = sum(float(v.sum()) for v in seg.values())
        key = (n_active, energy)
        if best_key is None or key > best_key:
            best, best_key = start, key
    return best


def pick_segment(segments, threshold, pool_median, rng, size_pct=80):
    """A segment with one clean burst and off context on both sides, for panel (a).

    The length floor is a percentile of THIS pool rather than a constant: a kettle
    activation runs 21-76 samples and a dishwasher's fills the whole 128-sample window,
    so any fixed threshold either rejects every kettle segment or accepts unreadable ones.
    """
    lo = float(np.percentile([x.size for x in segments], size_pct))
    order = rng.permutation(len(segments))
    best = None
    for i in order:
        seg = segments[i]
        if seg.size < lo:
            continue
        on = seg >= threshold
        if not on.any():
            continue
        runs = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
        if int(np.count_nonzero(runs == 1)) != 1:
            continue  # exactly one activation reads most clearly
        if on[0] or on[-1]:
            continue  # need genuine off context at both ends
        smp = sampler_for(seg, threshold, pool_median)
        if not smp.usable:
            continue
        bounds = sample_interval_with_bounds(smp, rng)
        if bounds is None:
            continue
        best = (seg, smp, bounds)
        break
    return best


# ============================ panel drawing ============================ #
#
# One function per panel, each taking only an Axes and its data. The composed figure
# below and scripts/synth_panels.py (which renders the panels separately, to be placed
# by hand) both call these, so the two cannot drift apart.


def strip(ax, y, color, label, right_text=None, lw=0.9):
    """One trace on a bare strip: no y-scale, label at the left, note above the right."""
    ax.plot(np.arange(y.size), y, color=color, lw=lw)
    ax.set_xlim(0, y.size - 1)
    ax.set_ylim(0, max(1.0, float(y.max())) * 1.18)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=6, fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if right_text:
        ax.text(0.998, 1.04, right_text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="#555555")


def draw_pool(ax, pool_show, app, n_pool, title=True):
    """Three activation segments side by side, as cut from the training houses."""
    cursor = 0
    for s in pool_show:
        ax.plot(np.arange(cursor, cursor + s.size), s, color=COLOR[app], lw=0.9)
        cursor += s.size + 25
    ax.set_xlim(-5, cursor)
    ax.set_yticks([]); ax.set_xticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("activation\npool", rotation=0, ha="right", va="center", labelpad=6)
    if title:
        ax.set_title(f"{NICE[app]} activations ({n_pool:,} in the pool, 3 shown)",
                     loc="left", fontsize=9, pad=10)


def draw_segment(ax, smp, ca, cb, app):
    """One segment with its OFF regions shaded and the sampled crop interval marked."""
    x = np.arange(smp.signal.size)
    ax.fill_between(x, 0, 1, where=smp.likelihood, transform=ax.get_xaxis_transform(),
                    color=OFF_SHADE, lw=0, zorder=0)
    ax.plot(x, smp.signal, color=COLOR[app], lw=1.0, zorder=3)
    for v in (ca, cb):
        ax.axvline(v, color=CROP_COLOR, lw=1.2, ls="--", zorder=4)
    ax.set_xlim(0, smp.signal.size - 1)
    top = float(smp.signal.max()) * 1.35
    ax.set_ylim(0, top)
    ax.annotate("", xy=(ca, top * 0.86), xytext=(cb, top * 0.86),
                arrowprops=dict(arrowstyle="<->", color=CROP_COLOR, lw=1.1))
    ax.text((ca + cb) / 2, top * 0.90, "sampled crop", color=CROP_COLOR,
            ha="center", va="bottom", fontsize=7.5)
    ax.text(0.0, 1.04, "shaded = OFF region (boxcar mean < threshold); both crop "
                       "endpoints must land inside one",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color="#0072B2")
    ax.set_ylabel("one\nsegment", rotation=0, ha="right", va="center", labelpad=6)
    ax.set_xticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_crop(ax, crop):
    """The crop on its own -- real data, taken unmodified."""
    strip(ax, crop, CROP_COLOR, "crop", right_text=f"{crop.size} samples, taken unmodified")
    ax.set_xticks([])


def draw_stream(ax, stream, spans, app, duty, frac, offset, full_length):
    """Crops interleaved with the zero gaps, their positions shaded."""
    for s0, s1 in spans:
        ax.axvspan(s0, s1, color=CROP_COLOR, alpha=0.16, lw=0)
    strip(ax, stream, COLOR[app], "assembled\nstream")
    ax.text(0.0, 1.04,
            f"crops placed at random offsets, gaps are zero  |  measured duty "
            f"{duty:.4f} $\\rightarrow$ active fraction {frac:.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color="#555555")
    ax.set_xlabel(f"sampling points  (excerpt from {offset:,} of {full_length:,})",
                  fontsize=7.5)


def draw_appliance(ax, y, app, duty, frac):
    """One appliance's stream, with its duty, calibrated fraction and peak."""
    strip(ax, y, COLOR[app], NICE[app],
          right_text=f"duty {duty:.4f} $\\rightarrow$ active {frac:.3f}   |   "
                     f"peak {y.max():.0f} W")
    ax.set_xticks([])


def draw_aggregate(ax, aggregate, window_size, offset, full_length):
    """The sum of the five streams, with window boundaries."""
    strip(ax, aggregate, AGG_COLOR, "Synthetic\naggregate", lw=0.8)
    ax.text(0.998, 1.04, f"peak {aggregate.max():.0f} W", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="#555555")
    for b in range(window_size, aggregate.size, window_size):
        ax.axvline(b, color="#999999", lw=0.6, ls=":", zorder=0)
    # The window-boundary note lives in the xlabel rather than beside the peak: at
    # single-column width the two top annotations overlap.
    ax.text(0.0, 1.04, "sum of the five streams, no added noise",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color="#555555")
    ax.set_xlabel(f"sampling points  (excerpt from {offset:,} of {full_length:,}; "
                  f"dotted = {window_size}-sample windows)", fontsize=7.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel-a-app", default="kettle")
    p.add_argument("--panel-b-length", type=int, default=2560,
                   help="width of the excerpt drawn, in samples (10 windows)")
    p.add_argument("--full-length", type=int, default=25600,
                   help="length the streams are actually generated at, as in training")
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=OUT_STEM)
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    if not a.cache.is_file():
        raise SystemExit(
            f"{a.cache} not found. It is written by scripts/run_one_expe.py on the first "
            "TCN+Syn run; train one arm, or point --cache at another cached pool."
        )
    with open(a.cache, "rb") as f:
        segments, on_fractions, _off_levels = pickle.load(f)

    param = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=APPS, sampling_rate="10s",
        window_size=a.window_size,
    ).appliance_param

    rng = np.random.default_rng(a.seed)

    # ---------------- panel (a): one crop, from pool to stream ----------------
    app_a = a.panel_a_app
    thr_a = float(param[app_a]["min_threshold"])
    median_a = float(np.median([x.size for x in segments[app_a]]))
    picked = pick_segment(segments[app_a], thr_a, median_a, rng)
    if picked is None:
        raise SystemExit(f"No segment of {app_a} suited panel (a); try another --seed.")
    seg, smp, (ca, cb) = picked
    pad = smp.off_window + 1                      # OffCutoffSampler pads both ends
    crop = smp.signal[ca:cb]

    pool_show = [segments[app_a][i] for i in rng.permutation(len(segments[app_a]))[:3]]

    mux_a = _build_multiplexer(app_a, segments[app_a], thr_a)
    frac_a = _calibrate_active_fraction(mux_a, float(on_fractions[app_a]), thr_a)
    stream_a_full, spans_a_full = check_mirror(mux_a, a.full_length, frac_a, a.seed)
    # Excerpt around the first pair of crops close enough to appear together.
    view = 5 * a.window_size
    pair = next(
        ((s0, s1) for (s0, _), (s1, e1) in zip(spans_a_full, spans_a_full[1:])
         if e1 - s0 < view * 0.8),
        None,
    )
    anchor_a = pair[0] if pair else (spans_a_full[0][0] if spans_a_full else 0)
    a0 = int(np.clip(anchor_a - view * 0.25, 0, max(0, a.full_length - view)))
    stream_a = stream_a_full[a0:a0 + view]
    spans_a = [(s - a0, e - a0) for s, e in spans_a_full if s < a0 + view and e > a0]

    # ---------------- panel (b): five streams, summed ----------------
    full, fracs = {}, {}
    for app in APPS:
        thr = float(param[app]["min_threshold"])
        mux = _build_multiplexer(app, segments[app], thr)
        fracs[app] = _calibrate_active_fraction(mux, float(on_fractions[app]), thr)
        np.random.seed(a.seed + APPS.index(app))
        full[app] = generate_stream(mux, a.full_length, fracs[app])
    b0 = pick_excerpt(full, a.full_length, a.panel_b_length)
    streams = {x: full[x][b0:b0 + a.panel_b_length] for x in APPS}
    aggregate = np.sum([streams[x] for x in APPS], axis=0)

    # ---------------- draw ----------------
    fig = plt.figure(figsize=(7.2, 9.2))
    gs = fig.add_gridspec(
        11, 1, height_ratios=[1.0, 1.5, 0.7, 1.1, 1.15] + [0.8] * 5 + [1.4],
        hspace=0.95, left=0.19, right=0.97, top=0.95, bottom=0.05,
    )

    ax = fig.add_subplot(gs[0])
    draw_pool(ax, pool_show, app_a, len(segments[app_a]), title=False)
    ax.set_title(
        f"(a)  Where one crop comes from: {NICE[app_a].lower()} activations "
        f"({len(segments[app_a]):,} in the pool, 3 shown)", loc="left", fontsize=9, pad=10,
    )

    draw_segment(fig.add_subplot(gs[1]), smp, ca, cb, app_a)
    draw_crop(fig.add_subplot(gs[2]), crop)
    draw_stream(fig.add_subplot(gs[3]), stream_a, spans_a, app_a,
                float(on_fractions[app_a]), frac_a, a0, a.full_length)

    ax = fig.add_subplot(gs[4]); ax.axis("off")
    ax.text(0.0, 0.62, "(b)  One stream per appliance, summed into the synthetic aggregate",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.text(0.0, 0.0, "each strip is scaled to its own peak, given at the right",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color="#555555")

    for i, app in enumerate(APPS):
        draw_appliance(fig.add_subplot(gs[5 + i]), streams[app], app,
                       float(on_fractions[app]), fracs[app])

    draw_aggregate(fig.add_subplot(gs[10]), aggregate, a.window_size, b0, a.full_length)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {a.out}.pdf and {a.out}.png")
    print(f"panel (a): {NICE[app_a]}, segment {seg.size} samples, crop [{ca}:{cb}] "
          f"= {crop.size} samples, {len(spans_a)} crops in the stream")
    print(f"{'appliance':18s} {'duty':>8} {'active frac':>12} {'stream max W':>13}")
    for app in APPS:
        print(f"{NICE[app]:18s} {on_fractions[app]:8.4f} {fracs[app]:12.4f} "
              f"{streams[app].max():13.0f}")
    print(f"{'aggregate':18s} {'':8} {'':12} {aggregate.max():13.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
