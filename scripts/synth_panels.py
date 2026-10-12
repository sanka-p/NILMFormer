import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import pickle
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(os.path.dirname(__file__)).resolve().parents[0]))

    # Every helper comes from the composed-figure script, so the panels drawn here cannot
    # drift from results/fig_synth_generator.pdf. Importing it also applies its rcParams,
    # which is what makes a panel dropped into the paper match the rest.
    from scripts.make_synth_figure import (
        APPS,
        NICE,
        check_mirror,
        draw_aggregate,
        draw_appliance,
        draw_crop,
        draw_pool,
        draw_segment,
        draw_stream,
        pick_excerpt,
        pick_segment,
    )
    from scripts.make_sampler_figure import (
        TABLE_APPS,
        compute_contrast,
        compute_cut_rates,
        compute_diversity,
        draw_peak_distribution,
        render_contrast,
    )
    from src.helpers.preprocessing import UKDALE_DataBuilder
    from src.helpers.tcn_ablation import (
        _build_multiplexer,
        _calibrate_active_fraction,
        generate_stream,
    )

    mo.md(
        """
        # Synthetic generator — panels

        Each section of `results/fig_synth_generator.pdf` as its own figure, so they can be
        placed and sized independently. Drawing code and styling are imported from
        `scripts/make_synth_figure.py`; nothing is duplicated, so a change there shows up
        here.
        """
    )
    return (
        APPS,
        Path,
        UKDALE_DataBuilder,
        check_mirror,
        compute_contrast,
        compute_cut_rates,
        compute_diversity,
        draw_aggregate,
        draw_appliance,
        draw_crop,
        draw_peak_distribution,
        draw_pool,
        draw_segment,
        draw_stream,
        generate_stream,
        io,
        mo,
        np,
        pd,
        pick_excerpt,
        pick_segment,
        pickle,
        plt,
        render_contrast,
    )


@app.cell
def _(mo):
    cache_input = mo.ui.text(
        value="results/tcn_aug_cache/UKDALE_h1-3-4-5_10s_w128_s0.pkl",
        label="Activation pool", full_width=True,
    )
    seed_num = mo.ui.number(value=0, label="Seed")
    app_a_dd = mo.ui.dropdown(
        ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"],
        value="kettle", label="Panel (a) appliance",
    )
    full_len_num = mo.ui.number(value=25600, label="Stream length (generated)")
    excerpt_num = mo.ui.number(value=2560, label="Excerpt width (drawn)")
    ws_num = mo.ui.number(value=256, label="Window size")
    build_btn = mo.ui.run_button(label="Build panels")

    mo.vstack([
        cache_input,
        mo.hstack([seed_num, app_a_dd, full_len_num, excerpt_num, ws_num]),
        build_btn,
    ])
    return (
        app_a_dd,
        build_btn,
        cache_input,
        excerpt_num,
        full_len_num,
        seed_num,
        ws_num,
    )


@app.cell
def _(mo):
    fig_w_num = mo.ui.number(value=6.5, step=0.25, label="Panel width (in)")
    strip_h_num = mo.ui.number(value=1.1, step=0.1, label="Strip height (in)")
    tall_h_num = mo.ui.number(value=1.8, step=0.1, label="Segment panel height (in)")
    mo.hstack([fig_w_num, strip_h_num, tall_h_num])
    return fig_w_num, strip_h_num, tall_h_num


@app.cell
def _(
    APPS,
    UKDALE_DataBuilder,
    app_a_dd,
    build_btn,
    cache_input,
    check_mirror,
    excerpt_num,
    full_len_num,
    generate_stream,
    mo,
    np,
    pick_excerpt,
    pick_segment,
    pickle,
    seed_num,
    ws_num,
):
    mo.stop(not build_btn.value, mo.md("Click **Build panels**."))

    _seed = int(seed_num.value)
    _ws = int(ws_num.value)
    _full = int(full_len_num.value)
    _excerpt = int(excerpt_num.value)
    app_a = app_a_dd.value

    with open(cache_input.value, "rb") as _f:
        segments, on_fractions, _off_levels = pickle.load(_f)

    param = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=APPS, sampling_rate="10s", window_size=_ws,
    ).appliance_param

    _rng = np.random.default_rng(_seed)

    # ---- panel (a) ----
    thr_a = float(param[app_a]["min_threshold"])
    _median_a = float(np.median([x.size for x in segments[app_a]]))
    _picked = pick_segment(segments[app_a], thr_a, _median_a, _rng)
    mo.stop(
        _picked is None,
        mo.md(f"No segment of {app_a} suited panel (a) — try another seed."),
    )
    seg, smp, (ca, cb) = _picked
    crop = smp.signal[ca:cb]
    pool_show = [segments[app_a][i] for i in _rng.permutation(len(segments[app_a]))[:3]]

    _mux_a = _build_multiplexer(app_a, segments[app_a], thr_a)
    frac_a = _calibrate_active_fraction(_mux_a, float(on_fractions[app_a]), thr_a)
    _stream_full, _spans_full = check_mirror(_mux_a, _full, frac_a, _seed)

    _view = 5 * _ws
    _pair = next(
        ((s0, s1) for (s0, _), (s1, e1) in zip(_spans_full, _spans_full[1:])
         if e1 - s0 < _view * 0.8),
        None,
    )
    _anchor = _pair[0] if _pair else (_spans_full[0][0] if _spans_full else 0)
    a0 = int(np.clip(_anchor - _view * 0.25, 0, max(0, _full - _view)))
    stream_a = _stream_full[a0:a0 + _view]
    spans_a = [(s - a0, e - a0) for s, e in _spans_full if s < a0 + _view and e > a0]

    # ---- panel (b) ----
    # Generated at the full training length, then excerpted: generate_stream lays out
    # z1,c1,...,zK,cK, so the last crop always ends on the final sample. At figure width
    # that edge effect would park every rare appliance in the last column.
    _fullstreams, fracs = {}, {}
    for _app in APPS:
        _thr = float(param[_app]["min_threshold"])
        _mux = _build_multiplexer(_app, segments[_app], _thr)
        fracs[_app] = _calibrate_active_fraction(_mux, float(on_fractions[_app]), _thr)
        np.random.seed(_seed + APPS.index(_app))
        _fullstreams[_app] = generate_stream(_mux, _full, fracs[_app])

    b0 = pick_excerpt(_fullstreams, _full, _excerpt)
    streams = {x: _fullstreams[x][b0:b0 + _excerpt] for x in APPS}
    aggregate = np.sum([streams[x] for x in APPS], axis=0)
    ws = _ws
    full_length = _full

    mo.md(
        f"Built. Panel (a): **{app_a}**, segment {seg.size} samples, crop "
        f"`[{ca}:{cb}]` = {crop.size} samples, {len(spans_a)} crops in the excerpt "
        f"(from sample {a0:,}). Panel (b) excerpt from sample {b0:,} of {_full:,}."
    )
    return (
        a0,
        aggregate,
        app_a,
        b0,
        ca,
        cb,
        crop,
        frac_a,
        fracs,
        full_length,
        on_fractions,
        param,
        pool_show,
        segments,
        smp,
        spans_a,
        stream_a,
        streams,
        ws,
    )


@app.cell
def _(io, mo, plt):
    def panel(draw, name, w, h, left=0.22):
        """Render one draw function into its own figure, with a download button."""
        fig, ax = plt.subplots(1, 1, figsize=(w, h))
        fig.subplots_adjust(left=left, right=0.98, top=0.80, bottom=0.30)
        draw(ax)
        png, pdf = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return mo.vstack([
            mo.image(src=png.getvalue()),
            mo.hstack([
                mo.download(data=png.getvalue(), filename=f"{name}.png", label="PNG"),
                mo.download(data=pdf.getvalue(), filename=f"{name}.pdf", label="PDF"),
            ], justify="start"),
        ])

    return (panel,)


@app.cell
def _(
    app_a,
    draw_pool,
    fig_w_num,
    mo,
    panel,
    pool_show,
    segments,
    strip_h_num,
):
    mo.vstack([
        mo.md("### (a1) Activation pool"),
        panel(lambda ax: draw_pool(ax, pool_show, app_a, len(segments[app_a])),
              "synth_a1_pool", float(fig_w_num.value), float(strip_h_num.value)),
    ])
    return


@app.cell
def _(app_a, ca, cb, draw_segment, fig_w_num, mo, panel, smp, tall_h_num):
    mo.vstack([
        mo.md("### (a2) One segment, OFF regions shaded, crop interval marked"),
        panel(lambda ax: draw_segment(ax, smp, ca, cb, app_a),
              "synth_a2_segment", float(fig_w_num.value), float(tall_h_num.value)),
    ])
    return


@app.cell
def _(crop, draw_crop, fig_w_num, mo, panel, strip_h_num):
    mo.vstack([
        mo.md("### (a3) The crop"),
        panel(lambda ax: draw_crop(ax, crop),
              "synth_a3_crop", float(fig_w_num.value), float(strip_h_num.value) * 0.85),
    ])
    return


@app.cell
def _(
    a0,
    app_a,
    draw_stream,
    fig_w_num,
    frac_a,
    full_length,
    mo,
    on_fractions,
    panel,
    spans_a,
    stream_a,
    strip_h_num,
):
    mo.vstack([
        mo.md("### (a4) Crops interleaved with zero gaps"),
        panel(
            lambda ax: draw_stream(ax, stream_a, spans_a, app_a,
                                   float(on_fractions[app_a]), frac_a, a0, full_length),
            "synth_a4_stream", float(fig_w_num.value), float(strip_h_num.value) * 1.25,
        ),
    ])
    return


@app.cell
def _(
    APPS,
    draw_appliance,
    fig_w_num,
    fracs,
    mo,
    on_fractions,
    panel,
    streams,
    strip_h_num,
):
    mo.vstack(
        [mo.md("### (b1–b5) One stream per appliance")]
        + [
            panel(
                lambda ax, _a=_app: draw_appliance(
                    ax, streams[_a], _a, float(on_fractions[_a]), fracs[_a]
                ),
                f"synth_b_{_app}", float(fig_w_num.value), float(strip_h_num.value),
            )
            for _app in APPS
        ]
    )
    return


@app.cell
def _(
    aggregate,
    b0,
    draw_aggregate,
    fig_w_num,
    full_length,
    mo,
    panel,
    strip_h_num,
    ws,
):
    mo.vstack([
        mo.md("### (b6) Synthetic aggregate"),
        panel(lambda ax: draw_aggregate(ax, aggregate, ws, b0, full_length),
              "synth_b6_aggregate", float(fig_w_num.value),
              float(strip_h_num.value) * 1.35),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Why the sampler and the multiplexer are there

    The panels above show *how* the generator works. These two show *why* each piece
    is needed, by measuring what happens without it.
    """)
    return


@app.cell
def _(mo):
    ev_app_dd = mo.ui.dropdown(
        ["washing_machine", "dishwasher", "kettle", "microwave", "fridge"],
        value="washing_machine", label="Contrast appliance",
    )
    ev_div_dd = mo.ui.multiselect(
        ["washing_machine", "dishwasher", "kettle", "microwave", "fridge"],
        value=["washing_machine", "kettle"], label="Diversity appliances",
    )
    ev_n_num = mo.ui.number(value=3000, label="Crops sampled per appliance")
    ev_btn = mo.ui.run_button(label="Measure")
    mo.vstack([mo.hstack([ev_app_dd, ev_n_num, ev_btn]), ev_div_dd])
    return ev_app_dd, ev_btn, ev_div_dd, ev_n_num


@app.cell
def _(compute_cut_rates, ev_btn, ev_n_num, mo, param, pd, seed_num, segments):
    mo.stop(not ev_btn.value, mo.md("Click **Measure**."))
    cut_stats = compute_cut_rates(segments, param, int(seed_num.value), int(ev_n_num.value))
    mo.vstack([
        mo.md(
            "**Boundary-aware vs naive cropping.** A naive crop takes two uniform "
            "endpoints, so it cuts through an activation most of the time; spliced into "
            "a stream, that edge becomes a step no meter ever produces."
        ),
        pd.DataFrame([
            {"appliance": _a,
             "naive cut %": round(cut_stats[_a]["naive_cut"], 1),
             "boundary-aware cut %": round(cut_stats[_a]["aware_cut"], 1),
             "naive step (W)": round(cut_stats[_a]["naive_jump"]),
             "boundary-aware step (W)": round(cut_stats[_a]["aware_jump"], 1)}
            for _a in cut_stats
        ]),
    ])
    return (cut_stats,)


@app.cell
def _(
    compute_contrast,
    cut_stats,
    ev_app_dd,
    fig_w_num,
    io,
    mo,
    on_fractions,
    param,
    plt,
    render_contrast,
    seed_num,
    segments,
):
    _data = compute_contrast(segments, on_fractions, param, ev_app_dd.value,
                             int(seed_num.value))
    _fig = plt.figure(figsize=(float(fig_w_num.value), 3.8))
    render_contrast(_fig, _data, cut_stats)
    _png, _pdf = io.BytesIO(), io.BytesIO()
    _fig.savefig(_png, format="png", dpi=300, bbox_inches="tight")
    _fig.savefig(_pdf, format="pdf", bbox_inches="tight")
    plt.close(_fig)
    mo.vstack([
        mo.md("### Sampler: boundary-aware vs naive cropping"),
        mo.image(src=_png.getvalue()),
        mo.hstack([
            mo.download(data=_png.getvalue(), filename="fig_sampler_contrast.png",
                        label="PNG"),
            mo.download(data=_pdf.getvalue(), filename="fig_sampler_contrast.pdf",
                        label="PDF"),
        ], justify="start"),
    ])
    return


@app.cell
def _(
    compute_diversity,
    draw_peak_distribution,
    ev_div_dd,
    fig_w_num,
    io,
    mo,
    np,
    param,
    plt,
    seed_num,
    segments,
):
    _apps = list(ev_div_dd.value) or ["washing_machine"]
    _fig, _axes = plt.subplots(len(_apps), 1,
                               figsize=(float(fig_w_num.value), 1.7 * len(_apps)))
    for _ax, _a in zip(np.atleast_1d(_axes), _apps):
        _pool, _mux, _single = compute_diversity(segments, param, _a, int(seed_num.value))
        draw_peak_distribution(_ax, _pool, _mux, _single, _a)
    _fig.suptitle("Activation amplitude spread: the multiplexer reproduces the measured "
                  "ensemble", fontsize=9, x=0.12, ha="left")
    _fig.tight_layout(rect=(0, 0, 1, 0.95))
    _png2, _pdf2 = io.BytesIO(), io.BytesIO()
    _fig.savefig(_png2, format="png", dpi=300, bbox_inches="tight")
    _fig.savefig(_pdf2, format="pdf", bbox_inches="tight")
    plt.close(_fig)
    mo.vstack([
        mo.md(
            "### Multiplexer: pool-wide sampling vs reusing one segment\n"
            "Without the multiplexer every generated activation carries a single "
            "amplitude; with it, the stream inherits the measured ensemble's spread."
        ),
        mo.image(src=_png2.getvalue()),
        mo.hstack([
            mo.download(data=_png2.getvalue(),
                        filename="fig_multiplexer_diversity.png", label="PNG"),
            mo.download(data=_pdf2.getvalue(),
                        filename="fig_multiplexer_diversity.pdf", label="PDF"),
        ], justify="start"),
    ])
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save every panel to results/synth_panels/")
    save_btn
    return (save_btn,)


@app.cell
def _(
    APPS,
    Path,
    a0,
    aggregate,
    app_a,
    b0,
    ca,
    cb,
    crop,
    draw_aggregate,
    draw_appliance,
    draw_crop,
    draw_pool,
    draw_segment,
    draw_stream,
    fig_w_num,
    frac_a,
    fracs,
    full_length,
    mo,
    on_fractions,
    plt,
    pool_show,
    save_btn,
    segments,
    smp,
    spans_a,
    stream_a,
    streams,
    strip_h_num,
    tall_h_num,
    ws,
):
    mo.stop(not save_btn.value, mo.md("Click to write PDF + PNG for every panel."))

    _w = float(fig_w_num.value)
    _h = float(strip_h_num.value)
    _jobs = [
        ("synth_a1_pool", _h,
         lambda ax: draw_pool(ax, pool_show, app_a, len(segments[app_a]))),
        ("synth_a2_segment", float(tall_h_num.value),
         lambda ax: draw_segment(ax, smp, ca, cb, app_a)),
        ("synth_a3_crop", _h * 0.85, lambda ax: draw_crop(ax, crop)),
        ("synth_a4_stream", _h * 1.25,
         lambda ax: draw_stream(ax, stream_a, spans_a, app_a,
                                float(on_fractions[app_a]), frac_a, a0, full_length)),
    ] + [
        (f"synth_b_{_a}", _h,
         lambda ax, _a=_a: draw_appliance(ax, streams[_a], _a,
                                          float(on_fractions[_a]), fracs[_a]))
        for _a in APPS
    ] + [
        ("synth_b6_aggregate", _h * 1.35,
         lambda ax: draw_aggregate(ax, aggregate, ws, b0, full_length)),
    ]

    _out = Path("results/synth_panels")
    _out.mkdir(parents=True, exist_ok=True)
    _written = []
    for _name, _hh, _draw in _jobs:
        _fig, _ax = plt.subplots(1, 1, figsize=(_w, _hh))
        _fig.subplots_adjust(left=0.22, right=0.98, top=0.80, bottom=0.30)
        _draw(_ax)
        for _ext in ("pdf", "png"):
            _fig.savefig(_out / f"{_name}.{_ext}", dpi=300, bbox_inches="tight")
        plt.close(_fig)
        _written.append(_name)

    mo.md("**Wrote** `results/synth_panels/`:\n"
          + "\n".join(f"- `{n}.pdf` / `{n}.png`" for n in _written))
    return


if __name__ == "__main__":
    app.run()
