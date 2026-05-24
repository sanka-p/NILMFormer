import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import glob
    import io
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    import torch
    import yaml

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.helpers.preprocessing import (
        UKDALE_DataBuilder,
        REFIT_DataBuilder,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
    )
    from src.helpers.metrics import NILMmetrics
    return (
        NILMmetrics,
        REFIT_DataBuilder,
        UKDALE_DataBuilder,
        glob,
        io,
        matplotlib,
        mo,
        np,
        os,
        pd,
        plt,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
        sys,
        torch,
        yaml,
    )


@app.cell
def _(mo):
    result_path_input = mo.ui.text(value="result/", label="Result path", full_width=True)
    data_path_input = mo.ui.text(value="data/", label="Data path", full_width=True)
    dataset_dd = mo.ui.dropdown(["UKDALE", "REFIT"], value="UKDALE", label="Dataset")
    sr_dd = mo.ui.dropdown(
        ["10s", "1min", "10min", "30min"], value="10s", label="Sampling rate"
    )
    window_size_num = mo.ui.number(value=512, label="Window size")
    seed_num = mo.ui.number(value=0, label="Seed")
    zoom_frac_num = mo.ui.number(value=0.15, label="Zoom fraction (0–1)")
    max_pts_num = mo.ui.number(value=0, label="Max points to visualize (0 = all)")
    threshold_num = mo.ui.number(value=10, label="On/off threshold (W)")
    load_btn = mo.ui.run_button(label="Load & Plot")

    mo.vstack([
        mo.hstack([result_path_input, data_path_input]),
        mo.hstack([dataset_dd, sr_dd, window_size_num, seed_num, zoom_frac_num, max_pts_num, threshold_num]),
        load_btn,
    ])
    return (
        data_path_input,
        dataset_dd,
        load_btn,
        max_pts_num,
        result_path_input,
        seed_num,
        sr_dd,
        threshold_num,
        window_size_num,
        zoom_frac_num,
    )


@app.cell
def _(
    REFIT_DataBuilder,
    UKDALE_DataBuilder,
    glob,
    io,
    mo,
    np,
    os,
    plt,
    split_train_test_pdl_nilmdataset,
    torch,
):
    def load_test_data(dataset, app_cfg, data_path, sr, ws, seed):
        if dataset == "UKDALE":
            builder = UKDALE_DataBuilder(
                data_path=f"{data_path}UKDALE/",
                mask_app=app_cfg["app"],
                sampling_rate=sr,
                window_size=ws,
            )
            data_test, _ = builder.get_nilm_dataset(
                house_indicies=app_cfg["ind_house_test"]
            )
        else:
            builder = REFIT_DataBuilder(
                data_path=f"{data_path}REFIT/RAW_DATA_CLEAN/",
                mask_app=app_cfg["app"].strip(),
                sampling_rate=sr,
                window_size=ws,
            )
            data_all, st_all = builder.get_nilm_dataset(
                house_indicies=app_cfg["house_with_app_i"]
            )
            _, _, data_test, _ = split_train_test_pdl_nilmdataset(
                data_all, st_all, nb_house_test=2, seed=seed
            )
        return data_test

    def load_predictions(result_path, dataset, appliance_key, sr, ws, seed):
        ckpt_dir = os.path.join(result_path, f"{dataset}_{appliance_key}_{sr}", str(ws))
        predictions = {}
        if not os.path.isdir(ckpt_dir):
            return predictions
        for pt_file in sorted(glob.glob(os.path.join(ckpt_dir, f"*_{seed}.pt"))):
            model_name = os.path.basename(pt_file).replace(f"_{seed}.pt", "")
            try:
                log = torch.load(pt_file, map_location="cpu", weights_only=False)
                yhat = log.get("test_metrics_yhat")
                if yhat is not None:
                    predictions[model_name] = np.array(yhat).reshape(-1, ws)
            except Exception:
                pass
        return predictions

    def _pick_zoom(gt_power, zoom_frac):
        n_pts = len(gt_power)
        zoom_len = max(int(n_pts * zoom_frac), 50)
        kernel = np.ones(zoom_len) / zoom_len
        smoothed = np.convolve(gt_power, kernel, mode="valid")
        zoom_start = int(np.argmax(smoothed)) if len(smoothed) else 0
        zoom_end = min(zoom_start + zoom_len, n_pts - 1)
        return zoom_start, zoom_end

    def _power_unit(ymax_w):
        if ymax_w >= 1500:
            return 1.0 / 1000.0, "Power (kW)"
        return 1.0, "Power (W)"

    def _draw_panel(ax, x, gt, pred, zoom_start, zoom_end,
                    ymax, scale, ylabel, title):
        n_pts = len(gt)
        gt_s = gt * scale
        pred_s = pred * scale
        ymax_s = ymax * scale

        ax.plot(x, pred_s, color="#ff7f0e", lw=0.9, label="Prediction", zorder=2)
        ax.plot(x, gt_s, color="#2ca02c", lw=0.9, label="Ground-Truth", zorder=3)
        ax.set_xlim(0, n_pts - 1)
        ax.set_ylim(0, ymax_s)
        ax.set_xlabel("Sampling points")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        zs = max(int(zoom_start), 0)
        ze = min(int(zoom_end), n_pts - 1)
        if ze <= zs:
            return

        span_frac_l = zs / max(n_pts - 1, 1)
        span_frac_r = ze / max(n_pts - 1, 1)
        span_width = span_frac_r - span_frac_l
        inset_width = min(max(span_width * 1.6, 0.18), 0.55)
        center = (span_frac_l + span_frac_r) / 2
        x0 = min(max(center - inset_width / 2, 0.02), 0.98 - inset_width)
        inset_rect = (x0, 0.58, inset_width, 0.38)

        axins = ax.inset_axes(inset_rect)
        axins.plot(x[zs:ze + 1], pred_s[zs:ze + 1], color="#ff7f0e", lw=0.9, zorder=2)
        axins.plot(x[zs:ze + 1], gt_s[zs:ze + 1], color="#2ca02c", lw=0.9, zorder=3)

        zoom_max = float(max(gt_s[zs:ze + 1].max(), pred_s[zs:ze + 1].max(), 1e-6))
        axins.set_xlim(zs, ze)
        axins.set_ylim(0, zoom_max * 1.1)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_facecolor("white")
        for spine in axins.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("black")
        ax.indicate_inset_zoom(axins, edgecolor="black", lw=0.5, alpha=0.85)

    def _ymax_with_headroom(gt_power, predictions):
        ymax_data = max(float(gt_power.max()), 1.0)
        for p in predictions.values():
            ymax_data = max(ymax_data, float(p.max()))
        return ymax_data * 1.95

    def make_figure(gt_power, predictions, app_name, zoom_frac):
        n_pts = len(gt_power)
        x = np.arange(n_pts)
        model_names = list(predictions.keys())
        n_models = len(model_names)

        ymax = _ymax_with_headroom(gt_power, predictions)
        scale, ylabel = _power_unit(ymax)
        zoom_start, zoom_end = _pick_zoom(gt_power, zoom_frac)

        if n_models == 0:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.4), constrained_layout=True)
            ax.plot(x, gt_power * scale, color="#2ca02c", lw=0.9, label="Ground-Truth")
            ax.set_xlabel("Sampling points")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{app_name} — no predictions loaded", loc="left")
            ax.legend(frameon=False)
            return fig

        ncols = 3 if n_models >= 3 else n_models
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 2.6 * nrows), squeeze=False,
            constrained_layout=True, facecolor="white",
        )

        handles, labels = None, None
        for i, model_name in enumerate(model_names):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            _draw_panel(
                ax, x, gt_power, predictions[model_name],
                zoom_start, zoom_end, ymax, scale, ylabel,
                title=f"({chr(ord('a') + i)}) {model_name}",
            )
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

        for j in range(n_models, nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].set_visible(False)

        if handles:
            fig.legend(handles, labels, loc="upper center", ncols=2,
                       frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(app_name, fontsize=10, y=1.06, x=0.02, ha="left",
                     fontweight="bold")
        return fig

    def make_single_figure(gt_power, pred, model_name, app_name, zoom_frac):
        n_pts = len(gt_power)
        x = np.arange(n_pts)
        ymax = _ymax_with_headroom(gt_power, {"_": pred})
        scale, ylabel = _power_unit(ymax)
        zoom_start, zoom_end = _pick_zoom(gt_power, zoom_frac)

        fig, ax = plt.subplots(
            1, 1, figsize=(9.5, 3.0), constrained_layout=True, facecolor="white",
        )
        _draw_panel(
            ax, x, gt_power, pred, zoom_start, zoom_end, ymax, scale, ylabel,
            title=f"{model_name} — {app_name}",
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.04))
        return fig

    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return mo.image(src=buf.read())

    return fig_to_image, load_predictions, load_test_data, make_figure, make_single_figure


@app.cell
def _(
    data_path_input,
    dataset_dd,
    fig_to_image,
    load_btn,
    load_predictions,
    load_test_data,
    make_figure,
    max_pts_num,
    mo,
    np,
    os,
    result_path_input,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
    zoom_frac_num,
):
    mo.stop(not load_btn.value, mo.md("Configure settings above and click **Load & Plot**."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _f:
        _datasets_cfg = yaml.safe_load(_f)

    _dataset = dataset_dd.value
    _sr = sr_dd.value
    _ws = int(window_size_num.value)
    _seed = int(seed_num.value)
    _zoom_frac = float(zoom_frac_num.value)
    _data_path = data_path_input.value
    _result_path = result_path_input.value
    _appliances_cfg = _datasets_cfg[_dataset]

    _plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(_plots_dir, exist_ok=True)

    _tabs = {}
    for _app_key, _app_cfg in _appliances_cfg.items():
        _app_name = _app_cfg["app"].strip()
        try:
            _data_test = load_test_data(_dataset, _app_cfg, _data_path, _sr, _ws, _seed)
        except Exception as _e:
            _tabs[_app_key] = mo.md(f"**Error loading data:** {_e}")
            continue

        if _data_test is None or len(_data_test) == 0:
            _tabs[_app_key] = mo.md(f"No test data for **{_app_name}**.")
            continue

        _predictions_windowed = load_predictions(
            _result_path, _dataset, _app_key, _sr, _ws, _seed
        )

        _gt_power_full = _data_test[:, 1, 0, :].reshape(-1).astype(float)
        _total_pts = len(_gt_power_full)
        _max_pts = int(max_pts_num.value)
        _gt_power = _gt_power_full[:_max_pts] if _max_pts > 0 else _gt_power_full
        _predictions_full = {
            name: (pred.reshape(-1).astype(float)[:_max_pts] if _max_pts > 0 else pred.reshape(-1).astype(float))
            for name, pred in _predictions_windowed.items()
        }

        _fig = make_figure(_gt_power, _predictions_full, _app_name, _zoom_frac)
        _n_models = len(_predictions_full)
        _pts_note = f"{len(_gt_power):,} / {_total_pts:,} pts (max available)"
        _caption = (
            mo.md(f"{_n_models} model(s) — {_pts_note}")
            if _n_models
            else mo.md(f"No checkpoints found at `{_result_path}` — {_pts_note}")
        )
        if _n_models:
            _svg_path = os.path.join(
                _plots_dir,
                f"{_dataset}_{_app_key}_{_sr}_{_ws}_seed{_seed}_grid.svg",
            )
            _fig.savefig(_svg_path, format="svg", bbox_inches="tight", facecolor="white")
        _tabs[_app_key] = mo.vstack([fig_to_image(_fig), _caption])

    mo.ui.tabs(_tabs)
    return


@app.cell
def _(
    NILMmetrics,
    data_path_input,
    dataset_dd,
    load_btn,
    load_predictions,
    load_test_data,
    max_pts_num,
    mo,
    np,
    os,
    pd,
    result_path_input,
    seed_num,
    sr_dd,
    threshold_num,
    window_size_num,
    yaml,
):
    mo.stop(not load_btn.value, mo.md("Configure settings above and click **Load & Plot** to see metrics."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _mf:
        _metrics_cfg = yaml.safe_load(_mf)

    _m_dataset = dataset_dd.value
    _m_sr = sr_dd.value
    _m_ws = int(window_size_num.value)
    _m_seed = int(seed_num.value)
    _m_data_path = data_path_input.value
    _m_result_path = result_path_input.value
    _m_max_pts = int(max_pts_num.value)
    _m_threshold = float(threshold_num.value)
    _nilm_metrics = NILMmetrics()

    _rows = []
    for _app_key, _app_cfg in _metrics_cfg[_m_dataset].items():
        _app_name = _app_cfg["app"].strip()
        try:
            _data_test = load_test_data(_m_dataset, _app_cfg, _m_data_path, _m_sr, _m_ws, _m_seed)
        except Exception:
            continue
        if _data_test is None or len(_data_test) == 0:
            continue

        _preds = load_predictions(_m_result_path, _m_dataset, _app_key, _m_sr, _m_ws, _m_seed)
        _gt_full = _data_test[:, 1, 0, :].reshape(-1).astype(float)
        _gt = _gt_full[:_m_max_pts] if _m_max_pts > 0 else _gt_full
        _gt_state = (_gt > _m_threshold).astype(int)

        for _model_name, _pred_win in _preds.items():
            _pred_full = _pred_win.reshape(-1).astype(float)
            _pred = _pred_full[:_m_max_pts] if _m_max_pts > 0 else _pred_full
            _pred_state = (_pred > _m_threshold).astype(int)

            _reg = _nilm_metrics(y=_gt, y_hat=_pred)
            _cls = _nilm_metrics(y_state=_gt_state, y_hat_state=_pred_state)

            _rows.append({
                "Appliance": _app_name,
                "Model": _model_name,
                "MAE": _reg["MAE"],
                "SAE": _reg["SAE"],
                "F1": _cls["F1_SCORE"],
            })

    if _rows:
        _metrics_df = pd.DataFrame(_rows).sort_values(["Appliance", "Model"])
        mo.vstack([
            mo.md(f"### Metrics (threshold = {_m_threshold} W)"),
            mo.ui.table(_metrics_df, sortable=True, filterable=True),
        ])
    else:
        mo.md("No predictions found — run models first.")
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save Fridge CSVs")
    plot_csv_btn = mo.ui.run_button(label="Plot Fridge from CSV")
    mo.vstack([
        mo.md("### Fridge CSV Cache"),
        mo.hstack([save_btn, plot_csv_btn]),
    ])
    return plot_csv_btn, save_btn


@app.cell
def _(
    data_path_input,
    dataset_dd,
    load_predictions,
    load_test_data,
    mo,
    np,
    os,
    pd,
    result_path_input,
    save_btn,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
):
    mo.stop(not save_btn.value, mo.md("Click **Save Fridge CSVs** to cache fridge predictions."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _f2:
        _cfg2 = yaml.safe_load(_f2)

    _dataset2 = dataset_dd.value
    _sr2 = sr_dd.value
    _ws2 = int(window_size_num.value)
    _seed2 = int(seed_num.value)
    _data_path2 = data_path_input.value
    _result_path2 = result_path_input.value

    _fridge_key = None
    for _k, _v in _cfg2[_dataset2].items():
        if any(word in _v["app"].lower() for word in ("fridge", "refrigerator")):
            _fridge_key = _k
            _fridge_app_name = _v["app"].strip()
            _fridge_cfg = _v
            break

    if _fridge_key is None:
        _save_status = mo.md("**No fridge/refrigerator appliance found in config for selected dataset.**")
    else:
        _data_test2 = load_test_data(_dataset2, _fridge_cfg, _data_path2, _sr2, _ws2, _seed2)
        _preds2 = load_predictions(_result_path2, _dataset2, _fridge_key, _sr2, _ws2, _seed2)

        _agg2 = _data_test2[:, 0, 0, :].reshape(-1).astype(float)
        _gt2 = _data_test2[:, 1, 0, :].reshape(-1).astype(float)

        _plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(_plots_dir, exist_ok=True)

        _saved = []
        for _model, _pred_win in _preds2.items():
            _pred2 = _pred_win.reshape(-1).astype(float)
            _fname = f"{_dataset2}_{_fridge_key}_{_sr2}_{_ws2}_seed{_seed2}_{_model}.csv"
            _fpath = os.path.join(_plots_dir, _fname)
            pd.DataFrame({
                "aggregate": _agg2,
                "ground_truth": _gt2,
                "prediction": _pred2,
            }).to_csv(_fpath, index=False)
            _saved.append(_fname)

        if _saved:
            _save_status = mo.md("**Saved:**\n" + "\n".join(f"- `plots/{f}`" for f in _saved))
        else:
            _save_status = mo.md("**No prediction checkpoints found.** Run model training first.")

    _save_status
    return


@app.cell
def _(
    dataset_dd,
    fig_to_image,
    glob,
    make_single_figure,
    max_pts_num,
    mo,
    np,
    os,
    pd,
    plot_csv_btn,
    plt,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
    zoom_frac_num,
):
    mo.stop(not plot_csv_btn.value, mo.md("Click **Plot Fridge from CSV** to generate per-model plots from cached CSVs."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _f3:
        _cfg3 = yaml.safe_load(_f3)

    _dataset3 = dataset_dd.value
    _sr3 = sr_dd.value
    _ws3 = int(window_size_num.value)
    _seed3 = int(seed_num.value)
    _zoom3 = float(zoom_frac_num.value)
    _max_pts3 = int(max_pts_num.value)

    _fridge_key3 = None
    for _k3, _v3 in _cfg3[_dataset3].items():
        if any(word in _v3["app"].lower() for word in ("fridge", "refrigerator")):
            _fridge_key3 = _k3
            _fridge_app_name3 = _v3["app"].strip()
            break

    _plots_dir3 = os.path.join(os.path.dirname(__file__), "..", "plots")
    _pattern = os.path.join(
        _plots_dir3,
        f"{_dataset3}_{_fridge_key3}_{_sr3}_{_ws3}_seed{_seed3}_*.csv",
    )
    _csv_files = sorted(glob.glob(_pattern))

    if not _csv_files:
        _csv_output = mo.md(f"No CSVs found matching `plots/{_dataset3}_{_fridge_key3}_{_sr3}_{_ws3}_seed{_seed3}_*.csv`. Click **Save Fridge CSVs** first.")
    else:
        _tabs3 = {}
        for _csv_path in _csv_files:
            _stem = os.path.splitext(os.path.basename(_csv_path))[0]
            _model_name3 = _stem.split(f"_seed{_seed3}_", 1)[-1]
            _df = pd.read_csv(_csv_path)
            _gt3 = _df["ground_truth"].to_numpy()
            _pred3 = _df["prediction"].to_numpy()
            _total_pts3 = len(_gt3)
            if _max_pts3 > 0:
                _gt3 = _gt3[:_max_pts3]
                _pred3 = _pred3[:_max_pts3]

            _fig3 = make_single_figure(_gt3, _pred3, _model_name3, _fridge_app_name3, _zoom3)
            _png_path = os.path.join(_plots_dir3, f"{_stem}.png")
            _svg_path3 = os.path.join(_plots_dir3, f"{_stem}.svg")
            _fig3.savefig(_svg_path3, format="svg", bbox_inches="tight", facecolor="white")
            _buf3 = io.BytesIO()
            _fig3.savefig(_buf3, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(_fig3)
            _img_bytes3 = _buf3.getvalue()
            with open(_png_path, "wb") as _pf:
                _pf.write(_img_bytes3)
            _tabs3[_model_name3] = mo.image(src=_img_bytes3)

        _csv_output = mo.ui.tabs(_tabs3)

    _csv_output
    return


if __name__ == "__main__":
    app.run()
