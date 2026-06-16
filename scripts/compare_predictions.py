import marimo

__generated_with = "0.23.5"
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
    from matplotlib.patches import Rectangle, ConnectionPatch
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

    import plotly.graph_objects as go

    return (
        REFIT_DataBuilder,
        UKDALE_DataBuilder,
        glob,
        go,
        io,
        mo,
        np,
        os,
        pd,
        plt,
        split_train_test_pdl_nilmdataset,
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
    window_size_num = mo.ui.number(value=256, label="Window size")
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
        result_path_input,
        seed_num,
        sr_dd,
        window_size_num,
    )


@app.cell
def _(
    REFIT_DataBuilder,
    UKDALE_DataBuilder,
    mo,
    split_train_test_pdl_nilmdataset,
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

    save_data_btn = mo.ui.run_button(label="Save Data CSVs")
    mo.vstack([mo.md("### Save Data Samples to CSV"), save_data_btn])
    return load_test_data, save_data_btn


@app.cell
def _(
    data_path_input,
    dataset_dd,
    load_test_data,
    mo,
    os,
    pd,
    save_data_btn,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
):
    mo.stop(not save_data_btn.value, mo.md("Click **Save Data CSVs** to cache data samples."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _fds:
        _ds_cfg = yaml.safe_load(_fds)

    _ds = dataset_dd.value
    _ds_sr = sr_dd.value
    _ds_ws = int(window_size_num.value)
    _ds_seed = int(seed_num.value)
    _ds_data_path = data_path_input.value

    _ds_csv_root = os.path.join(os.path.dirname(__file__), "..", "data_csv")
    os.makedirs(_ds_csv_root, exist_ok=True)

    _ds_saved_dirs = []
    for _ds_app_key, _ds_app_cfg in _ds_cfg[_ds].items():
        try:
            _ds_data_test = load_test_data(_ds, _ds_app_cfg, _ds_data_path, _ds_sr, _ds_ws, _ds_seed)
        except Exception:
            continue
        if _ds_data_test is None or len(_ds_data_test) == 0:
            continue
        _ds_app_dir = os.path.join(
            _ds_csv_root, f"{_ds}_{_ds_app_key}_{_ds_sr}_{_ds_ws}_seed{_ds_seed}"
        )
        os.makedirs(_ds_app_dir, exist_ok=True)
        _ds_n = len(_ds_data_test)
        for _ds_i in range(_ds_n):
            pd.DataFrame({
                "aggregate": _ds_data_test[_ds_i, 0, 0, :].astype(float),
                "ground_truth": _ds_data_test[_ds_i, 1, 0, :].astype(float),
            }).to_csv(os.path.join(_ds_app_dir, f"{_ds_i}.csv"), index=False)
        _ds_saved_dirs.append(f"{os.path.basename(_ds_app_dir)}/ ({_ds_n} samples)")

    (
        mo.md("**Saved:**\n" + "\n".join(f"- `data_csv/{d}`" for d in _ds_saved_dirs))
        if _ds_saved_dirs
        else mo.md("**No data saved.**")
    )
    return


@app.cell
def _(glob, np, os, torch):
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

    return (load_predictions,)


@app.cell
def _(mo):
    save_pred_btn = mo.ui.run_button(label="Save Predictions")
    mo.vstack([mo.md("### Save Model Predictions to CSV"), save_pred_btn])
    return (save_pred_btn,)


@app.cell
def _(
    dataset_dd,
    load_predictions,
    mo,
    os,
    pd,
    result_path_input,
    save_pred_btn,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
):
    mo.stop(not save_pred_btn.value, mo.md("Click **Save Predictions** to cache model predictions."))

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _fp:
        _pred_cfg = yaml.safe_load(_fp)

    _pr_ds = dataset_dd.value
    _pr_sr = sr_dd.value
    _pr_ws = int(window_size_num.value)
    _pr_seed = int(seed_num.value)
    _pr_result_path = result_path_input.value

    _pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(_pred_root, exist_ok=True)

    _pr_saved = []
    for _pr_app_key in _pred_cfg[_pr_ds]:
        _pr_preds = load_predictions(_pr_result_path, _pr_ds, _pr_app_key, _pr_sr, _pr_ws, _pr_seed)
        if not _pr_preds:
            continue
        for _pr_model, _pr_arr in _pr_preds.items():
            _pr_model_dir = os.path.join(
                _pred_root, f"{_pr_ds}_{_pr_app_key}_{_pr_sr}_{_pr_ws}_seed{_pr_seed}", _pr_model
            )
            os.makedirs(_pr_model_dir, exist_ok=True)
            for _pr_i in range(len(_pr_arr)):
                pd.DataFrame({"prediction": _pr_arr[_pr_i].astype(float)}).to_csv(
                    os.path.join(_pr_model_dir, f"{_pr_i}.csv"), index=False
                )
            _pr_saved.append(f"{_pr_ds}_{_pr_app_key}_{_pr_sr}_{_pr_ws}_seed{_pr_seed}/{_pr_model}/ ({len(_pr_arr)} samples)")

    (
        mo.md("**Saved:**\n" + "\n".join(f"- `predictions/{s}`" for s in _pr_saved))
        if _pr_saved
        else mo.md("**No predictions found.**")
    )
    return


@app.cell(hide_code=True)
def _(dataset_dd, mo, os, seed_num, sr_dd, window_size_num):
    _sv_ds = dataset_dd.value
    _sv_sr = sr_dd.value
    _sv_ws = int(window_size_num.value)
    _sv_seed = int(seed_num.value)
    _sv_prefix = f"{_sv_ds}_"
    _sv_suffix = f"_{_sv_sr}_{_sv_ws}_seed{_sv_seed}"
    _sv_csv_root = os.path.join(os.path.dirname(__file__), "..", "data_csv")
    _sv_app_keys = []
    if os.path.isdir(_sv_csv_root):
        for _d in sorted(os.listdir(_sv_csv_root)):
            if _d.startswith(_sv_prefix) and _d.endswith(_sv_suffix):
                _sv_app_keys.append(_d[len(_sv_prefix):-len(_sv_suffix)])
    appliance_dd = mo.ui.dropdown(
        _sv_app_keys or ["(none)"], value=_sv_app_keys[0] if _sv_app_keys else "(none)",
        label="Appliance",
    )
    appliance_dd
    return (appliance_dd,)


@app.cell(hide_code=True)
def _(appliance_dd, dataset_dd, mo, os, seed_num, sr_dd, window_size_num):
    _mv_ds = dataset_dd.value
    _mv_sr = sr_dd.value
    _mv_ws = int(window_size_num.value)
    _mv_seed = int(seed_num.value)
    _mv_app_dir = f"{_mv_ds}_{appliance_dd.value}_{_mv_sr}_{_mv_ws}_seed{_mv_seed}"
    _mv_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", _mv_app_dir)
    _mv_models = sorted(os.listdir(_mv_pred_root)) if os.path.isdir(_mv_pred_root) else []
    model_dd = mo.ui.dropdown(
        _mv_models or ["(none)"], value=_mv_models[0] if _mv_models else "(none)",
        label="Model",
    )
    model_dd
    return (model_dd,)


@app.cell(hide_code=True)
def _(mo):
    get_idx, set_idx = mo.state(0)
    return get_idx, set_idx


@app.cell(hide_code=True)
def _(
    appliance_dd,
    dataset_dd,
    get_idx,
    mo,
    os,
    seed_num,
    set_idx,
    sr_dd,
    window_size_num,
):
    _sl_ds = dataset_dd.value
    _sl_sr = sr_dd.value
    _sl_ws = int(window_size_num.value)
    _sl_seed = int(seed_num.value)
    _sl_app_dir = f"{_sl_ds}_{appliance_dd.value}_{_sl_sr}_{_sl_ws}_seed{_sl_seed}"
    _sl_data_dir = os.path.join(os.path.dirname(__file__), "..", "data_csv", _sl_app_dir)
    _sl_n = len([f for f in os.listdir(_sl_data_dir) if f.endswith(".csv")]) if os.path.isdir(_sl_data_dir) else 1
    _sl_max = max(_sl_n - 1, 0)
    prev_btn = mo.ui.button(label="◀", on_click=lambda _: set_idx(max(0, get_idx() - 1)))
    next_btn = mo.ui.button(label="▶", on_click=lambda _: set_idx(min(_sl_max, get_idx() + 1)))
    sample_slider = mo.ui.slider(0, _sl_max, value=get_idx(), on_change=set_idx, label="Sample")
    idx_input = mo.ui.number(start=0, stop=_sl_max, value=get_idx(), on_change=lambda v: set_idx(int(v)), label="Index")
    mo.hstack([prev_btn, sample_slider, next_btn, idx_input, mo.md(f"/ **{_sl_max}**")])
    return


@app.cell(hide_code=True)
def _(
    appliance_dd,
    dataset_dd,
    get_idx,
    io,
    mo,
    model_dd,
    np,
    os,
    pd,
    plt,
    seed_num,
    sr_dd,
    window_size_num,
):
    _pl_ds = dataset_dd.value
    _pl_sr = sr_dd.value
    _pl_ws = int(window_size_num.value)
    _pl_seed = int(seed_num.value)
    _pl_app_dir = f"{_pl_ds}_{appliance_dd.value}_{_pl_sr}_{_pl_ws}_seed{_pl_seed}"
    _pl_i = get_idx()

    _pl_data_path = os.path.join(os.path.dirname(__file__), "..", "data_csv", _pl_app_dir, f"{_pl_i}.csv")
    _pl_pred_path = os.path.join(os.path.dirname(__file__), "..", "predictions", _pl_app_dir, model_dd.value, f"{_pl_i}.csv")

    if not os.path.isfile(_pl_data_path):
        _pl_out = mo.md(f"Data CSV not found: `{_pl_data_path}`. Run **Save Data CSVs** first.")
    elif not os.path.isfile(_pl_pred_path):
        _pl_out = mo.md(f"Prediction CSV not found: `{_pl_pred_path}`. Run **Save Predictions** first.")
    else:
        _pl_data = pd.read_csv(_pl_data_path)
        _pl_pred = pd.read_csv(_pl_pred_path)
        _pl_x = np.arange(_pl_ws)

        _pl_fig, _pl_ax = plt.subplots(1, 1, figsize=(9.5, 3.0), constrained_layout=True, facecolor="white")
        _pl_ax.plot(_pl_x, _pl_data["aggregate"].values, color="#1f77b4", lw=0.9, label="Aggregate")
        _pl_ax.plot(_pl_x, _pl_data["ground_truth"].values, color="#2ca02c", lw=0.9, label="Ground-Truth")
        _pl_ax.plot(_pl_x, _pl_pred["prediction"].values, color="#ff7f0e", lw=0.9, label="Prediction")
        _pl_ax.set_xlim(0, _pl_ws - 1)
        _pl_ax.set_xlabel("Timestep")
        _pl_ax.set_ylabel("Power (W)")
        _pl_ax.set_title(f"{appliance_dd.value} — {model_dd.value} — sample {_pl_i}", loc="left")
        _pl_ax.legend(frameon=False, ncols=3)
        _pl_buf = io.BytesIO()
        _pl_fig.savefig(_pl_buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(_pl_fig)
        _pl_buf.seek(0)
        _pl_out = mo.image(src=_pl_buf.read())
    _pl_out
    return


@app.cell(hide_code=True)
def _(
    appliance_dd,
    dataset_dd,
    go,
    mo,
    model_dd,
    np,
    os,
    pd,
    seed_num,
    sr_dd,
    window_size_num,
):
    _fs_ds = dataset_dd.value
    _fs_sr = sr_dd.value
    _fs_ws = int(window_size_num.value)
    _fs_seed = int(seed_num.value)
    _fs_app_dir = f"{_fs_ds}_{appliance_dd.value}_{_fs_sr}_{_fs_ws}_seed{_fs_seed}"
    _fs_data_root = os.path.join(os.path.dirname(__file__), "..", "data_csv", _fs_app_dir)
    _fs_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", _fs_app_dir, model_dd.value)

    mo.stop(not os.path.isdir(_fs_data_root), mo.md("No data CSVs found. Run **Save Data CSVs** first."))
    mo.stop(not os.path.isdir(_fs_pred_root), mo.md("No prediction CSVs found. Run **Save Predictions** first."))

    _fs_indices = sorted([int(f[:-4]) for f in os.listdir(_fs_data_root) if f.endswith(".csv")])
    _fs_agg_list, _fs_gt_list, _fs_pred_list = [], [], []
    for _fs_i in _fs_indices:
        _df_d = pd.read_csv(os.path.join(_fs_data_root, f"{_fs_i}.csv"))
        _fs_agg_list.append(_df_d["aggregate"].values)
        _fs_gt_list.append(_df_d["ground_truth"].values)
        _fp = os.path.join(_fs_pred_root, f"{_fs_i}.csv")
        _fs_pred_list.append(pd.read_csv(_fp)["prediction"].values if os.path.isfile(_fp) else np.full(_fs_ws, np.nan))

    fs_agg_full = np.concatenate(_fs_agg_list)
    fs_gt_full = np.concatenate(_fs_gt_list)
    fs_pred_full = np.concatenate(_fs_pred_list)
    _fs_total = len(fs_agg_full)

    _fs_step = max(1, _fs_total // 8000)
    _fs_x_ov = np.arange(_fs_total)[::_fs_step]
    _fs_fig_ov = go.Figure()
    _fs_fig_ov.add_trace(go.Scatter(x=_fs_x_ov, y=fs_agg_full[::_fs_step], name="Aggregate", line=dict(color="#1f77b4", width=1), opacity=0.7))
    _fs_fig_ov.add_trace(go.Scatter(x=_fs_x_ov, y=fs_gt_full[::_fs_step], name="Ground-Truth", line=dict(color="#2ca02c", width=1)))
    _fs_fig_ov.add_trace(go.Scatter(x=_fs_x_ov, y=fs_pred_full[::_fs_step], name="Prediction", line=dict(color="#ff7f0e", width=1)))
    _fs_fig_ov.update_layout(
        title=dict(text=f"{appliance_dd.value} — {model_dd.value} — overview ({_fs_total} pts)", x=0, xanchor="left"),
        xaxis_title="Timestep", yaxis_title="Power (W)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified", template="simple_white",
    )

    fs_zoom_sl = mo.ui.range_slider(0, _fs_total - 1, value=[0, min(_fs_total - 1, _fs_ws * 10)], step=1, label="Zoom window")
    mo.vstack([mo.ui.plotly(_fs_fig_ov), fs_zoom_sl])
    return fs_agg_full, fs_gt_full, fs_pred_full, fs_zoom_sl


@app.cell(hide_code=True)
def _(
    appliance_dd,
    fs_agg_full,
    fs_gt_full,
    fs_pred_full,
    fs_zoom_sl,
    go,
    mo,
    model_dd,
    np,
):
    _x0, _x1 = int(fs_zoom_sl.value[0]), int(fs_zoom_sl.value[1])
    _det_agg = fs_agg_full[_x0:_x1 + 1]
    _det_gt = fs_gt_full[_x0:_x1 + 1]
    _det_pred = fs_pred_full[_x0:_x1 + 1]
    _det_x = np.arange(_x0, _x1 + 1)
    _det_step = max(1, len(_det_x) // 5000)
    _det_fig = go.Figure()
    _det_fig.add_trace(go.Scatter(x=_det_x[::_det_step], y=_det_agg[::_det_step], name="Aggregate", line=dict(color="#1f77b4", width=1), opacity=0.7))
    _det_fig.add_trace(go.Scatter(x=_det_x[::_det_step], y=_det_gt[::_det_step], name="Ground-Truth", line=dict(color="#2ca02c", width=1)))
    _det_fig.add_trace(go.Scatter(x=_det_x[::_det_step], y=_det_pred[::_det_step], name="Prediction", line=dict(color="#ff7f0e", width=1)))
    _det_fig.update_layout(
        title=dict(text=f"{appliance_dd.value} — {model_dd.value} — detail [{_x0}:{_x1}]", x=0, xanchor="left"),
        xaxis_title="Timestep", yaxis_title="Power (W)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified", template="simple_white",
    )
    mo.ui.plotly(_det_fig)
    return


@app.cell(hide_code=True)
def _(mo, os):
    _viz_csv_root = os.path.join(os.path.dirname(__file__), "..", "data_csv")
    _viz_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions")

    _viz_app_dirs = sorted(os.listdir(_viz_csv_root)) if os.path.isdir(_viz_csv_root) else []

    viz_dataset_dd = mo.ui.dropdown(
        _viz_app_dirs or ["(none)"],
        value=_viz_app_dirs[0] if _viz_app_dirs else "(none)",
        label="Dataset/Appliance/SR/Window/Seed",
    )
    viz_seed_num = mo.ui.number(value=0, label="Seed")
    viz_sr_dd = mo.ui.dropdown(["10s", "1min", "10min", "30min"], value="10s", label="Sampling rate")
    viz_ws_num = mo.ui.number(value=256, label="Window size")
    viz_start_num = mo.ui.number(value=0, label="Start sample point")
    viz_end_num = mo.ui.number(value=256, label="End sample point")

    mo.vstack([
        viz_dataset_dd,
        mo.hstack([viz_seed_num, viz_sr_dd, viz_ws_num, viz_start_num, viz_end_num]),
    ])
    return viz_dataset_dd, viz_end_num, viz_start_num, viz_ws_num


@app.cell(hide_code=True)
def _(mo, os, viz_dataset_dd):
    _viz_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", viz_dataset_dd.value)
    _viz_models = sorted(os.listdir(_viz_pred_root)) if os.path.isdir(_viz_pred_root) else []

    viz_model_dd = mo.ui.dropdown(
        _viz_models or ["(none)"],
        value=_viz_models[0] if _viz_models else "(none)",
        label="Model",
    )
    viz_model_dd
    return (viz_model_dd,)


@app.cell(hide_code=True)
def _(mo, np, os, pd, viz_dataset_dd, viz_model_dd, viz_ws_num):
    _vz_data_root = os.path.join(os.path.dirname(__file__), "..", "data_csv", viz_dataset_dd.value)
    _vz_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", viz_dataset_dd.value, viz_model_dd.value)

    mo.stop(not os.path.isdir(_vz_data_root), mo.md("No data CSVs found. Run **Save Data CSVs** first."))
    mo.stop(not os.path.isdir(_vz_pred_root), mo.md("No prediction CSVs found. Run **Save Predictions** first."))

    _vz_ws = int(viz_ws_num.value)
    _vz_indices = sorted([int(f[:-4]) for f in os.listdir(_vz_data_root) if f.endswith(".csv")])

    _vz_agg, _vz_gt, _vz_pred = [], [], []
    for _vz_i in _vz_indices:
        _vz_df = pd.read_csv(os.path.join(_vz_data_root, f"{_vz_i}.csv"))
        _vz_agg.append(_vz_df["aggregate"].values)
        _vz_gt.append(_vz_df["ground_truth"].values)
        _vz_pf = os.path.join(_vz_pred_root, f"{_vz_i}.csv")
        _vz_pred.append(pd.read_csv(_vz_pf)["prediction"].values if os.path.isfile(_vz_pf) else np.full(_vz_ws, np.nan))

    viz_agg_full = np.concatenate(_vz_agg)
    viz_gt_full = np.concatenate(_vz_gt)
    viz_pred_full = np.concatenate(_vz_pred)
    return viz_agg_full, viz_gt_full, viz_pred_full


@app.cell(hide_code=True)
def _(
    io,
    mo,
    np,
    plt,
    viz_agg_full,
    viz_end_num,
    viz_gt_full,
    viz_pred_full,
    viz_start_num,
):
    _vp_start = int(viz_start_num.value)
    _vp_end = int(viz_end_num.value)

    _vp_agg = viz_agg_full[_vp_start:_vp_end]
    _vp_gt = viz_gt_full[_vp_start:_vp_end]
    _vp_pred = viz_pred_full[_vp_start:_vp_end]
    _vp_x = np.arange(len(_vp_agg))

    _vp_fig, _vp_ax = plt.subplots(1, 1, figsize=(9.5, 3.0), constrained_layout=True)
    _vp_ax.plot(_vp_x, _vp_agg, color="#1f77b4", lw=0.9)
    _vp_ax.plot(_vp_x, _vp_gt, color="#2ca02c", lw=0.9)
    _vp_ax.plot(_vp_x, _vp_pred, color="#ff7f0e", lw=0.9)
    _vp_ax.set_xlim(0, len(_vp_x) - 1)
    _vp_ax.set_xlabel("")
    _vp_ax.set_ylabel("")

    viz_buf = io.BytesIO()
    _vp_fig.savefig(viz_buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(_vp_fig)
    viz_buf.seek(0)
    viz_png_bytes = viz_buf.getvalue()

    mo.vstack([
        mo.image(src=viz_png_bytes),
        mo.download(data=viz_png_bytes, filename="prediction_plot.png", label="Download PNG"),
    ])
    return


if __name__ == "__main__":
    app.run()
