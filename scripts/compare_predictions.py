import marimo

__generated_with = "0.23.7"
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
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
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
        REDD_DataBuilder,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
    )
    from src.helpers.metrics import NILMmetrics

    import plotly.graph_objects as go

    return (
        REDD_DataBuilder,
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
def _():
    # Checkpoints are spread across directories: the TCN arms live in result-tcn-ablation,
    # the UK-DALE baselines in result-synagg, the REDD baselines in 2207-results/result.
    # Searched in this order; the first hit for a (dataset, appliance, model) wins, which
    # matches the dedup in scripts/make_table.py.
    RESULT_ROOTS = [
        "result-tcn-ablation",
        "result-synagg",
        "2207-results/result",
    ]

    #: The Proposed model is arm 3: the in-repo, 3-epoch, one-model-per-appliance arm with
    #: KL features and synthetic augmentation from repo-extracted activations.
    PROPOSED_KEY = "TCN_KL_aug"

    #: Display names. Anything not listed keeps its raw key.
    MODEL_LABELS = {
        PROPOSED_KEY: "Proposed (TCN + KL + Syn)",
        "NILMFormer": "NILMFormer",
        "BERT4NILM": "BERT4NILM",
        "BiLSTM": "BiLSTM",
        "BiGRU": "BiGRU",
    }

    #: Plot order for the multi-model figure; Proposed first.
    PLOT_MODELS = [PROPOSED_KEY, "NILMFormer", "BERT4NILM", "BiLSTM", "BiGRU"]

    #: REDD's washer/dryer is keyed WasherDryer by the TCN arms and WashingMachine by the
    #: baselines; REDD_METER_MAP resolves both to the same meter, so look under both.
    REDD_APP_ALIASES = {
        "WasherDryer": ["WasherDryer", "WashingMachine"],
        "WashingMachine": ["WashingMachine", "WasherDryer"],
    }

    DATASETS = ["UKDALE", "REDD"]
    return DATASETS, MODEL_LABELS, PROPOSED_KEY, REDD_APP_ALIASES, RESULT_ROOTS


@app.cell
def _(mo):
    result_path_input = mo.ui.text(value="result/", label="Result path", full_width=True)
    data_path_input = mo.ui.text(value="data/", label="Data path", full_width=True)
    dataset_dd = mo.ui.dropdown(["UKDALE", "REFIT", "REDD"], value="UKDALE", label="Dataset")
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
    return data_path_input, dataset_dd, seed_num, sr_dd, window_size_num


@app.cell
def _(
    REDD_DataBuilder,
    REFIT_DataBuilder,
    UKDALE_DataBuilder,
    mo,
    split_train_test_pdl_nilmdataset,
):
    def load_test_data(dataset, app_cfg, data_path, sr, ws, seed, synth_aggregate_apps=None):
        """(windows, st_date) for the test split.

        st_date is returned because it is the ONLY key that aligns two loads of the same
        house: whether `synth_aggregate_apps` is set changes which windows survive the
        NaN check, so window i of one load is not window i of the other.
        """
        if dataset == "UKDALE":
            builder = UKDALE_DataBuilder(
                data_path=f"{data_path}UKDALE/",
                mask_app=app_cfg["app"],
                sampling_rate=sr,
                window_size=ws,
                synth_aggregate_apps=synth_aggregate_apps,
            )
            data_test, st_test = builder.get_nilm_dataset(
                house_indicies=app_cfg["ind_house_test"]
            )
        elif dataset == "REDD":
            # REDD holds out a whole house, so the test split is that house as-is --
            # no 80/20 carve-out, unlike UK-DALE and REFIT.
            builder = REDD_DataBuilder(
                data_path=f"{data_path}REDD/redd.h5",
                mask_app=app_cfg["app"],
                sampling_rate=sr,
                window_size=ws,
                synth_aggregate_apps=synth_aggregate_apps,
            )
            data_test, st_test = builder.get_nilm_dataset(
                house_indicies=list(app_cfg["ind_house_test"])
            )
        else:
            builder = REFIT_DataBuilder(
                data_path=f"{data_path}REFIT/RAW_DATA_CLEAN/",
                mask_app=app_cfg["app"].strip(),
                sampling_rate=sr,
                window_size=ws,
                synth_aggregate_apps=synth_aggregate_apps,
            )
            data_all, st_all = builder.get_nilm_dataset(
                house_indicies=app_cfg["house_with_app_i"]
            )
            _, _, data_test, st_test = split_train_test_pdl_nilmdataset(
                data_all, st_all, nb_house_test=2, seed=seed
            )
        return data_test, st_test

    save_data_btn = mo.ui.run_button(label="Save Data CSVs")
    mo.vstack([mo.md("### Save Data Samples to CSV"), save_data_btn])
    return load_test_data, save_data_btn


@app.cell
def _(
    DATASETS,
    data_path_input,
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

    _ds_sr = sr_dd.value
    _ds_ws = int(window_size_num.value)
    _ds_seed = int(seed_num.value)
    _ds_data_path = data_path_input.value

    _ds_csv_root = os.path.join(os.path.dirname(__file__), "..", "data_csv")
    os.makedirs(_ds_csv_root, exist_ok=True)

    _ds_saved_dirs = []
    _ds_skipped = []
    # Both datasets in one pass -- the figures compare across them, and exporting only the
    # dropdown's selection left the other half silently stale.
    for _ds in DATASETS:
        if _ds not in _ds_cfg:
            continue
        _ds_synth_apps = _ds_cfg[_ds].get("synth_aggregate_apps")
        for _ds_app_key, _ds_app_cfg in _ds_cfg[_ds].items():
            if _ds_app_key == "synth_aggregate_apps":
                continue
            try:
                # THE EXPORT IS DRIVEN BY THE RUN THE MODELS ACTUALLY SAW. When
                # synth_aggregate_apps is set the builder puts the synthetic aggregate in
                # channel 0 INSTEAD of the real mains (preprocessing.py:_get_stems), and
                # the NaN check that drops windows then runs against it. The real mains
                # carries NaNs the synthetic sum does not, so the two loads yield
                # DIFFERENT window counts (UK-DALE kettle: 7914 synthetic vs 6711 real).
                # Exporting the real load while the checkpoints hold 7914 synthetic
                # windows silently misaligned every figure.
                _ds_primary, _ds_primary_st = load_test_data(
                    _ds, _ds_app_cfg, _ds_data_path, _ds_sr, _ds_ws, _ds_seed,
                    synth_aggregate_apps=_ds_synth_apps,
                )
            except Exception as _ds_err:
                _ds_skipped.append(f"{_ds}/{_ds_app_key}: {type(_ds_err).__name__}: {_ds_err}")
                continue
            if _ds_primary is None or len(_ds_primary) == 0:
                _ds_skipped.append(f"{_ds}/{_ds_app_key}: empty test split")
                continue

            # The real mains is kept for reference, matched window-by-window on start
            # date. Windows with no counterpart get NaN rather than a wrong neighbour.
            _ds_real_by_date = {}
            if _ds_synth_apps:
                try:
                    _ds_real, _ds_real_st = load_test_data(
                        _ds, _ds_app_cfg, _ds_data_path, _ds_sr, _ds_ws, _ds_seed
                    )
                    _ds_real_by_date = {
                        _d: _k for _k, _d in enumerate(_ds_real_st["start_date"].values)
                    }
                except Exception:
                    _ds_real = None

            _ds_app_dir = os.path.join(
                _ds_csv_root, f"{_ds}_{_ds_app_key}_{_ds_sr}_{_ds_ws}_seed{_ds_seed}"
            )
            os.makedirs(_ds_app_dir, exist_ok=True)
            # Drop stale windows from an earlier export: a shorter run would otherwise
            # leave the tail behind and the directory would mix two window sets.
            for _ds_old in os.listdir(_ds_app_dir):
                if _ds_old.endswith(".csv"):
                    os.remove(os.path.join(_ds_app_dir, _ds_old))
            _ds_n = len(_ds_primary)
            _ds_dates = list(_ds_primary_st["start_date"].values)
            _ds_matched = 0
            for _ds_i in range(_ds_n):
                _ds_row = {
                    "aggregate": _ds_primary[_ds_i, 0, 0, :].astype(float),
                    "ground_truth": _ds_primary[_ds_i, 1, 0, :].astype(float),
                    "ground_truth_status": _ds_primary[_ds_i, 1, 1, :].astype(float),
                }
                if _ds_synth_apps:
                    # Channel 0 IS the synthetic aggregate in this run; name it so.
                    _ds_row["synthetic_aggregate"] = _ds_row.pop("aggregate")
                    _ds_j = _ds_real_by_date.get(_ds_dates[_ds_i])
                    if _ds_j is not None:
                        _ds_row["aggregate"] = _ds_real[_ds_j, 0, 0, :].astype(float)
                        _ds_matched += 1
                pd.DataFrame(_ds_row).to_csv(
                    os.path.join(_ds_app_dir, f"{_ds_i}.csv"), index=False
                )
            # Window start dates, so a section can be checked for time continuity: the
            # per-window CSVs carry no timestamps and dropped windows leave real gaps.
            pd.DataFrame({"window": range(_ds_n), "start_date": _ds_dates}).to_csv(
                os.path.join(_ds_app_dir, "_index.csv"), index=False
            )
            _ds_saved_dirs.append(
                f"{os.path.basename(_ds_app_dir)}/ ({_ds_n} samples"
                + (f", {_ds_matched} with real mains)" if _ds_synth_apps else ")")
            )

    _ds_msg = (
        "**Saved:**\n" + "\n".join(f"- `data_csv/{d}`" for d in _ds_saved_dirs)
        if _ds_saved_dirs
        else "**No data saved.**"
    )
    if _ds_skipped:
        _ds_msg += "\n\n**Skipped:**\n" + "\n".join(f"- {x}" for x in _ds_skipped)
    mo.md(_ds_msg)
    return


@app.cell
def _(REDD_APP_ALIASES, RESULT_ROOTS, glob, np, os, torch):
    def load_predictions(dataset, appliance_key, sr, ws, seed, roots=None):
        """{model: (n_windows, ws)} gathered across every result directory.

        One appliance's models are split across directories (TCN arms in
        result-tcn-ablation, baselines in result-synagg / 2207-results/result), so all are
        searched. Roots earlier in the list win, matching make_table.py's dedup, so a
        model present in two directories is taken from the preferred one.
        """
        roots = roots if roots is not None else RESULT_ROOTS
        # REDD keys its washer/dryer two ways for the same meter; try both.
        keys = (
            REDD_APP_ALIASES.get(appliance_key, [appliance_key])
            if dataset == "REDD"
            else [appliance_key]
        )
        predictions = {}
        for root in roots:
            for key in keys:
                ckpt_dir = os.path.join(root, f"{dataset}_{key}_{sr}", str(ws))
                if not os.path.isdir(ckpt_dir):
                    continue
                for pt_file in sorted(glob.glob(os.path.join(ckpt_dir, f"*_{seed}.pt"))):
                    model_name = os.path.basename(pt_file).replace(f"_{seed}.pt", "")
                    if model_name in predictions:
                        continue  # an earlier root already supplied this model
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
    DATASETS,
    load_predictions,
    mo,
    os,
    pd,
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

    _pr_sr = sr_dd.value
    _pr_ws = int(window_size_num.value)
    _pr_seed = int(seed_num.value)

    _pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(_pred_root, exist_ok=True)

    _pr_saved = []
    for _pr_ds in DATASETS:
        if _pr_ds not in _pred_cfg:
            continue
        for _pr_app_key in _pred_cfg[_pr_ds]:
            if _pr_app_key == "synth_aggregate_apps":
                continue  # a config key, not an appliance
            _pr_preds = load_predictions(_pr_ds, _pr_app_key, _pr_sr, _pr_ws, _pr_seed)
            if not _pr_preds:
                continue
            for _pr_model, _pr_arr in _pr_preds.items():
                _pr_model_dir = os.path.join(
                    _pred_root,
                    f"{_pr_ds}_{_pr_app_key}_{_pr_sr}_{_pr_ws}_seed{_pr_seed}",
                    _pr_model,
                )
                os.makedirs(_pr_model_dir, exist_ok=True)
                for _pr_i in range(len(_pr_arr)):
                    pd.DataFrame({"prediction": _pr_arr[_pr_i].astype(float)}).to_csv(
                        os.path.join(_pr_model_dir, f"{_pr_i}.csv"), index=False
                    )
                _pr_saved.append(
                    f"{_pr_ds}_{_pr_app_key}_{_pr_sr}_{_pr_ws}_seed{_pr_seed}/"
                    f"{_pr_model}/ ({len(_pr_arr)} samples)"
                )

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
def _(
    MODEL_LABELS,
    PROPOSED_KEY,
    appliance_dd,
    dataset_dd,
    mo,
    os,
    seed_num,
    sr_dd,
    window_size_num,
):
    _mv_ds = dataset_dd.value
    _mv_sr = sr_dd.value
    _mv_ws = int(window_size_num.value)
    _mv_seed = int(seed_num.value)
    _mv_app_dir = f"{_mv_ds}_{appliance_dd.value}_{_mv_sr}_{_mv_ws}_seed{_mv_seed}"
    _mv_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", _mv_app_dir)
    _mv_models = sorted(os.listdir(_mv_pred_root)) if os.path.isdir(_mv_pred_root) else []
    # Proposed (arm 3) first when present, so it is the default selection.
    _mv_models = (
        [m for m in _mv_models if m == PROPOSED_KEY]
        + [m for m in _mv_models if m != PROPOSED_KEY]
    )
    model_dd = mo.ui.dropdown(
        {MODEL_LABELS.get(m, m): m for m in _mv_models} or {"(none)": "(none)"},
        value=MODEL_LABELS.get(_mv_models[0], _mv_models[0]) if _mv_models else "(none)",
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
    _sl_n = len([f for f in os.listdir(_sl_data_dir)
                 if f.endswith(".csv") and f[0].isdigit()]) if os.path.isdir(_sl_data_dir) else 1
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
        # Synthetic aggregate for EVERY model: it is what they were trained and scored
        # on. Falls back to the real mains only if the column is absent, and says so.
        _pl_has_syn = "synthetic_aggregate" in _pl_data.columns
        _pl_agg_col = "synthetic_aggregate" if _pl_has_syn else "aggregate"
        _pl_agg_label = "Synthetic Aggregate" if _pl_has_syn else "Aggregate (real mains)"

        _pl_fig, _pl_ax = plt.subplots(1, 1, figsize=(9.5, 3.0), constrained_layout=True, facecolor="white")
        _pl_ax.plot(_pl_x, _pl_data["ground_truth"].values, color="#2ca02c", lw=0.9, label="Ground-Truth")
        _pl_ax.plot(_pl_x, _pl_pred["prediction"].values, color="#ff7f0e", lw=0.9, label="Prediction")
        _pl_ax.set_xlim(0, _pl_ws - 1)
        _pl_ax.set_xlabel("Timestep")
        _pl_ax.set_ylabel("Power (W)")
        _pl_ax.set_title(f"{appliance_dd.value} — {model_dd.value} — sample {_pl_i}", loc="left")
        _pl_ax.ticklabel_format(style="plain", useOffset=False, axis="both")
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

    # f[0].isdigit() skips _index.csv, which is window metadata rather than a window.
    _fs_indices = sorted([int(f[:-4]) for f in os.listdir(_fs_data_root)
                          if f.endswith(".csv") and f[0].isdigit()])

    # A count mismatch means the data CSVs and the checkpoint were built from different
    # runs (typically one with and one without the synthetic aggregate), in which case
    # window i of one is NOT window i of the other and every plot is silently wrong.
    _fs_npred = len([f for f in os.listdir(_fs_pred_root) if f.endswith(".csv")])
    mo.stop(
        _fs_npred != len(_fs_indices),
        mo.md(
            f"**Misaligned:** {len(_fs_indices)} data windows vs "
            f"{_fs_npred} prediction windows. Re-run **Save Data CSVs** "
            "(it must be driven by the same synthetic-aggregate run the model was "
            "evaluated on)."
        ),
    )
    _fs_agg_list, _fs_gt_list, _fs_pred_list = [], [], []
    for _fs_i in _fs_indices:
        _df_d = pd.read_csv(os.path.join(_fs_data_root, f"{_fs_i}.csv"))
        _fs_agg_col = ("synthetic_aggregate" if "synthetic_aggregate" in _df_d.columns
                       else "aggregate")
        _fs_agg_list.append(_df_d[_fs_agg_col].values)
        _fs_gt_list.append(_df_d["ground_truth"].values)
        _fp = os.path.join(_fs_pred_root, f"{_fs_i}.csv")
        _fs_pred_list.append(pd.read_csv(_fp)["prediction"].values if os.path.isfile(_fp) else np.full(_fs_ws, np.nan))

    fs_agg_full = np.concatenate(_fs_agg_list)
    fs_gt_full = np.concatenate(_fs_gt_list)
    fs_pred_full = np.concatenate(_fs_pred_list)
    _fs_total = len(fs_agg_full)
    _fs_agg_label = (
        "Synthetic Aggregate" if _fs_agg_col == "synthetic_aggregate"
        else "Aggregate (real mains)"
    )

    _fs_step = max(1, _fs_total // 8000)
    _fs_x_ov = np.arange(_fs_total)[::_fs_step]
    _fs_fig_ov = go.Figure()
    _fs_fig_ov.add_trace(go.Scatter(x=_fs_x_ov, y=fs_gt_full[::_fs_step], name="Ground-Truth", line=dict(color="#2ca02c", width=1)))
    _fs_fig_ov.add_trace(go.Scatter(x=_fs_x_ov, y=fs_pred_full[::_fs_step], name="Prediction", line=dict(color="#ff7f0e", width=1)))
    _fs_fig_ov.update_layout(
        title=dict(text=f"{appliance_dd.value} — {model_dd.value} — overview ({_fs_total} pts)", x=0, xanchor="left"),
        xaxis_title="Timestep", yaxis_title="Power (W)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified", template="simple_white",
        font=dict(size=14),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True,
                   title_font=dict(size=16), tickfont=dict(size=13)),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True,
                   title_font=dict(size=16), tickfont=dict(size=13)),
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
    # Relative to the start of the zoom, so the axis reads 0..N rather than carrying the
    # absolute index; the absolute range stays in the title.
    _det_x = np.arange(len(_det_gt))
    _det_step = max(1, len(_det_x) // 5000)
    _det_fig = go.Figure()
    _det_fig.add_trace(go.Scatter(x=_det_x[::_det_step], y=_det_gt[::_det_step], name="Ground-Truth", line=dict(color="#2ca02c", width=1)))
    _det_fig.add_trace(go.Scatter(x=_det_x[::_det_step], y=_det_pred[::_det_step], name="Prediction", line=dict(color="#ff7f0e", width=1)))
    _det_fig.update_layout(
        title=dict(text=f"{appliance_dd.value} — {model_dd.value} — detail [{_x0}:{_x1}]", x=0, xanchor="left"),
        xaxis_title=f"Sampling Points (from {_x0})", yaxis_title="Power (W)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified", template="simple_white",
        font=dict(size=14),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True,
                   title_font=dict(size=16), tickfont=dict(size=13)),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True,
                   title_font=dict(size=16), tickfont=dict(size=13)),
    )
    mo.ui.plotly(_det_fig)
    return


@app.cell(hide_code=True)
def _(MODEL_LABELS, PROPOSED_KEY, mo, os, viz_dataset_dd):
    _viz_pred_root = os.path.join(os.path.dirname(__file__), "..", "predictions", viz_dataset_dd.value)
    _viz_models = sorted(os.listdir(_viz_pred_root)) if os.path.isdir(_viz_pred_root) else []
    _viz_models = (
        [m for m in _viz_models if m == PROPOSED_KEY]
        + [m for m in _viz_models if m != PROPOSED_KEY]
    )
    viz_model_dd = mo.ui.dropdown(
        {MODEL_LABELS.get(m, m): m for m in _viz_models} or {"(none)": "(none)"},
        value=MODEL_LABELS.get(_viz_models[0], _viz_models[0]) if _viz_models else "(none)",
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
    _vz_indices = sorted([int(f[:-4]) for f in os.listdir(_vz_data_root)
                          if f.endswith(".csv") and f[0].isdigit()])

    # A count mismatch means the data CSVs and the checkpoint were built from different
    # runs (typically one with and one without the synthetic aggregate), in which case
    # window i of one is NOT window i of the other and every plot is silently wrong.
    _vz_npred = len([f for f in os.listdir(_vz_pred_root) if f.endswith(".csv")])
    mo.stop(
        _vz_npred != len(_vz_indices),
        mo.md(
            f"**Misaligned:** {len(_vz_indices)} data windows vs "
            f"{_vz_npred} prediction windows. Re-run **Save Data CSVs** "
            "(it must be driven by the same synthetic-aggregate run the model was "
            "evaluated on)."
        ),
    )

    _vz_agg, _vz_gt, _vz_pred = [], [], []
    for _vz_i in _vz_indices:
        _vz_df = pd.read_csv(os.path.join(_vz_data_root, f"{_vz_i}.csv"))
        _vz_agg_col = ("synthetic_aggregate" if "synthetic_aggregate" in _vz_df.columns
                       else "aggregate")
        _vz_agg.append(_vz_df[_vz_agg_col].values)
        _vz_gt.append(_vz_df["ground_truth"].values)
        _vz_pf = os.path.join(_vz_pred_root, f"{_vz_i}.csv")
        _vz_pred.append(pd.read_csv(_vz_pf)["prediction"].values if os.path.isfile(_vz_pf) else np.full(_vz_ws, np.nan))

    viz_agg_full = np.concatenate(_vz_agg)
    viz_gt_full = np.concatenate(_vz_gt)
    viz_pred_full = np.concatenate(_vz_pred)
    return viz_agg_full, viz_gt_full, viz_pred_full


@app.cell(hide_code=True)
def _(mo, os):
    # Dataset/window controls only. The start/end inputs live in their own cell further
    # down, bound to persistent state, so that re-running anything up here cannot reset
    # a section you just chose.
    _viz_csv_root = os.path.join(os.path.dirname(__file__), "..", "data_csv")

    _viz_app_dirs = sorted(os.listdir(_viz_csv_root)) if os.path.isdir(_viz_csv_root) else []

    viz_dataset_dd = mo.ui.dropdown(
        _viz_app_dirs or ["(none)"],
        value=_viz_app_dirs[0] if _viz_app_dirs else "(none)",
        label="Dataset/Appliance/SR/Window/Seed",
    )
    viz_seed_num = mo.ui.number(value=0, label="Seed")
    viz_sr_dd = mo.ui.dropdown(["10s", "1min", "10min", "30min"], value="10s", label="Sampling rate")
    viz_ws_num = mo.ui.number(value=256, label="Window size")

    mo.vstack([
        viz_dataset_dd,
        mo.hstack([viz_seed_num, viz_sr_dd, viz_ws_num]),
    ])
    return viz_dataset_dd, viz_ws_num


@app.cell(hide_code=True)
def _(mo):
    # The chosen section lives in state, not in the widgets. A marimo cell that re-runs
    # rebuilds every widget it defines at its DEFAULT value, which is why typing a start
    # and end by hand kept getting thrown away; state survives the re-run.
    get_sec, set_sec = mo.state((0, 1000))
    return get_sec, set_sec


@app.cell(hide_code=True)
def _(mo):
    sec_len_num = mo.ui.number(value=1000, label="Section length (samples)")
    sec_topk_num = mo.ui.number(value=8, label="How many candidates to show")
    sec_btn = mo.ui.run_button(label="Suggest sections")
    mo.vstack([
        mo.md("### Find a section worth plotting"),
        mo.md(
            "Ranks every window of the requested length and reports the best candidates. "
            "A section scores well when it contains one to three COMPLETE activations "
            "(both edges inside the section), is neither almost always on nor almost "
            "always off, and carries competing load in the aggregate so disaggregation "
            "is non-trivial. Pick one below and click **Use this section** -- the final "
            "figure follows it, so nothing has to be copied by hand."
        ),
        mo.hstack([sec_len_num, sec_topk_num, sec_btn]),
    ])
    return sec_btn, sec_len_num, sec_topk_num


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pd,
    sec_btn,
    sec_len_num,
    sec_topk_num,
    viz_agg_full,
    viz_gt_full,
    viz_pred_full,
):
    mo.stop(not sec_btn.value, mo.md("Click **Suggest sections** to rank candidates."))

    _sc_L = int(sec_len_num.value)
    _sc_k = int(sec_topk_num.value)
    _sc_n = len(viz_gt_full)
    mo.stop(
        _sc_n < _sc_L,
        mo.md(f"Series is only {_sc_n} samples; need at least {_sc_L}."),
    )

    # ON/OFF from the ground truth itself. A fixed small floor is used rather than the
    # appliance's min_threshold because the exported CSVs do not carry the threshold, and
    # the aim here is to FIND a section, not to score the model.
    _sc_floor = max(10.0, 0.02 * float(np.nanmax(viz_gt_full)))
    _sc_on = (viz_gt_full > _sc_floor).astype(np.int8)
    _sc_edge = np.diff(np.concatenate([[0], _sc_on, [0]]))
    _sc_rise = np.flatnonzero(_sc_edge == 1)
    _sc_fall = np.flatnonzero(_sc_edge == -1)

    # Stride a quarter of the section so candidates overlap but the scan stays cheap.
    _sc_stride = max(1, _sc_L // 4)
    _sc_rows = []
    for _sc_s in range(0, _sc_n - _sc_L + 1, _sc_stride):
        _sc_e = _sc_s + _sc_L
        _sc_seg_gt = viz_gt_full[_sc_s:_sc_e]
        _sc_seg_ag = viz_agg_full[_sc_s:_sc_e]
        _sc_seg_pr = viz_pred_full[_sc_s:_sc_e]
        if not np.isfinite(_sc_seg_pr).all():
            continue  # a window with no prediction would plot as a gap

        # Complete activations: both the rise and the fall inside the section.
        _sc_n_act = int(
            sum(1 for _r, _f in zip(_sc_rise, _sc_fall) if _r >= _sc_s and _f <= _sc_e)
        )
        _sc_on_frac = float(_sc_on[_sc_s:_sc_e].mean())
        # How much of the aggregate is NOT this appliance -- the disaggregation difficulty.
        _sc_agg_sum = float(_sc_seg_ag.sum())
        _sc_other = (
            1.0 - float(_sc_seg_gt.sum()) / _sc_agg_sum if _sc_agg_sum > 0 else 0.0
        )

        if _sc_n_act == 0:
            continue  # nothing to show

        # Transparent additive score; each term is reported so a choice can be overridden.
        _sc_act_term = 1.0 if 1 <= _sc_n_act <= 3 else max(0.0, 1.0 - 0.2 * (_sc_n_act - 3))
        _sc_dut_term = 1.0 - abs(_sc_on_frac - 0.25) / 0.25      # peaks at 25% on
        _sc_dut_term = max(0.0, _sc_dut_term)
        _sc_oth_term = min(1.0, max(0.0, _sc_other))
        # How much of the activation the model actually caught, reported so a section is
        # not picked that only flatters (or only damns) the selected model.
        _sc_cov = (
            float(_sc_seg_pr[_sc_on[_sc_s:_sc_e] > 0].mean())
            / max(1.0, float(_sc_seg_gt[_sc_on[_sc_s:_sc_e] > 0].mean()))
            if _sc_on[_sc_s:_sc_e].any()
            else 0.0
        )
        _sc_score = 2.0 * _sc_act_term + 1.5 * _sc_dut_term + 1.0 * _sc_oth_term

        _sc_rows.append({
            "start": _sc_s,
            "end": _sc_e,
            "activations": _sc_n_act,
            "on_frac": round(_sc_on_frac, 3),
            "other_load_frac": round(_sc_other, 3),
            "gt_peak_W": round(float(_sc_seg_gt.max()), 1),
            "pred_recovery": round(_sc_cov, 2),
            "score": round(_sc_score, 3),
        })

    mo.stop(
        not _sc_rows,
        mo.md(
            "No section contains a complete activation. Try a longer section, or check "
            "that the selected appliance actually activates in this test house."
        ),
    )

    _sc_df = (
        pd.DataFrame(_sc_rows)
        .sort_values("score", ascending=False)
        .head(_sc_k)
        .reset_index(drop=True)
    )
    sec_cand = _sc_df.to_dict("records")
    mo.vstack([
        mo.md(
            f"**{len(_sc_rows)} candidate sections**; top {len(sec_cand)} shown. "
            "`pred_recovery` is the mean prediction over the ON samples divided by the "
            "mean ground truth there: ~1 means the model tracked the activation, ~0 "
            "means it missed it entirely."
        ),
        _sc_df,
    ])
    return (sec_cand,)


@app.cell(hide_code=True)
def _(mo, sec_cand):
    def _label(c):
        return (f"{c['start']}–{c['end']}  ({c['activations']} act, "
                f"on {c['on_frac']:.0%}, recovery {c['pred_recovery']:.2f})")

    _sp_opts = {_label(_c): _i for _i, _c in enumerate(sec_cand)}
    sec_pick_dd = mo.ui.dropdown(
        _sp_opts, value=_label(sec_cand[0]), label="Candidate section",
    )
    sec_pick_dd
    return (sec_pick_dd,)


@app.cell(hide_code=True)
def _(mo, sec_cand, sec_pick_dd, set_sec):
    # Bounds are captured here, at cell-run time, so the button applies the candidate
    # currently selected above rather than whatever was selected when it was created.
    _ap_c = sec_cand[int(sec_pick_dd.value)]
    _ap_bounds = (int(_ap_c["start"]), int(_ap_c["end"]))
    apply_sec_btn = mo.ui.button(
        label=f"Use this section ({_ap_bounds[0]}–{_ap_bounds[1]})",
        on_click=lambda _, _b=_ap_bounds: set_sec(_b),
    )
    mo.vstack([
        apply_sec_btn,
        mo.md(
            "_Caveat: the per-window CSVs carry no timestamps and windows containing NaN "
            "were dropped when the test set was built, so a section can splice across a "
            "real-time gap. `_index.csv` in the data directory holds each window's start "
            "date if you need to confirm a section is continuous._"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(get_sec, mo, set_sec):
    # Values come FROM state and write back TO it, so these survive an upstream re-run.
    viz_start_num = mo.ui.number(
        value=get_sec()[0], label="Start sample point",
        on_change=lambda v: set_sec((int(v), get_sec()[1])),
    )
    viz_end_num = mo.ui.number(
        value=get_sec()[1], label="End sample point",
        on_change=lambda v: set_sec((get_sec()[0], int(v))),
    )
    mo.hstack([viz_start_num, viz_end_num])
    return viz_end_num, viz_start_num


@app.cell(hide_code=True)
def _(
    MODEL_LABELS,
    io,
    mo,
    np,
    plt,
    viz_agg_full,
    viz_end_num,
    viz_gt_full,
    viz_model_dd,
    viz_pred_full,
    viz_start_num,
):
    _vp_start = max(0, int(viz_start_num.value))
    _vp_end = min(len(viz_agg_full), int(viz_end_num.value))

    mo.stop(
        _vp_end <= _vp_start,
        mo.md(f"Empty section: start={_vp_start}, end={_vp_end}."),
    )

    _vp_agg = viz_agg_full[_vp_start:_vp_end]
    _vp_gt = viz_gt_full[_vp_start:_vp_end]
    _vp_pred = viz_pred_full[_vp_start:_vp_end]
    # X runs from 0 rather than from the absolute sample index: matplotlib abbreviates a
    # large shared offset into a "+1.075e6" annotation in the corner, which is noise on a
    # figure whose x-axis is only ever read as elapsed samples.
    _vp_x = np.arange(len(_vp_gt))

    _vp_fig, _vp_ax = plt.subplots(1, 1, figsize=(9.5, 3.0), constrained_layout=True)
    # The aggregate is deliberately not drawn: it dwarfs the appliance on a shared axis
    # and the comparison that matters is prediction against ground truth. It is still
    # loaded, and still drives the section scoring.
    _vp_ax.plot(_vp_x, _vp_gt, color="#2ca02c", lw=0.9, label="Ground-Truth")
    _vp_ax.plot(
        _vp_x, _vp_pred, color="#ff7f0e", lw=0.9,
        label=f"Prediction — {MODEL_LABELS.get(viz_model_dd.value, viz_model_dd.value)}",
    )
    _vp_ax.set_xlim(0, len(_vp_x) - 1)
    _vp_ax.set_xlabel("Sampling Points")
    _vp_ax.set_ylabel("Power (W)")
    _vp_ax.ticklabel_format(style="plain", useOffset=False, axis="both")

    viz_buf = io.BytesIO()
    _vp_fig.savefig(viz_buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(_vp_fig)
    viz_buf.seek(0)
    viz_png_bytes = viz_buf.getvalue()

    # An all-zero prediction over the section is usually the section's fault, not the
    # model's, so say which it is rather than leaving a flat orange line unexplained.
    _vp_note = ""
    if np.nanmax(_vp_pred) <= 1.0:
        _vp_note = (
            "**The prediction is flat at zero over this section.** "
            + ("The ground truth is also off here, so the section simply contains no "
               "activation -- use **Suggest sections** above."
               if np.nanmax(_vp_gt) <= 1.0 else
               f"The ground truth peaks at {np.nanmax(_vp_gt):.0f} W here, so this model "
               "genuinely missed the activation.")
        )

    mo.vstack(
        ([mo.md(_vp_note)] if _vp_note else [])
        + [
            mo.image(src=viz_png_bytes),
            mo.download(data=viz_png_bytes, filename="prediction_plot.png", label="Download PNG"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
