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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import yaml

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.helpers.preprocessing import (
        UKDALE_DataBuilder,
        REFIT_DataBuilder,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
    )
    return (
        REFIT_DataBuilder,
        UKDALE_DataBuilder,
        glob,
        io,
        matplotlib,
        mo,
        np,
        os,
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
        ["10s", "1min", "10min", "30min"], value="1min", label="Sampling rate"
    )
    window_size_num = mo.ui.number(value=128, label="Window size")
    seed_num = mo.ui.number(value=0, label="Seed")
    n_windows_num = mo.ui.number(value=4, label="Windows to compare")
    load_btn = mo.ui.run_button(label="Load & Plot")

    mo.vstack([
        mo.hstack([result_path_input, data_path_input]),
        mo.hstack([dataset_dd, sr_dd, window_size_num, seed_num, n_windows_num]),
        load_btn,
    ])
    return (
        data_path_input,
        dataset_dd,
        load_btn,
        n_windows_num,
        result_path_input,
        seed_num,
        sr_dd,
        window_size_num,
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

    def find_good_windows(data_test, n=4):
        activity = data_test[:, 1, 1, :].mean(axis=1)
        power_sum = data_test[:, 1, 0, :].sum(axis=1)
        candidates = np.where(activity > 0.3)[0]
        if len(candidates) == 0:
            candidates = np.arange(len(data_test))
        sorted_candidates = candidates[np.argsort(power_sum[candidates])[::-1]]
        return sorted_candidates[:n].tolist()

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

    def make_figure(agg, gt_power, gt_state, predictions, good_idx, app_name, sr):
        sr_seconds = {"10s": 10, "1min": 60, "10min": 600, "30min": 1800}.get(sr, 60)
        t = np.arange(len(agg)) * sr_seconds

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(t, agg, color="steelblue", lw=1, label="Aggregate")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title(f"{app_name}  —  test window {good_idx}")
        axes[0].legend(loc="upper right")

        axes[1].plot(t, gt_power, color="black", lw=1.8, label="Ground truth", zorder=5)
        axes[1].fill_between(
            t, 0, gt_power, where=gt_state > 0, alpha=0.12, color="black"
        )
        colors = plt.cm.tab10.colors
        for i, (model_name, pred_2d) in enumerate(predictions.items()):
            if good_idx < len(pred_2d):
                axes[1].plot(
                    t,
                    pred_2d[good_idx],
                    color=colors[i % 10],
                    lw=1,
                    alpha=0.85,
                    label=model_name,
                )
        axes[1].set_ylabel("Appliance power (W)")
        axes[1].set_xlabel(f"Time (s, Δ={sr_seconds}s)")
        axes[1].legend(loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()
        return fig

    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return mo.image(src=buf.read())

    return fig_to_image, find_good_windows, load_predictions, load_test_data, make_figure


@app.cell
def _(
    data_path_input,
    dataset_dd,
    fig_to_image,
    find_good_windows,
    load_btn,
    load_predictions,
    load_test_data,
    make_figure,
    mo,
    n_windows_num,
    os,
    result_path_input,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
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
    _n_windows = int(n_windows_num.value)
    _data_path = data_path_input.value
    _result_path = result_path_input.value
    _appliances_cfg = _datasets_cfg[_dataset]

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

        _predictions = load_predictions(
            _result_path, _dataset, _app_key, _sr, _ws, _seed
        )
        _n_models = len(_predictions)
        _good_indices = find_good_windows(_data_test, n=_n_windows)

        _panels = []
        for _idx in _good_indices:
            _agg = _data_test[_idx, 0, 0, :]
            _gt_power = _data_test[_idx, 1, 0, :]
            _gt_state = _data_test[_idx, 1, 1, :]
            _fig = make_figure(
                _agg, _gt_power, _gt_state, _predictions, _idx, _app_name, _sr
            )
            _caption = (
                mo.md(f"Window {_idx} · {_n_models} model(s) loaded")
                if _n_models
                else mo.md(f"Window {_idx} · no checkpoints found at `{_result_path}`")
            )
            _panels.append(mo.vstack([fig_to_image(_fig), _caption]))

        _tabs[_app_key] = mo.vstack(_panels)

    mo.ui.tabs(_tabs)
    return


if __name__ == "__main__":
    app.run()
