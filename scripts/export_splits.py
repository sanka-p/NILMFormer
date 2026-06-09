import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import io
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

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

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.helpers.preprocessing import (
        UKDALE_DataBuilder,
        REFIT_DataBuilder,
        REDD_DataBuilder,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
    )

    return (
        REDD_DataBuilder,
        REFIT_DataBuilder,
        UKDALE_DataBuilder,
        io,
        mo,
        np,
        os,
        pd,
        plt,
        split_train_test_nilmdataset,
        split_train_test_pdl_nilmdataset,
        yaml,
    )


@app.cell
def _(mo):
    data_path_input = mo.ui.text(value="data/", label="Data path", full_width=True)
    out_path_input = mo.ui.text(value="data_csv_splits/", label="Output path", full_width=True)
    dataset_dd = mo.ui.dropdown(["UKDALE", "REFIT", "REDD"], value="UKDALE", label="Dataset")
    sr_dd = mo.ui.dropdown(
        ["10s", "1min", "10min", "30min"], value="10s", label="Sampling rate"
    )
    window_size_num = mo.ui.number(value=256, label="Window size")
    seed_num = mo.ui.number(value=0, label="Seed")
    max_samples_num = mo.ui.number(value=500, label="Max samples per split (0 = all)")
    export_btn = mo.ui.run_button(label="Export Splits")

    mo.vstack([
        mo.hstack([data_path_input, out_path_input]),
        mo.hstack([dataset_dd, sr_dd, window_size_num, seed_num, max_samples_num]),
        export_btn,
    ])
    return (
        data_path_input,
        dataset_dd,
        export_btn,
        max_samples_num,
        out_path_input,
        seed_num,
        sr_dd,
        window_size_num,
    )


@app.cell
def _(
    REDD_DataBuilder,
    REFIT_DataBuilder,
    UKDALE_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
):
    def build_splits(dataset, app_cfg, data_path, sr, ws, seed):
        if dataset == "UKDALE":
            builder = UKDALE_DataBuilder(
                data_path=f"{data_path}UKDALE/",
                mask_app=app_cfg["app"],
                sampling_rate=sr,
                window_size=ws,
            )
            data_train, st_date_train = builder.get_nilm_dataset(
                house_indicies=app_cfg["ind_house_train"]
            )
            data_test, _ = builder.get_nilm_dataset(
                house_indicies=app_cfg["ind_house_test"]
            )
            if isinstance(ws, str):
                ws = builder.window_size
            data_train, st_date_train, data_valid, _ = split_train_test_nilmdataset(
                data_train,
                st_date_train,
                perc_house_test=0.2,
                seed=seed,
            )
        elif dataset == "REFIT":
            builder = REFIT_DataBuilder(
                data_path=f"{data_path}REFIT/RAW_DATA_CLEAN/",
                mask_app=app_cfg["app"].strip(),
                sampling_rate=sr,
                window_size=ws,
            )
            data, st = builder.get_nilm_dataset(
                house_indicies=app_cfg["house_with_app_i"]
            )
            if isinstance(ws, str):
                ws = builder.window_size
            data_train, st_date_train, data_test, _ = split_train_test_pdl_nilmdataset(
                data.copy(), st.copy(), nb_house_test=2, seed=seed
            )
            data_train, st_date_train, data_valid, _ = split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=seed
            )
        elif dataset == "REDD":
            builder = REDD_DataBuilder(
                data_path=f"{data_path}REDD/redd.h5",
                mask_app=app_cfg["app"],
                sampling_rate=sr,
                window_size=ws,
            )
            data_train, _ = builder.get_nilm_dataset(
                house_indicies=list(app_cfg["ind_house_train"])
            )
            data_valid, _ = builder.get_nilm_dataset(
                house_indicies=list(app_cfg["ind_house_valid"])
            )
            data_test, _ = builder.get_nilm_dataset(
                house_indicies=list(app_cfg["ind_house_test"])
            )
            if isinstance(ws, str):
                ws = builder.window_size
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return data_train, data_valid, data_test

    return (build_splits,)


@app.cell
def _(
    build_splits,
    data_path_input,
    dataset_dd,
    export_btn,
    max_samples_num,
    mo,
    np,
    os,
    out_path_input,
    pd,
    seed_num,
    sr_dd,
    window_size_num,
    yaml,
):
    mo.stop(
        not export_btn.value,
        mo.md("Click **Export Splits** to export train/valid/test CSVs."),
    )

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")
    ) as _fds:
        _ds_cfg = yaml.safe_load(_fds)

    _ds = dataset_dd.value
    _ds_sr = sr_dd.value
    _ds_ws = int(window_size_num.value)
    _ds_seed = int(seed_num.value)
    _max_samples = int(max_samples_num.value)
    _ds_data_path = data_path_input.value
    _ds_out_root = os.path.join(os.path.dirname(__file__), "..", out_path_input.value)
    os.makedirs(_ds_out_root, exist_ok=True)

    _split_names = ["train", "valid", "test"]
    _export_counts = {}
    _failures = []
    _summary_rows = []
    _app_items = list(_ds_cfg[_ds].items())

    with mo.status.progress_bar(
        total=len(_app_items),
        title="Exporting appliances",
        completion_title="Export complete",
    ) as _app_bar:
        for _app_key, _app_cfg in _app_items:
            _app_bar.update(title=f"Building splits: {_app_key}")
            try:
                _splits = build_splits(
                    _ds, _app_cfg, _ds_data_path, _ds_sr, _ds_ws, _ds_seed
                )
            except Exception as _e:
                _failures.append((_app_key, str(_e)))
                _app_bar.update(increment=0)
                continue

            _app_dir = os.path.join(
                _ds_out_root,
                f"{_ds}_{_app_key}_{_ds_sr}_{_ds_ws}_seed{_ds_seed}",
            )
            _counts = {}
            _written = {}
            _n_on = {}
            _pct_on = {}
            for _split_name, _split_data in zip(_split_names, _splits):
                if _split_data is None or len(_split_data) == 0:
                    _counts[_split_name] = 0
                    _written[_split_name] = 0
                    _n_on[_split_name] = 0
                    _pct_on[_split_name] = 0.0
                    _split_dir = os.path.join(_app_dir, _split_name)
                    os.makedirs(_split_dir, exist_ok=True)
                    pd.DataFrame({"idx": pd.Series([], dtype=int)}).to_csv(
                        os.path.join(_app_dir, f"{_split_name}_on_idx.csv"), index=False
                    )
                    continue
                _split_dir = os.path.join(_app_dir, _split_name)
                os.makedirs(_split_dir, exist_ok=True)
                _n_true = len(_split_data)
                _n_write = _n_true if _max_samples == 0 else min(_max_samples, _n_true)
                _counts[_split_name] = _n_true
                _written[_split_name] = _n_write

                _on_flag_full = (_split_data[:, 1, 1, :] == 1).any(axis=1)
                _n_on[_split_name] = int(_on_flag_full.sum())
                _pct_on[_split_name] = round(100.0 * _n_on[_split_name] / _n_true, 1) if _n_true > 0 else 0.0

                _on_flag_written = _on_flag_full[:_n_write]
                _on_written_indices = list(np.where(_on_flag_written)[0].astype(int))
                pd.DataFrame({"idx": _on_written_indices}).to_csv(
                    os.path.join(_app_dir, f"{_split_name}_on_idx.csv"), index=False
                )

                with mo.status.progress_bar(
                    total=_n_write,
                    title=f"{_app_key} / {_split_name}",
                    remove_on_exit=True,
                ) as _wbar:
                    for _i in range(_n_write):
                        pd.DataFrame({
                            "aggregate": _split_data[_i, 0, 0, :].astype(float),
                            "ground_truth": _split_data[_i, 1, 0, :].astype(float),
                            "ground_truth_status": _split_data[_i, 1, 1, :].astype(float),
                        }).to_csv(os.path.join(_split_dir, f"{_i}.csv"), index=False)
                        _wbar.update()
            _export_counts[_app_key] = {"true": _counts, "written": _written}
            _summary_rows.append({
                "appliance": _app_key,
                "n_train": _counts.get("train", 0),
                "n_valid": _counts.get("valid", 0),
                "n_test": _counts.get("test", 0),
                "n_train_written": _written.get("train", 0),
                "n_valid_written": _written.get("valid", 0),
                "n_test_written": _written.get("test", 0),
                "n_on_train": _n_on.get("train", 0),
                "n_on_valid": _n_on.get("valid", 0),
                "n_on_test": _n_on.get("test", 0),
                "pct_on_train": _pct_on.get("train", 0.0),
                "pct_on_valid": _pct_on.get("valid", 0.0),
                "pct_on_test": _pct_on.get("test", 0.0),
            })
            _app_bar.update()

    if _summary_rows:
        _summary_path = os.path.join(
            _ds_out_root,
            f"summary_{_ds}_{_ds_sr}_{_ds_ws}_seed{_ds_seed}.csv",
        )
        pd.DataFrame(_summary_rows).to_csv(_summary_path, index=False)

    _lines = []
    if _export_counts:
        _lines.append("**Exported:**")
        for _k, _v in _export_counts.items():
            _tc = _v["true"]
            _wc = _v["written"]
            def _fmt(sp, tc=_tc, wc=_wc):
                _t = tc.get(sp, 0)
                _w = wc.get(sp, 0)
                return f"{sp}={_t} (wrote {_w})" if _w != _t else f"{sp}={_t}"
            _lines.append(
                f"- `{_ds}_{_k}_{_ds_sr}_{_ds_ws}_seed{_ds_seed}/`  "
                + ", ".join(_fmt(sp) for sp in _split_names)
            )
    else:
        _lines.append("**No data exported.**")

    if _failures:
        _lines.append("\n**Failed:**")
        for _fapp, _ferr in _failures:
            _lines.append(f"- `{_fapp}`: {_ferr}")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(
    dataset_dd,
    io,
    mo,
    np,
    os,
    out_path_input,
    pd,
    plt,
    seed_num,
    sr_dd,
    window_size_num,
):
    _st_ds = dataset_dd.value
    _st_sr = sr_dd.value
    _st_ws = int(window_size_num.value)
    _st_seed = int(seed_num.value)
    _st_out_root = os.path.join(os.path.dirname(__file__), "..", out_path_input.value)

    _st_prefix = f"{_st_ds}_"
    _st_suffix = f"_{_st_sr}_{_st_ws}_seed{_st_seed}"
    _summary_csv = os.path.join(
        _st_out_root,
        f"summary_{_st_ds}_{_st_sr}_{_st_ws}_seed{_st_seed}.csv",
    )

    _st_rows = []
    _used_summary = False

    if os.path.isfile(_summary_csv):
        _used_summary = True
        _sum_df = pd.read_csv(_summary_csv)
        for _, _sr_row in _sum_df.iterrows():
            _row = {"appliance": str(_sr_row["appliance"])}
            for _sp in ["train", "valid", "test"]:
                _row[f"n_{_sp}"] = int(_sr_row[f"n_{_sp}"])
            _total = _row["n_train"] + _row["n_valid"] + _row["n_test"]
            _row["total"] = _total
            for _sp in ["train", "valid", "test"]:
                _row[f"pct_{_sp}"] = round(100 * _row[f"n_{_sp}"] / _total, 1) if _total > 0 else 0.0
            for _sp in ["train", "valid", "test"]:
                _row[f"pct_on_{_sp}"] = float(_sr_row[f"pct_on_{_sp}"]) if f"pct_on_{_sp}" in _sr_row.index else 0.0
            _st_rows.append(_row)
    elif os.path.isdir(_st_out_root):
        for _d in sorted(os.listdir(_st_out_root)):
            if _d.startswith(_st_prefix) and _d.endswith(_st_suffix):
                _app_name = _d[len(_st_prefix):-len(_st_suffix)]
                _app_path = os.path.join(_st_out_root, _d)
                _row = {"appliance": _app_name}
                _total = 0
                for _sp in ["train", "valid", "test"]:
                    _sp_path = os.path.join(_app_path, _sp)
                    _n = len([f for f in os.listdir(_sp_path) if f.endswith(".csv")]) if os.path.isdir(_sp_path) else 0
                    _row[f"n_{_sp}"] = _n
                    _total += _n
                _row["total"] = _total
                for _sp in ["train", "valid", "test"]:
                    _row[f"pct_{_sp}"] = round(100 * _row[f"n_{_sp}"] / _total, 1) if _total > 0 else 0.0
                for _sp in ["train", "valid", "test"]:
                    _sp_path = os.path.join(_app_path, _sp)
                    _n_on_fb = 0
                    if os.path.isdir(_sp_path):
                        for _fn in sorted(os.listdir(_sp_path)):
                            if _fn.endswith(".csv"):
                                try:
                                    _fb_df = pd.read_csv(os.path.join(_sp_path, _fn))
                                    if (_fb_df["ground_truth_status"] == 1).any():
                                        _n_on_fb += 1
                                except Exception:
                                    pass
                    _n_sp = _row[f"n_{_sp}"]
                    _row[f"pct_on_{_sp}"] = round(100.0 * _n_on_fb / _n_sp, 1) if _n_sp > 0 else 0.0
                _st_rows.append(_row)

    mo.stop(
        len(_st_rows) == 0,
        mo.md("No exported splits found. Click **Export Splits** first."),
    )

    _st_df = pd.DataFrame(_st_rows).set_index("appliance")
    _st_cols = ["n_train", "n_valid", "n_test", "total", "pct_train", "pct_valid", "pct_test", "pct_on_train", "pct_on_valid", "pct_on_test"]
    _st_df = _st_df[_st_cols]

    _app_names = _st_df.index.tolist()
    _x = np.arange(len(_app_names))
    _width = 0.25
    _colors = {"train": "#1f77b4", "valid": "#ff7f0e", "test": "#2ca02c"}

    _bar_fig, _bar_ax = plt.subplots(figsize=(max(5, len(_app_names) * 1.2 + 1.5), 3.5), constrained_layout=True, facecolor="white")
    for _bi, _sp in enumerate(["train", "valid", "test"]):
        _bar_ax.bar(
            _x + (_bi - 1) * _width,
            _st_df[f"n_{_sp}"].values,
            width=_width,
            label=_sp.capitalize(),
            color=_colors[_sp],
        )
    _bar_ax.set_xticks(_x)
    _bar_ax.set_xticklabels(_app_names, rotation=20, ha="right")
    _bar_ax.set_ylabel("Sample count")
    _bar_ax.set_title(f"{_st_ds} — {_st_sr} — ws={_st_ws} — seed={_st_seed}", loc="left")
    _bar_ax.legend(frameon=False, ncols=3)
    _bar_buf = io.BytesIO()
    _bar_fig.savefig(_bar_buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(_bar_fig)
    _bar_buf.seek(0)
    _bar_img = mo.image(src=_bar_buf.read())

    if _used_summary:
        _note = mo.md("_Counts show full dataset sizes (from summary CSV); fewer CSVs may be written on disk. `pct_on_*` is over the full split._")
    else:
        _note = mo.md("_Counts and `pct_on_*` are computed over written CSVs only (no summary CSV found)._")
    mo.vstack([_note, _st_df, _bar_img])
    return


@app.cell(hide_code=True)
def _(dataset_dd, mo, os, out_path_input, seed_num, sr_dd, window_size_num):
    _vz_ds = dataset_dd.value
    _vz_sr = sr_dd.value
    _vz_ws = int(window_size_num.value)
    _vz_seed = int(seed_num.value)
    _vz_out_root = os.path.join(os.path.dirname(__file__), "..", out_path_input.value)
    _vz_prefix = f"{_vz_ds}_"
    _vz_suffix = f"_{_vz_sr}_{_vz_ws}_seed{_vz_seed}"
    _vz_app_keys = []
    if os.path.isdir(_vz_out_root):
        for _d in sorted(os.listdir(_vz_out_root)):
            if _d.startswith(_vz_prefix) and _d.endswith(_vz_suffix):
                _vz_app_keys.append(_d[len(_vz_prefix):-len(_vz_suffix)])

    vz_appliance_dd = mo.ui.dropdown(
        _vz_app_keys or ["(none)"],
        value=_vz_app_keys[0] if _vz_app_keys else "(none)",
        label="Appliance",
    )
    vz_split_dd = mo.ui.dropdown(
        ["train", "valid", "test"],
        value="train",
        label="Split",
    )
    vz_order_dd = mo.ui.dropdown(
        ["File order", "On first", "On only"],
        value="File order",
        label="Order",
    )
    mo.hstack([vz_appliance_dd, vz_split_dd, vz_order_dd])
    return vz_appliance_dd, vz_order_dd, vz_split_dd


@app.cell(hide_code=True)
def _(mo):
    get_idx, set_idx = mo.state(0)
    return get_idx, set_idx


@app.cell(hide_code=True)
def _(
    dataset_dd,
    get_idx,
    mo,
    os,
    out_path_input,
    pd,
    seed_num,
    set_idx,
    sr_dd,
    vz_appliance_dd,
    vz_order_dd,
    vz_split_dd,
    window_size_num,
):
    _sl_ds = dataset_dd.value
    _sl_sr = sr_dd.value
    _sl_ws = int(window_size_num.value)
    _sl_seed = int(seed_num.value)
    _sl_out_root = os.path.join(os.path.dirname(__file__), "..", out_path_input.value)
    _sl_app_dir_name = f"{_sl_ds}_{vz_appliance_dd.value}_{_sl_sr}_{_sl_ws}_seed{_sl_seed}"
    _sl_app_dir = os.path.join(_sl_out_root, _sl_app_dir_name)
    _sl_split_dir = os.path.join(_sl_app_dir, vz_split_dd.value)
    _sl_n = (
        len([f for f in os.listdir(_sl_split_dir) if f.endswith(".csv")])
        if os.path.isdir(_sl_split_dir)
        else 0
    )

    _on_idx_path = os.path.join(_sl_app_dir, f"{vz_split_dd.value}_on_idx.csv")
    if os.path.isfile(_on_idx_path):
        try:
            _on_idx_df = pd.read_csv(_on_idx_path)
            _on_indices = list(_on_idx_df["idx"].astype(int)) if len(_on_idx_df) > 0 else []
        except Exception:
            _on_indices = []
    else:
        _on_indices = []
        if os.path.isdir(_sl_split_dir):
            for _fn in sorted(os.listdir(_sl_split_dir), key=lambda x: int(x[:-4]) if x[:-4].isdigit() else -1):
                if _fn.endswith(".csv") and _fn[:-4].isdigit():
                    try:
                        _fb_df = pd.read_csv(os.path.join(_sl_split_dir, _fn))
                        if (_fb_df["ground_truth_status"] == 1).any():
                            _on_indices.append(int(_fn[:-4]))
                    except Exception:
                        pass

    _all_indices = list(range(_sl_n))
    _on_set = set(_on_indices)
    _off_indices = [i for i in _all_indices if i not in _on_set]

    _order_mode = vz_order_dd.value
    if _order_mode == "File order":
        vz_order = _all_indices
    elif _order_mode == "On first":
        vz_order = sorted(_on_indices) + sorted(_off_indices)
    else:
        vz_order = sorted(_on_indices)

    vz_n_order = len(vz_order)
    vz_sl_max = max(vz_n_order - 1, 0)
    _pos = min(get_idx(), vz_sl_max) if vz_n_order > 0 else 0
    _file_idx = vz_order[_pos] if vz_n_order > 0 else 0

    prev_btn = mo.ui.button(label="◀", on_click=lambda _: set_idx(max(0, get_idx() - 1)))
    next_btn = mo.ui.button(label="▶", on_click=lambda _: set_idx(min(vz_sl_max, get_idx() + 1)))
    sample_slider = mo.ui.slider(0, vz_sl_max, value=_pos, on_change=set_idx, label="Sample")
    idx_input = mo.ui.number(start=0, stop=vz_sl_max, value=_pos, on_change=lambda v: set_idx(int(v)), label="Index")

    _browser_ui = (
        mo.hstack([prev_btn, sample_slider, next_btn, idx_input, mo.md(f"/ **{vz_sl_max}** (file #{_file_idx})")])
        if vz_n_order > 0
        else mo.md("No samples for this order/split.")
    )
    _browser_ui
    return vz_n_order, vz_order, vz_sl_max


@app.cell(hide_code=True)
def _(
    dataset_dd,
    get_idx,
    io,
    mo,
    np,
    os,
    out_path_input,
    pd,
    plt,
    seed_num,
    sr_dd,
    vz_appliance_dd,
    vz_n_order,
    vz_order,
    vz_sl_max,
    vz_split_dd,
    window_size_num,
):
    _pl_ds = dataset_dd.value
    _pl_sr = sr_dd.value
    _pl_ws = int(window_size_num.value)
    _pl_seed = int(seed_num.value)
    _pl_out_root = os.path.join(os.path.dirname(__file__), "..", out_path_input.value)
    _pl_app_dir = f"{_pl_ds}_{vz_appliance_dd.value}_{_pl_sr}_{_pl_ws}_seed{_pl_seed}"

    if vz_n_order == 0:
        _pl_out = mo.md("No samples for this order/split.")
    else:
        _pl_pos = min(get_idx(), vz_sl_max)
        _pl_file_idx = vz_order[_pl_pos]
        _pl_csv_path = os.path.join(
            _pl_out_root, _pl_app_dir, vz_split_dd.value, f"{_pl_file_idx}.csv"
        )

        if not os.path.isfile(_pl_csv_path):
            _pl_out = mo.md(
                f"CSV not found: `{_pl_csv_path}`. Run **Export Splits** first."
            )
        else:
            _pl_df = pd.read_csv(_pl_csv_path)
            _pl_x = np.arange(len(_pl_df))
            _pl_is_on = bool((_pl_df["ground_truth_status"] == 1).any())
            _pl_on_label = "ON" if _pl_is_on else "OFF"

            _pl_fig, _pl_ax = plt.subplots(
                1, 1, figsize=(9.5, 3.0), constrained_layout=True, facecolor="white"
            )
            _pl_on_mask = _pl_df["ground_truth_status"].values == 1
            if _pl_on_mask.any():
                _pl_ax.fill_between(
                    _pl_x,
                    0,
                    _pl_df["aggregate"].values.max(),
                    where=_pl_on_mask,
                    alpha=0.12,
                    color="#2ca02c",
                    label="On (status=1)",
                )
            _pl_ax.plot(
                _pl_x,
                _pl_df["aggregate"].values,
                color="#1f77b4",
                lw=0.9,
                label="Aggregate",
            )
            _pl_ax.plot(
                _pl_x,
                _pl_df["ground_truth"].values,
                color="#2ca02c",
                lw=0.9,
                label="Ground-Truth",
            )
            _pl_ax.set_xlim(0, len(_pl_df) - 1)
            _pl_ax.set_xlabel("Timestep")
            _pl_ax.set_ylabel("Power (W)")
            _pl_ax.set_title(
                f"{_pl_ds} — {vz_appliance_dd.value} — {vz_split_dd.value} — pos {_pl_pos} (file #{_pl_file_idx}) [{_pl_on_label}]",
                loc="left",
            )
            _pl_ax.legend(frameon=False, ncols=3)
            _pl_buf = io.BytesIO()
            _pl_fig.savefig(
                _pl_buf, format="png", dpi=130, bbox_inches="tight", facecolor="white"
            )
            plt.close(_pl_fig)
            _pl_buf.seek(0)
            _pl_out = mo.image(src=_pl_buf.read())
    _pl_out
    return


if __name__ == "__main__":
    app.run()
