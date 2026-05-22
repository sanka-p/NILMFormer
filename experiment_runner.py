import marimo

__generated_with = "0.23.5"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import subprocess
    import sys
    import os
    import pathlib
    import torch
    import polars as pl
    import altair as alt

    return alt, mo, os, pathlib, pl, subprocess, sys, torch


@app.cell
def _():
    DATASET = "UKDALE"
    SAMPLING_RATE = "10s"
    MODELS = ["NILMFormer", "BiLSTM", "BiGRU", "BERT4NILM", "CNN1D"]
    SEEDS = [0, 1, 2]
    WINDOWS = [128, 256, 512]
    APPLIANCES = ["WashingMachine", "Microwave", "Dishwasher", "Kettle", "Fridge"]
    return APPLIANCES, DATASET, MODELS, SAMPLING_RATE, SEEDS, WINDOWS


@app.cell
def _(DATASET, SAMPLING_RATE, os, pathlib):
    def result_path(appliance, window, model, seed):
        return f"result/{DATASET}_{appliance}_{SAMPLING_RATE}/{window}/{model}_{seed}.pt"

    def log_path(appliance, window, model, seed):
        return f"logs/{DATASET}_{appliance}_{SAMPLING_RATE}/{window}/{model}_{seed}.log"

    def is_trained(appliance, window, model, seed):
        return pathlib.Path(result_path(appliance, window, model, seed)).exists()

    def build_experiment_matrix(appliances, models, seeds, windows):
        rows = []
        for appliance in appliances:
            for window in windows:
                for model in models:
                    for seed in seeds:
                        trained = is_trained(appliance, window, model, seed)
                        rows.append({
                            "appliance": appliance,
                            "window": window,
                            "model": model,
                            "seed": seed,
                            "status": "✓ Done" if trained else "⏳ Pending",
                        })
        return rows

    # ensure logs root exists
    os.makedirs("logs", exist_ok=True)
    return build_experiment_matrix, is_trained, log_path, result_path


@app.cell
def _(mo):
    mo.md("""
    # NILMFormer Experiment Runner
    """)
    return


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="↻ Refresh Status")
    refresh
    return (refresh,)


@app.cell
def _(APPLIANCES, MODELS, SEEDS, WINDOWS, build_experiment_matrix, refresh):
    _trigger = refresh.value
    experiment_matrix = build_experiment_matrix(APPLIANCES, MODELS, SEEDS, WINDOWS)
    return (experiment_matrix,)


@app.cell
def _(APPLIANCES, MODELS, mo):
    filter_appliance = mo.ui.dropdown(
        options=["All"] + APPLIANCES,
        value="All",
        label="Appliance",
    )
    filter_model = mo.ui.dropdown(
        options=["All"] + MODELS,
        value="All",
        label="Model",
    )
    filter_status = mo.ui.dropdown(
        options=["All", "Pending", "Done"],
        value="All",
        label="Status",
    )
    mo.hstack([filter_appliance, filter_model, filter_status])
    return filter_appliance, filter_model, filter_status


@app.cell
def _(
    experiment_matrix,
    filter_appliance,
    filter_model,
    filter_status,
    mo,
    pl,
):
    _df = pl.DataFrame(experiment_matrix)

    if filter_appliance.value != "All":
        _df = _df.filter(pl.col("appliance") == filter_appliance.value)
    if filter_model.value != "All":
        _df = _df.filter(pl.col("model") == filter_model.value)
    if filter_status.value == "Pending":
        _df = _df.filter(pl.col("status") == "⏳ Pending")
    elif filter_status.value == "Done":
        _df = _df.filter(pl.col("status") == "✓ Done")

    _done = (_df["status"] == "✓ Done").sum()
    _total = len(_df)

    mo.vstack([
        mo.md(f"**Progress: {_done}/{_total}** experiments complete in current filter"),
        mo.ui.table(_df, sortable=True, filterable=False),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Run Experiment
    """)
    return


@app.cell
def _(experiment_matrix, mo):
    _pending = [
        f"{e['appliance']} / {e['model']} / seed={e['seed']} / win={e['window']}"
        for e in experiment_matrix
        if e["status"] == "⏳ Pending"
    ]

    experiment_selector = mo.ui.dropdown(
        options=_pending if _pending else ["(none pending)"],
        value=_pending[0] if _pending else "(none pending)",
        label="Select experiment to run",
        full_width=True,
    )
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="▶ Run Selected Experiment")
    run_btn
    return (run_btn,)


@app.cell
def _(
    DATASET,
    SAMPLING_RATE,
    experiment_selector,
    is_trained,
    log_path,
    mo,
    os,
    run_btn,
    subprocess,
    sys,
):
    mo.stop(
        not run_btn.value,
        mo.md("*Select an experiment above and click **▶ Run Selected Experiment** to start training.*"),
    )
    mo.stop(
        experiment_selector.value == "(none pending)",
        mo.md("**All experiments complete!** Nothing to run."),
    )

    _parts = experiment_selector.value.split(" / ")
    _appliance = _parts[0]
    _model = _parts[1]
    _seed = int(_parts[2].split("=")[1])
    _window = int(_parts[3].split("=")[1])

    mo.stop(
        is_trained(_appliance, _window, _model, _seed),
        mo.md(f"**Already trained**: `{_appliance}/{_model}/seed={_seed}/win={_window}` — skipping."),
    )

    _log_file = log_path(_appliance, _window, _model, _seed)
    os.makedirs(os.path.dirname(_log_file), exist_ok=True)

    _cmd = [
        sys.executable, "scripts/run_one_expe.py",
        "--dataset", DATASET,
        "--sampling_rate", SAMPLING_RATE,
        "--window_size", str(_window),
        "--appliance", _appliance,
        "--name_model", _model,
        "--seed", str(_seed),
    ]

    _project_root = pathlib.Path(__file__).parent
    with open(_log_file, "w") as _f:
        _proc = subprocess.run(_cmd, stdout=_f, stderr=subprocess.STDOUT, text=True, cwd=_project_root)

    if _proc.returncode == 0:
        mo.output.replace(mo.md(f"**✓ Training complete!** `{_appliance} / {_model} / seed={_seed} / win={_window}`\n\nLog saved to `{_log_file}`"))
    else:
        mo.output.replace(mo.md(f"**✗ Training failed** (exit code {_proc.returncode})\n\nSee `{_log_file}` for details."))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Log Viewer
    """)
    return


@app.cell
def _(APPLIANCES, DATASET, MODELS, SAMPLING_RATE, SEEDS, WINDOWS, mo, os):
    def _find_logs():
        found = []
        for _a in APPLIANCES:
            for _w in WINDOWS:
                for _m in MODELS:
                    for _s in SEEDS:
                        _p = f"logs/{DATASET}_{_a}_{SAMPLING_RATE}/{_w}/{_m}_{_s}.log"
                        if os.path.exists(_p):
                            found.append(f"{_a} / {_m} / seed={_s} / win={_w}")
        return found

    _available_logs = _find_logs()

    log_selector = mo.ui.dropdown(
        options=_available_logs if _available_logs else ["(no logs yet)"],
        value=_available_logs[0] if _available_logs else "(no logs yet)",
        label="Select log to view",
        full_width=True,
    )
    log_selector
    return (log_selector,)


@app.cell
def _(DATASET, SAMPLING_RATE, log_selector, mo):
    mo.stop(log_selector.value == "(no logs yet)", mo.md("*No logs available yet. Run an experiment first.*"))

    _parts = log_selector.value.split(" / ")
    _appliance = _parts[0]
    _model = _parts[1]
    _seed = int(_parts[2].split("=")[1])
    _window = int(_parts[3].split("=")[1])
    _log_file = f"logs/{DATASET}_{_appliance}_{SAMPLING_RATE}/{_window}/{_model}_{_seed}.log"

    try:
        with open(_log_file) as _f:
            _content = _f.read()
        mo.output.replace(mo.md(f"**`{_log_file}`**\n\n```\n{_content[-8000:]}\n```"))
    except FileNotFoundError:
        mo.output.replace(mo.md(f"*Log file not found: `{_log_file}`*"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Training Curves
    """)
    return


@app.cell
def _(APPLIANCES, MODELS, SEEDS, WINDOWS, is_trained, mo):
    _trained = [
        f"{_a} / {_m} / seed={_s} / win={_w}"
        for _a in APPLIANCES
        for _w in WINDOWS
        for _m in MODELS
        for _s in SEEDS
        if is_trained(_a, _w, _m, _s)
    ]

    curves_selector = mo.ui.dropdown(
        options=_trained if _trained else ["(none trained yet)"],
        value=_trained[0] if _trained else "(none trained yet)",
        label="Select trained model",
        full_width=True,
    )
    curves_selector
    return (curves_selector,)


@app.cell
def _(alt, curves_selector, mo, pl, result_path, torch):
    mo.stop(
        curves_selector.value == "(none trained yet)",
        mo.md("*No trained models yet.*"),
    )

    _parts = curves_selector.value.split(" / ")
    _appliance = _parts[0]
    _model = _parts[1]
    _seed = int(_parts[2].split("=")[1])
    _window = int(_parts[3].split("=")[1])

    _ckpt_path = result_path(_appliance, _window, _model, _seed)
    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)

    _train_loss = _ckpt.get("loss_train_history", [])
    _valid_loss = _ckpt.get("loss_valid_history", [])
    _training_time = _ckpt.get("training_time", None)
    _best_epoch = _ckpt.get("epoch_best_loss", None)
    _best_loss = _ckpt.get("value_best_loss", None)

    _epochs = list(range(1, len(_train_loss) + 1))
    _df_loss = pl.DataFrame({
        "epoch": _epochs + _epochs,
        "loss": _train_loss + _valid_loss,
        "split": ["train"] * len(_train_loss) + ["valid"] * len(_valid_loss),
    })

    _chart = (
        alt.Chart(_df_loss)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="MSE Loss"),
            color=alt.Color("split:N", title="Split"),
            tooltip=["epoch:Q", "split:N", "loss:Q"],
        )
        .properties(
            title=f"{_model} | {_appliance} | seed={_seed} | win={_window}",
            width=600,
            height=300,
        )
    )

    _info_parts = []
    if _training_time is not None:
        _info_parts.append(f"**Training time:** {_training_time:.1f}s")
    if _best_epoch is not None:
        _info_parts.append(f"**Best epoch:** {_best_epoch}")
    if _best_loss is not None:
        _info_parts.append(f"**Best val loss:** {_best_loss:.6f}")

    mo.vstack([
        mo.md(" &nbsp;|&nbsp; ".join(_info_parts)) if _info_parts else mo.md(""),
        _chart,
    ])
    return


if __name__ == "__main__":
    app.run()
