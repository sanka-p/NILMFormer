#################################################################################################################
#
# @description : SotA comparison table built straight from the checkpoints.
#
# scripts/make_paper_table.py reads results/runs_cache.csv, which predates the REDD
# retrain, and refreshing that cache has been killed mid-write on this box before (it
# truncated 1395 rows to 901). This reads the .pt files directly for one explicit slice,
# so the numbers cannot come from a stale cache, and it takes --windows because the
# ws=128 sweep is still running and must be excluded until it finishes.
#
# Cells are the mean over the selected window sizes and seeds. Every cell prints its n,
# and the table is refused if coverage is uneven -- averaging 6 runs against 3 in the
# same column is the kind of thing that silently flatters whichever model happens to
# have finished.
#
# Usage:
#     PYTHONPATH=. ./.venv/bin/python -m scripts.make_sota_table_direct \
#         --dataset REDD --windows 256 512
#
#################################################################################################################

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Retrained roots first; the legacy 3-epoch roots are the fallback for anything not yet
#: redone. First hit wins, matching make_table.py's dedup.
ROOTS = ["result-redd-fix", "result-ukdale-fix", "result-redd-archfix",
         "result-tcn-ablation", "result-synagg", "2207-results/result"]

PROPOSED = "TCN_KL_aug"
#: Per-appliance window selection, written by scripts/pick_window.py and swept by
#: scripts/build_tod_joblist.sh. --windows dynamic reads it so the table scores each
#: appliance at the window it was actually trained at.
WS_JSON = "results/window_selection.json"
SOTA_METHODS = [
    (PROPOSED, r"Proposed"),
    ("NILMFormer", r"NILMFormer\cite{petralia_nilmformer_2025}"),
    ("BERT4NILM", r"BERT4NILM\cite{yue_bert4nilm_2020}"),
    ("BiLSTM", r"BiLSTM\cite{10.1145/2821650.2821672}"),
    ("BiGRU", r"BiGRU\cite{Precioso2023}"),
]
#: Ablation arms, in the paper's row order. TCN_aug_curated is "TCN + Aug" and
#: TCN_KL_aug (arm 3) is the proposed model, matching the existing ablation table.
ABLATION_METHODS = [
    ("TCN", r"TCN"),
    ("TCN_KL_scratch", r"TCN + KL"),
    ("TCN_aug_curated", r"TCN + Aug"),
    (PROPOSED, r"TCN + KL + Aug"),
]
#: Prior-information study (result-redd-tod/README.md): the two things NILMFormer has
#: that no TCN arm had -- a time-of-day channel, and per-window shape/magnitude
#: separation. The reference row is the EXISTING TCN_KL_scratch in result-redd-fix:
#:   --preset tod --roots result-redd-tod result-redd-fix
#: The timestamp-shuffle control is deliberately absent: it is run on one appliance
#: only, so listing it here would trip the uneven-coverage refusal every time.
#: The class-weighted-loss arms sit directly under the reference because they address
#: the dominant measured defect (over-firing) at the source rather than after the fact.
#: They are NOT the default fix. Under the prevalence-matched scoring this preset uses,
#: the unweighted reference beats w_off=3 on F1 for WM, DW and Microwave -- the loss
#: weighting's gain at the stock threshold was mostly calibration, which scoring fixes
#: without a retrain. They stay in the table for MAE, which rescoring cannot change
#: (w_off=3 cuts WM MAE 31.4 -> 23.3 W), and w_off=10 stays as the cliff: it silences
#: Microwave (F1 4.5 even after rescoring, i.e. lost ranking, not miscalibration).
#: See result-redd-tod/README.md.
TOD_METHODS = [
    ("TCN_KL_scratch", r"TCN + KL"),
    ("TCN_KL_cw1_5", r"TCN + KL + ClassW ($w_{off}$=1.5)"),
    ("TCN_KL_cw3", r"TCN + KL + ClassW ($w_{off}$=3)"),
    ("TCN_KL_cw10", r"TCN + KL + ClassW ($w_{off}$=10)"),
    ("TCN_KL_act_cw3", r"TCN + KL + ActCrop + ClassW ($w_{off}$=3)"),
    ("TCN_KL_stat", r"TCN + KL + Stat"),
    ("TCN_KL_act", r"TCN + KL + ActCrop"),
    ("TCN_KL_act_stat", r"TCN + KL + ActCrop + Stat"),
    ("TCN_KL_act_tod", r"TCN + KL + ActCrop + ToD"),
    ("TCN_KL_act_tod_film", r"TCN + KL + ActCrop + ToD (FiLM)"),
    ("TCN_KL_act_tod_stat", r"TCN + KL + ActCrop + ToD + Stat"),
]
METHODS = SOTA_METHODS
#: (directory keys, column header). REDD keys the washer/dryer two ways for one meter.
REDD_COLS = [(["WashingMachine", "WasherDryer"], "WM"),
             (["Dishwasher"], "DW"),
             (["Microwave"], "Micro."),
             (["Fridge"], "Fridge")]
UKDALE_COLS = [(["WashingMachine"], "WM"), (["Dishwasher"], "DW"),
               (["Kettle"], "Kettle"), (["Microwave"], "Micro."),
               (["Fridge"], "Fridge")]
#: (key, row label, higher_is_better, unit, bold_best)
METRICS = [("BALANCED_ACCURACY", "BA", True, "%", True),
           ("F1_SCORE", "F1", True, "%", True),
           ("MAE", "MAE", False, "", True)]
#: Convergence blocks, appended with --with-convergence. These come from the
#: convergence index rather than the checkpoints' test metrics.
#: Epochs is deliberately NOT bolded: fewer epochs can mean premature early stopping
#: rather than faster convergence (17% of NILMFormer's REDD runs keep a checkpoint from
#: the first two epochs), so marking the smallest as "best" would be misleading.
CONV_METRICS = [("epochs_to_best", "Epochs", False, "", False),
                ("time_to_best_s", "Time (s)", False, "", True)]


def collect(dataset, windows, seeds, sr, score="stored", postprocess=False,
            prior="test"):
    """{(model, column): [values...]} with each run counted once.

    `windows` is either a flat list applied to every appliance, or a dict keyed by
    column header giving that appliance's own window(s). The ToD study trains each
    appliance at the window scripts/pick_window.py selected for it, so averaging one
    fixed grid across all four would score most arms at windows they were never run at.
    """
    cols = REDD_COLS if dataset == "REDD" else UKDALE_COLS
    out, nruns = {}, {}
    for model, _ in METHODS:
        for keys, header in cols:
            vals = {m: [] for m, _, _, _, _ in METRICS}
            seen = set()
            for ws in (windows[header] if isinstance(windows, dict) else windows):
                for seed in seeds:
                    got = None
                    for root in ROOTS:
                        for key in keys:
                            f = os.path.join(root, f"{dataset}_{key}_{sr}", str(ws),
                                             f"{model}_{seed}.pt")
                            if os.path.exists(f) and (ws, seed) not in seen:
                                try:
                                    log = torch.load(f, map_location="cpu",
                                                     weights_only=False)
                                    got = (log["test_metrics_timestamp"]
                                           if score == "stored" else
                                           _prevalence_rescore(log, dataset,
                                                               key, ws,
                                                               postprocess,
                                                               prior))
                                except SystemExit:
                                    raise
                                except Exception:
                                    got = None
                                if got is not None:
                                    break
                        if got is not None:
                            break
                    if got is not None:
                        seen.add((ws, seed))
                        for m, _, _, _, _ in METRICS:
                            vals[m].append(float(got[m]))
            out[(model, header)] = {m: (np.mean(v) if v else np.nan)
                                    for m, v in vals.items()}

            nruns[(model, header)] = len(seen)
    return out, nruns, [h for _, h in cols]


#: Cache of reconstructed ground-truth states, keyed (dataset, appliance, window).
_GT_CACHE = {}


def _prevalence_rescore(log, dataset, appliance, ws, postprocess=False, prior="test"):
    """Classification metrics at the prevalence-matched threshold.

    The stored `test_metrics_timestamp` is scored at the appliance's `min_threshold`,
    a constant that also serves as the regression floor and is never tuned. On a rare
    appliance that operating point is badly miscalibrated -- the arms on this grid fire
    3-31x more often than the appliance is actually on -- so comparing arms there
    measures the decision rule as much as the model.

    Prevalence matching instead picks the threshold at which the model predicts ON as
    often as the appliance is on. It needs only the duty-cycle scalar, not labels, and
    it is the criterion with the lowest worst-case regret across BA and F1 once the
    validation-to-test prior shift is accounted for (7.8 points, against 11.9 for
    F1-tuned, 12.8 MCC, 18.1 BA, 20.1 for the stock rule).

    Regression metrics (MAE, RMSE, SAE, ...) are copied through untouched: only the
    DETECTION threshold moves, the regression floor does not, so they cannot change.
    """
    from scripts.calibrate_threshold import (DetectionCurve, build_config,
                                             ground_truth_states)
    from scripts.score_postprocess import _pick_threshold, _timestamp_metrics
    from src.helpers.event_metrics import event_scores
    from src.helpers.postprocess import durations_for, postprocess_state
    # REDD writes the washing machine under the WasherDryer channel, but datasets.yaml
    # only knows the config name, so map back before rebuilding.
    app_cfg = "WashingMachine" if appliance == "WasherDryer" else appliance
    key = (dataset, app_cfg, ws)
    if key not in _GT_CACHE:
        if dataset != "REDD":
            raise SystemExit("--score prevalence currently reconstructs REDD only")
        cfg = build_config(dataset, app_cfg, ws)
        _GT_CACHE[key] = ground_truth_states(cfg)
    y_valid, y_test, floor = _GT_CACHE[key]

    yhat = np.asarray(log["test_metrics_yhat"])
    if yhat.size != y_test.size:
        raise SystemExit(
            f"{dataset}/{appliance}/ws{ws}: stored predictions are {yhat.size} points "
            f"but the rebuilt ground truth is {y_test.size}; refusing to score.")
    curve = DetectionCurve(y_test, yhat, floor)

    # Self-check: the stock rule must reproduce what the trainer stored, or the ground
    # truth is not aligned and nothing below it can be trusted.
    stored, stock = log["test_metrics_timestamp"], curve.at(floor)
    for mine, theirs in (("F1", "F1_SCORE"), ("BA", "BALANCED_ACCURACY")):
        if abs(stock[mine] - 100 * float(stored[theirs])) > 0.15:
            raise SystemExit(
                f"{dataset}/{appliance}/ws{ws}: {mine} self-check failed "
                f"({stock[mine]:.2f} vs {100*float(stored[theirs]):.2f}).")

    # Threshold selection. Three paths, deliberately kept apart:
    #   * prior=test, no smoothing -- the exact sweep over every distinct score. This is
    #     the path the published prevalence numbers came from; it is left untouched.
    #   * prior=valid, no smoothing -- same exact sweep, but on the VALIDATION curve, so
    #     no target-house information is used.
    #   * smoothing on -- an exhaustive sweep is intractable because the duration rules
    #     must be re-applied at every candidate, so a 120-point quantile grid is used.
    if postprocess:
        dur = durations_for(appliance)
        src_scores = yhat if prior == "test" else np.asarray(log["valid_metrics_yhat"]).ravel()
        rate = float(y_test.mean()) if prior == "test" else float(y_valid.mean())
        t_star = _pick_threshold(src_scores, rate, floor, dur)
        pred = postprocess_state((yhat > t_star).astype(np.int8), **dur)
    else:
        if prior == "test":
            t_star, _ = curve.best("PRIOR")
        else:
            yh_v = np.asarray(log["valid_metrics_yhat"]).ravel()
            if yh_v.size != y_valid.size:
                raise SystemExit(
                    f"{dataset}/{appliance}/ws{ws}: validation predictions are "
                    f"{yh_v.size} points but the rebuilt ground truth is "
                    f"{y_valid.size}; refusing to score.")
            t_star, _ = DetectionCurve(y_valid, yh_v, floor).best("PRIOR")
        pred = (yhat > t_star).astype(np.int8)

    m = _timestamp_metrics(y_test, pred)
    ev = event_scores(y_test, pred)
    tp = int(np.sum((pred == 1) & (y_test == 1)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))
    out = dict(stored)
    out.update({
        "F1_SCORE": m["F1"] / 100.0,
        "BALANCED_ACCURACY": m["BA"] / 100.0,
        "PRECISION": m["PREC"] / 100.0,
        "RECALL": m["REC"] / 100.0,
        "ACCURACY": (tp + tn) / float(y_test.size),
        # Activation-level F1. Reported as a fraction so the table's "%" formatting
        # applies uniformly with the timestamp metrics.
        "EVENT_F1": ev["EV_F1"] / 100.0,
        "EVENT_RATIO": ev["ev_ratio"],
    })
    return out


def collect_convergence(dataset, windows, seeds, cols, methods, index_path):
    """{(model, column): {epochs_to_best, time_to_best_s}} from the convergence index.

    Same pooling as the metric cells: mean over the selected window sizes and seeds for
    one appliance. best_epoch is stored 0-based, so a run whose first epoch was best
    trained for one epoch.
    """
    import pandas as pd
    if not Path(index_path).exists():
        raise SystemExit(f"{index_path} missing; run scripts.export_convergence_index")
    df = pd.read_csv(index_path)
    df = df[(df.dataset == dataset) & df.window_size.isin(windows)
            & df.seed.isin(seeds)]
    df = df[df.root.str.contains("-fix|archfix", regex=True)]
    df = df.assign(epochs_to_best=df.best_epoch + 1)
    out, n = {}, {}
    for model, _ in methods:
        for keys, header in cols:
            g = df[(df.model == model) & (df.appliance.isin(keys))]
            # The same physical run can appear under two appliance spellings on REDD;
            # one row per (window, seed) is what the mean should see.
            g = g.drop_duplicates(subset=["window_size", "seed"])
            out[(model, header)] = {
                "epochs_to_best": g.epochs_to_best.mean() if len(g) else np.nan,
                "time_to_best_s": g.time_to_best_s.mean() if len(g) else np.nan,
            }
            n[(model, header)] = len(g)
    return out, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="REDD", choices=["REDD", "UKDALE"])
    p.add_argument("--windows", nargs="+", default=["256", "512"],
                   help="window sizes, or the single token 'dynamic' to use each "
                        "appliance's own window from results/window_selection.json "
                        "(the same file scripts/build_tod_joblist.sh sweeps at)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sr", default="10s")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--preset", choices=["sota", "ablation", "tod"], default="sota")
    p.add_argument("--with-convergence", action="store_true",
                   help="append Epochs and Time (s) blocks from the convergence index")
    p.add_argument("--conv-index", default="results/convergence_index.csv")
    # Without this the retrained roots win the dedup, so a UK-DALE table would take
    # ws512 from result-ukdale-fix (100-epoch recipe) and ws256/128 from the legacy
    # 3-epoch roots -- two recipes inside one half of one table, with nothing on screen
    # to say so. Pass the roots explicitly whenever a single recipe is required.
    p.add_argument("--roots", nargs="+", default=None,
                   help="override the result-directory search order")
    # Prevalence matching is the default for the exploratory `tod` preset only. The
    # `sota` and `ablation` presets keep the stored scoring, because those tables are
    # the published ones and silently re-scoring them would change numbers already in
    # the paper without anything on screen to say so. Pass --score explicitly to mix.
    p.add_argument("--score", choices=["stored", "prevalence"], default=None,
                   help="stored = the run's own min_threshold rule; prevalence = "
                        "threshold matched to the appliance duty cycle (default for "
                        "--preset tod)")
    p.add_argument("--postprocess", action="store_true",
                   help="apply the appliance's Kelly duration rules to predictions "
                        "(src/helpers/postprocess.py) and re-tune the threshold under "
                        "them. The ground-truth states satisfy these rules by "
                        "construction, so a shorter predicted run cannot match any "
                        "true activation. Only valid with --score prevalence.")
    p.add_argument("--prior", choices=["test", "valid"], default="test",
                   help="source of the duty cycle the threshold is matched to. "
                        "'test' assumes the target house's duty cycle is known (one "
                        "scalar, not labels, but still target information); 'valid' "
                        "uses none. See result-redd-tod/README.md.")
    p.add_argument("--with-events", action="store_true",
                   help="add an activation-level F1 block. Timestamp metrics cannot "
                        "see fragmentation: arms emitting 40+ predicted activations "
                        "per true one still score 45+ timestamp F1.")
    p.add_argument("--allow-uneven", action="store_true",
                   help="emit the table even if cells have different run counts")
    a = p.parse_args()

    global METHODS, ROOTS
    METHODS = {"sota": SOTA_METHODS, "ablation": ABLATION_METHODS,
               "tod": TOD_METHODS}[a.preset]
    if a.roots:
        ROOTS = list(a.roots)
    print(f"% roots: {' '.join(ROOTS)}", file=sys.stderr)
    score = a.score or ("prevalence" if a.preset == "tod" else "stored")
    print(f"% scoring: {score}"
          + (" (detection threshold matched to appliance duty cycle; "
             "regression metrics unchanged)" if score == "prevalence" else
             " (run's own min_threshold rule)"), file=sys.stderr)
    if len(a.windows) == 1 and str(a.windows[0]).lower() == "dynamic":
        sel = json.load(open(WS_JSON))["appliances"]
        cols = REDD_COLS if a.dataset == "REDD" else UKDALE_COLS
        windows, missing = {}, []
        for keys, header in cols:
            hit = next((sel[k]["window"] for k in keys if k in sel), None)
            if hit is None:
                missing.append(header)
            windows[header] = [hit]
        if missing:
            print(f"\nREFUSING: {WS_JSON} has no window for {', '.join(missing)}; "
                  f"run scripts/pick_window.py first.", file=sys.stderr)
            return 1
        wlabel = ", ".join(f"{h}:{windows[h][0]}" for _, h in cols)
        nwin = 1
    else:
        windows = [int(w) for w in a.windows]
        wlabel = ", ".join(map(str, windows))
        nwin = len(windows)
    print(f"% windows: {wlabel}", file=sys.stderr)
    if a.postprocess and score != "prevalence":
        print("\nREFUSING: --postprocess requires --score prevalence (the threshold "
              "must be re-tuned under the smoothing).", file=sys.stderr)
        return 1
    global METRICS
    if a.with_events:
        METRICS = METRICS + [("EVENT_F1", "Event F1", True, "%", True)]
    print(f"% post-processing: {'Kelly duration rules' if a.postprocess else 'none'}"
          + (f"; duty-cycle prior from {a.prior!r}" if score == "prevalence" else ""),
          file=sys.stderr)
    data, nruns, headers = collect(a.dataset, windows, a.seeds, a.sr, score=score,
                                   postprocess=a.postprocess, prior=a.prior)
    expected = nwin * len(a.seeds)

    print(f"% coverage (runs per cell, expected {expected}):", file=sys.stderr)
    uneven = []
    for model, _ in METHODS:
        row = " ".join(f"{h}={nruns[(model, h)]}" for h in headers)
        print(f"%   {model:14s} {row}", file=sys.stderr)
        for h in headers:
            if nruns[(model, h)] != expected:
                uneven.append(f"{model}/{h}={nruns[(model, h)]}")
    if uneven and not a.allow_uneven:
        print(f"\nREFUSING: uneven coverage -> {', '.join(uneven)}\n"
              f"Those cells would average a different number of runs than the rest. "
              f"Re-run when the sweep finishes, or pass --allow-uneven.", file=sys.stderr)
        return 1

    ncol = len(headers)
    lines = [r"\begin{table}[!t]", r"\centering",
             rf"\caption{{Performance Comparison on {a.dataset} "
             rf"(mean over window sizes {wlabel} "
             rf"and {len(a.seeds)} seeds)}}",
             rf"\label{{table:performance_comparison_{a.dataset.lower()}}}",
             rf"\begin{{tabular}}{{c|c|{'c' * ncol}}}", r"\hline",
             rf"Metrics & Methods & {' & '.join(headers)} \\ \hline"]

    blocks = list(METRICS)
    conv, conv_n = {}, {}
    if a.with_convergence:
        cols = REDD_COLS if a.dataset == "REDD" else UKDALE_COLS
        flat_ws = (sorted({w for v in windows.values() for w in v})
                   if isinstance(windows, dict) else windows)
        conv, conv_n = collect_convergence(a.dataset, flat_ws, a.seeds, cols,
                                           METHODS, a.conv_index)
        blocks += CONV_METRICS
        print("% convergence coverage:", file=sys.stderr)
        for model, _ in METHODS:
            row = " ".join(f"{h}={conv_n[(model, h)]}" for h in headers)
            print(f"%   {model:16s} {row}", file=sys.stderr)

    for metric, mlabel, higher_better, unit, bold_best in blocks:
        src = conv if metric in {m for m, _, _, _, _ in CONV_METRICS} else data
        lines.append(rf"\multirow{{{len(METHODS)}}}{{*}}{{{mlabel}}} ")
        # Best per column, so the bolding is per appliance rather than per row.
        best = {}
        for h in headers:
            vals = [src[(m, h)][metric] for m, _ in METHODS]
            vals = [v for v in vals if np.isfinite(v)]
            if vals and bold_best:
                best[h] = max(vals) if higher_better else min(vals)
        for model, label in METHODS:
            cells = []
            for h in headers:
                v = src[(model, h)][metric]
                if not np.isfinite(v):
                    cells.append("--"); continue
                shown = v * 100 if unit == "%" else v
                if metric == "epochs_to_best":
                    txt = f"{shown:.1f}"
                elif metric == "time_to_best_s":
                    txt = f"{shown:.0f}"
                else:
                    txt = f"{shown:.2f}{r'\%' if unit == '%' else ''}"
                if h in best and abs(v - best[h]) < 1e-12:
                    txt = rf"\textbf{{{txt}}}"
                cells.append(txt)
            shade = (r"\rowcolor{gray!15} "
                     if (model == PROPOSED and METHODS is SOTA_METHODS) else "")
            lines.append(rf" {shade}& {label} & {' & '.join(cells)} \\")
        lines.append(r" \hline")

    lines += [r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    print(tex)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(tex + "\n")
        print(f"\n% wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
