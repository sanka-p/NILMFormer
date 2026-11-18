"""Measure temporal post-processing and event-level scoring on the stored predictions.

Two questions, one pass over the `.pt` files, no GPU and no retraining:

  1. How much of the residual over-firing left by prevalence-matched thresholding is
     isolated spurious firings that Kelly's own duration rules delete for free?
  2. What does the arm ranking look like at the ACTIVATION level, where the 25:1-to-85:1
     negative class no longer dominates?

The threshold is re-tuned on validation AFTER post-processing, not reused from the
unsmoothed calibration. Smoothing changes the predicted-positive rate, so a threshold
chosen before it no longer matches the prior it was chosen to match; reusing it would
confound the two interventions and understate the combination.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.score_postprocess --root result-redd-tod
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from src.helpers.event_metrics import event_scores
from src.helpers.postprocess import durations_for, postprocess_state
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,
                                         ground_truth_states)

SEEDS = (0, 1, 2)
#: Candidate thresholds are quantiles of the non-zero score distribution. A grid is used
#: rather than every distinct score because post-processing must be re-run at each
#: candidate, which an exhaustive sweep over ~3e5 distinct values makes intractable.
N_GRID = 120


def _grid(scores, floor):
    """Threshold candidates at or above `floor`, coarsest-to-finest by quantile."""
    s = scores[scores > floor]
    if s.size == 0:
        return np.array([floor], dtype=float)
    q = np.quantile(s, np.linspace(0.0, 1.0, N_GRID))
    return np.unique(np.concatenate(([floor], q)))


def _timestamp_metrics(y_state, pred_state):
    tp = int(np.sum((pred_state == 1) & (y_state == 1)))
    fp = int(np.sum((pred_state == 1) & (y_state == 0)))
    fn = int(np.sum((pred_state == 0) & (y_state == 1)))
    tn = int(np.sum((pred_state == 0) & (y_state == 0)))
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    rec = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    spec = 100.0 * tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return dict(F1=f1, BA=0.5 * (rec + spec), PREC=prec, REC=rec,
                over=(rec / prec) if prec > 0 else float("nan"))


def _pick_threshold(yh, target_rate, floor, pp):
    """Threshold whose (optionally post-processed) positive RATE matches `target_rate`.

    This is the PRIOR criterion from scripts/calibrate_threshold.py, re-implemented
    because the candidate must be post-processed before its rate is read.

    WHERE `target_rate` COMES FROM IS THE WHOLE METHODOLOGICAL QUESTION. Prevalence
    matching needs an estimate of the TARGET house's duty cycle. Two sources, and they
    are not equivalent on this data because the splits disagree by 1.3x-7.3x:

      * validation ("valid")  -- honest: uses no target-house information, but REDD's
        validation house is unrepresentative, so the threshold comes out conservative.
      * target ("test")       -- assumes the appliance's duty cycle at the deployment
        site is known. That is ONE SCALAR, not labels, and in the label-shift
        literature (Saerens et al. 2002; Elkan 2001) an estimate of the target prior is
        exactly what the correction requires. It is still target-set information and
        must be reported as an assumption, never as a label-free result.

    scripts/make_sota_table_direct.py's --score prevalence uses the TARGET prior.
    """
    best, best_err = floor, float("inf")
    for t in _grid(yh, floor):
        state = (yh > t).astype(np.int8)
        if pp is not None:
            state = postprocess_state(state, **pp)
        err = abs(state.mean() / target_rate - 1.0) if target_rate > 0 else float("inf")
        if err < best_err:
            best, best_err = float(t), err
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arms", nargs="+", default=[
        "TCN_KL_scratch", "TCN_KL_cw3", "TCN_KL_act_cw3", "TCN_KL_act",
        "TCN_KL_act_tod", "TCN_KLdct", "TCN_KLperapp"])
    p.add_argument("--criterion", default="coverage", choices=["coverage", "iou"])
    p.add_argument("--overlap", type=float, default=0.0,
                   help="minimum overlap for an event match (0 = any overlap)")
    p.add_argument("--prior", default="valid", choices=["valid", "test"],
                   help="where the target duty cycle comes from; 'test' assumes it is "
                        "known at deployment (what the sota table does), 'valid' uses "
                        "no target information. See _pick_threshold.")
    a = p.parse_args()

    rows = []
    for app, ws in DEFAULT_GRID.items():
        cfg = build_config("REDD", app, ws)
        y_valid, y_test, floor = ground_truth_states(cfg)
        dur = durations_for(app)
        print(f"\n=== {app} / ws{ws} ===  valid {100*y_valid.mean():.2f}% / "
              f"test {100*y_test.mean():.2f}% ON, prior from {a.prior!r}, "
              f"floor {floor:.0f} W, min_on {dur['min_on']} min_off {dur['min_off']} "
              f"samples (10 s each)")
        print(f"{'arm':<20}{'F1':>7}{'F1pp':>7}{'BA':>7}{'BApp':>7}"
              f"{'over':>7}{'overpp':>7}{'evF1':>7}{'evF1pp':>7}{'nPred/nTrue':>13}")
        for arm in a.arms:
            agg = {k: [] for k in ("F1", "F1pp", "BA", "BApp", "over", "overpp",
                                   "evF1", "evF1pp", "ratio", "ratiopp")}
            for seed in SEEDS:
                for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
                    f = os.path.join(a.root, f"REDD_{key}_10s", str(ws),
                                     f"{arm}_{seed}.pt")
                    if os.path.exists(f + ".done"):
                        break
                else:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                yh_v = np.asarray(log["valid_metrics_yhat"]).ravel()
                yh_t = np.asarray(log["test_metrics_yhat"]).ravel()
                if yh_v.size != y_valid.size or yh_t.size != y_test.size:
                    print(f"  {arm}: size mismatch, skipping", file=sys.stderr)
                    continue
                # The scores the threshold is READ OFF must come from the same split
                # as the prior it is matched to, or the rate it achieves is not the
                # rate it was chosen for.
                src = yh_v if a.prior == "valid" else yh_t
                rate = float(y_valid.mean()) if a.prior == "valid" else float(y_test.mean())
                # Baseline: prevalence-matched, no smoothing.
                t0 = _pick_threshold(src, rate, floor, None)
                s0 = (yh_t > t0).astype(np.int8)
                # With smoothing: threshold re-tuned under the SAME post-processing.
                t1 = _pick_threshold(src, rate, floor, dur)
                s1 = postprocess_state((yh_t > t1).astype(np.int8), **dur)

                m0, m1 = _timestamp_metrics(y_test, s0), _timestamp_metrics(y_test, s1)
                e0 = event_scores(y_test, s0, a.overlap, a.criterion)
                e1 = event_scores(y_test, s1, a.overlap, a.criterion)
                agg["F1"].append(m0["F1"]);      agg["F1pp"].append(m1["F1"])
                agg["BA"].append(m0["BA"]);      agg["BApp"].append(m1["BA"])
                agg["over"].append(m0["over"]);  agg["overpp"].append(m1["over"])
                agg["evF1"].append(e0["EV_F1"]); agg["evF1pp"].append(e1["EV_F1"])
                agg["ratio"].append(e0["ev_ratio"]); agg["ratiopp"].append(e1["ev_ratio"])
            if not agg["F1"]:
                continue
            m = {k: float(np.nanmean(v)) for k, v in agg.items()}
            print(f"{arm:<20}{m['F1']:>7.1f}{m['F1pp']:>7.1f}{m['BA']:>7.1f}"
                  f"{m['BApp']:>7.1f}{m['over']:>6.2f}x{m['overpp']:>6.2f}x"
                  f"{m['evF1']:>7.1f}{m['evF1pp']:>7.1f}"
                  f"{m['ratio']:>6.1f}/{m['ratiopp']:<6.1f}")
            rows.append(dict(appliance=app, ws=ws, arm=arm, **m))

    if rows:
        import csv
        out = os.path.join(a.root, "postprocess_analysis.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
        print("F1/BA/over = prevalence-matched threshold, no smoothing; *pp = Kelly "
              "duration rules applied with the threshold re-tuned on validation.\n"
              "evF1 = activation-level F1; nPred/nTrue = predicted events per true "
              "event (>1 means fragmentation or event-level over-firing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
