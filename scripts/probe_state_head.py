"""Is the multi-task state head informative, or did it collapse?

The p>0.5 diagnostic logged by the trainer is near-useless at 1-4% prevalence: a
well-calibrated head on a rare class sits almost entirely below 0.5. So judge the head
on RANKING (AUC) and on prevalence-matched F1, exactly as the power head is judged,
and compare the two heads on the same footing.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.probe_state_head \
      --root result-redd-tod --arms TCN_KL_dice --apps WashingMachine
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.helpers.postprocess import durations_for, postprocess_state
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,
                                         ground_truth_states)

SEEDS = (0, 1, 2)


def _auc(score, label):
    """Rank-based AUC; 0.5 means the score carries no information about `label`."""
    order = np.argsort(score, kind="stable")
    rank = np.empty(score.size, dtype=np.float64)
    rank[order] = np.arange(score.size, dtype=np.float64)
    pos = int(label.sum())
    neg = score.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rank[label == 1].sum() - pos * (pos - 1) / 2.0) / (pos * neg))


def _f1(y, s):
    tp = int(np.sum((s == 1) & (y == 1)))
    fp = int(np.sum((s == 1) & (y == 0)))
    fn = int(np.sum((s == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 200.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _match_prior(src, rate):
    """Threshold on `src` that makes the predicted-positive rate equal `rate`."""
    k = int(round((1.0 - rate) * src.size))
    k = min(max(k, 0), src.size - 1)
    return float(np.partition(src, k)[k])


def _find(root, app, ws, arm, seed):
    for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
        f = os.path.join(root, f"REDD_{key}_10s", str(ws), f"{arm}_{seed}.pt")
        if os.path.exists(f + ".done"):
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arms", nargs="+", default=["TCN_KL_dice"])
    p.add_argument("--apps", nargs="+", default=list(DEFAULT_GRID))
    a = p.parse_args()

    for app in a.apps:
        ws = DEFAULT_GRID[app]
        cfg = build_config("REDD", app, ws)
        y_valid, y_test, floor = ground_truth_states(cfg)
        dur = durations_for(app)
        rate = float(y_valid.mean())
        print(f"\n=== {app} / ws{ws} ===  valid {100*rate:.2f}% / "
              f"test {100*y_test.mean():.2f}% ON, floor {floor:.0f} W")
        print(f"{'arm':<18}{'seed':>5}{'pAUC':>8}{'sAUC':>8}{'pF1':>7}{'sF1':>7}"
              f"{'pF1pp':>7}{'sF1pp':>7}{'pmax':>7}{'pstd':>8}{'>0.5':>9}")
        for arm in a.arms:
            for seed in SEEDS:
                f = _find(a.root, app, ws, arm, seed)
                if f is None:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                if "test_metrics_yhat_prob" not in log:
                    print(f"  {arm} seed{seed}: no state head in this run")
                    continue
                pv = np.asarray(log["valid_metrics_yhat_prob"]).ravel()
                pt = np.asarray(log["test_metrics_yhat_prob"]).ravel()
                wv = np.asarray(log["valid_metrics_yhat"]).ravel()
                wt = np.asarray(log["test_metrics_yhat"]).ravel()
                if pt.size != y_test.size or wt.size != y_test.size:
                    print(f"  {arm} seed{seed}: size mismatch, skipping")
                    continue
                # Both heads scored the SAME way: a label-free threshold read off
                # validation at the validation duty cycle, with and without Kelly's
                # duration rules. The power head's floor is the appliance threshold;
                # the state head is a probability, so its only floor is 0.
                tw = max(_match_prior(wv, rate), floor)
                tp_ = _match_prior(pv, rate)
                sw = (wt > tw).astype(np.int8)
                sp = (pt > tp_).astype(np.int8)
                row = (_auc(wt, y_test), _auc(pt, y_test), _f1(y_test, sw),
                       _f1(y_test, sp),
                       _f1(y_test, postprocess_state(sw.copy(), **dur)),
                       _f1(y_test, postprocess_state(sp.copy(), **dur)))
                print(f"{arm:<18}{seed:>5}{row[0]:>8.4f}{row[1]:>8.4f}"
                      f"{row[2]:>7.1f}{row[3]:>7.1f}{row[4]:>7.1f}{row[5]:>7.1f}"
                      f"{pt.max():>7.3f}{pt.std():>8.4f}{(pt>0.5).mean():>9.5f}")
    print("\npAUC/pF1 = power head; sAUC/sF1 = state head. AUC 0.5 = no information.")
    print("Thresholds are label-free (validation duty cycle); *pp adds Kelly durations.")


if __name__ == "__main__":
    main()
