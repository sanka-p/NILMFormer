"""Does prevalence matching beat simply tuning the threshold on validation?

Prevalence matching is described in prior-shift language (Saerens 2002, Elkan 2001),
which implies the target duty cycle carries information. But our validation and test
houses disagree on that duty cycle by 1.3x-7.3x, and the method still helps -- which
is equally consistent with a duller explanation: the stock threshold is simply badly
placed on the model's score distribution, and ANY sane recalibration beats it.

This settles it. Five rules, same stored predictions, same arms, no GPU:

  stock        t = the appliance threshold.  What the published tables use.
  prev_valid   rate-matched to the VALIDATION duty cycle.  Label-free wrt test.
  f1max_valid  the threshold maximising VALIDATION F1.     Label-free wrt test.
  prev_test    rate-matched to the TEST duty cycle.        ORACLE (one scalar).
  f1max_test   the threshold maximising TEST F1.           ORACLE ceiling.

The comparison that matters is prev_valid vs f1max_valid: both use exactly the same
information, so a gap between them is attributable to the criterion alone. If they
tie, prevalence matching has no special status and should be described plainly as
"threshold tuned on validation" rather than as a prior correction.

Every rule is re-fitted under post-processing for the *pp columns, because smoothing
changes the positive rate and a threshold chosen before it no longer matches.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.compare_threshold_rules \
      --root result-redd-tod --arms TCN_KL_scratch
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from src.helpers.postprocess import durations_for, postprocess_state
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,
                                         ground_truth_states)
from scripts.score_postprocess import (SEEDS, _grid, _pick_threshold,
                                       _timestamp_metrics)

RULES = ("stock", "prev_valid", "f1max_valid", "prev_test", "f1max_test")
ORACLE = {"prev_test", "f1max_test"}


def _pick_f1max(yh, y_state, floor, pp):
    """Threshold maximising F1 of `yh` against `y_state`, over the same grid."""
    best, best_f1 = float(floor), -1.0
    for t in _grid(yh, floor):
        s = (yh > t).astype(np.int8)
        if pp is not None:
            s = postprocess_state(s, **pp)
        f1 = _timestamp_metrics(y_state, s)["F1"]
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def _find(root, app, ws, arm, seed):
    for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
        f = os.path.join(root, f"REDD_{key}_10s", str(ws), f"{arm}_{seed}.pt")
        if os.path.exists(f + ".done"):
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arms", nargs="+", default=["TCN_KL_scratch"])
    p.add_argument("--apps", nargs="+", default=list(DEFAULT_GRID))
    p.add_argument("--out", default="result-redd-tod/threshold_rules.csv")
    a = p.parse_args()

    rows = []
    for arm in a.arms:
        print(f"\n################ {arm}")
        per_app = {}
        for app in a.apps:
            ws = DEFAULT_GRID[app]
            cfg = build_config("REDD", app, ws)
            y_valid, y_test, floor = ground_truth_states(cfg)
            dur = durations_for(app)
            r_valid, r_test = float(y_valid.mean()), float(y_test.mean())
            acc = {r: {"F1": [], "F1pp": []} for r in RULES}
            for seed in SEEDS:
                f = _find(a.root, app, ws, arm, seed)
                if f is None:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                yh_v = np.asarray(log["valid_metrics_yhat"]).ravel()
                yh_t = np.asarray(log["test_metrics_yhat"]).ravel()
                if yh_v.size != y_valid.size or yh_t.size != y_test.size:
                    continue
                for pp, col in ((None, "F1"), (dur, "F1pp")):
                    thr = {
                        "stock":       float(floor),
                        "prev_valid":  _pick_threshold(yh_v, r_valid, floor, pp),
                        "f1max_valid": _pick_f1max(yh_v, y_valid, floor, pp),
                        "prev_test":   _pick_threshold(yh_t, r_test, floor, pp),
                        "f1max_test":  _pick_f1max(yh_t, y_test, floor, pp),
                    }
                    for rule, t in thr.items():
                        s = (yh_t > t).astype(np.int8)
                        if pp is not None:
                            s = postprocess_state(s, **pp)
                        acc[rule][col].append(_timestamp_metrics(y_test, s)["F1"])
            if not acc["stock"]["F1"]:
                print(f"  {app}: no finished seeds, skipping")
                continue
            per_app[app] = {r: {c: float(np.mean(v)) for c, v in d.items()}
                            for r, d in acc.items()}
            print(f"\n=== {app} / ws{ws} ===  valid {100*r_valid:.2f}% / "
                  f"test {100*r_test:.2f}% ON, floor {floor:.0f} W, "
                  f"n_seeds={len(acc['stock']['F1'])}")
            print(f"{'rule':<14}{'F1':>8}{'F1pp':>8}")
            for rule in RULES:
                tag = "  (oracle)" if rule in ORACLE else ""
                print(f"{rule:<14}{per_app[app][rule]['F1']:>8.1f}"
                      f"{per_app[app][rule]['F1pp']:>8.1f}{tag}")
                rows.append(dict(arm=arm, appliance=app, rule=rule,
                                 f1=round(per_app[app][rule]["F1"], 2),
                                 f1pp=round(per_app[app][rule]["F1pp"], 2)))

        if per_app:
            print(f"\n=== {arm}: MEAN over {len(per_app)} appliances ===")
            print(f"{'rule':<14}{'F1':>8}{'F1pp':>8}")
            for rule in RULES:
                m = {c: float(np.mean([per_app[x][rule][c] for x in per_app]))
                     for c in ("F1", "F1pp")}
                tag = "  (oracle)" if rule in ORACLE else ""
                print(f"{rule:<14}{m['F1']:>8.1f}{m['F1pp']:>8.1f}{tag}")
                rows.append(dict(arm=arm, appliance="MEAN", rule=rule,
                                 f1=round(m["F1"], 2), f1pp=round(m["F1pp"], 2)))

    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "appliance", "rule", "f1", "f1pp"])
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out} ({len(rows)} rows)")
    print("\nprev_valid vs f1max_valid is the decisive pair: identical information, "
          "different criterion.\nprev_test / f1max_test use target-set information "
          "and are reported as bounds only.")


if __name__ == "__main__":
    main()
