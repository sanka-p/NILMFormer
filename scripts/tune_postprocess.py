"""Tune the temporal post-processing on VALIDATION instead of hardcoding Kelly's rules.

WHY
---
src/helpers/postprocess.py applies Kelly's `min_on_duration` / `min_off_duration`
straight from `preprocessing.appliance_param`. Those are LABEL-GENERATION parameters:
they were chosen to segment ground truth into activations, and nothing has ever checked
that they are the right filter for PREDICTIONS. Two reasons to doubt it --

  * the asymmetry argument only licenses `min_on`: a predicted run shorter than
    `min_on_duration` cannot match any true activation, so dropping it is free. There is
    no equivalent argument for `min_off`, and none at all for the strength of smoothing.
  * `median_filter` is implemented and tested in postprocess.py but **never applied
    anywhere** -- `durations_for()` returns only min_on/min_off, so every call so far has
    run with `median_k=0`.

So this searches (min_on, min_off, median_k) per cell on validation and measures what the
choice is worth on test.

THE HONEST PART
---------------
Selection uses validation only, so the whole pipeline stays label-free on the target --
the property established in the prior-estimation study (see README). But validation here
is a single unrepresentative house with very few activations (Microwave: 0.16% ON), so
tuning three hyperparameters on it can easily overfit. The script therefore reports the
same search run against TEST as an oracle, and the gap between them is the selection
regret -- the same framing used for the threshold criteria in calibrate_threshold.py.
A tuned setting that beats Kelly on test but loses to the oracle by more than it gains
is not a result, it is noise.

THRESHOLD INTERACTION
---------------------
Smoothing changes the predicted-positive rate, so a threshold chosen before it no longer
matches the prior it was chosen for. Threshold and post-processing are therefore fitted
ALTERNATELY to convergence rather than in one pass (ROUNDS below), each re-tuned under
the other, exactly as score_postprocess.py re-tunes the threshold under smoothing.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.tune_postprocess --root result-redd-tod
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from src.helpers.event_metrics import event_scores
from src.helpers.postprocess import (bridge_short_gaps, drop_short_runs,
                                     durations_for, median_filter, postprocess_state)
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,
                                         ground_truth_states)
from scripts.score_postprocess import SEEDS, _pick_threshold, _timestamp_metrics

#: Durations in SAMPLES (10 s each). The grid brackets Kelly's values by a wide margin
#: in both directions -- 0 (rule off) through 360 (1 h) -- because the question is
#: whether Kelly's choice is right, and a grid centred on it could not answer that.
GRID_MIN_ON = (0, 3, 6, 12, 30, 60, 120, 180, 360)
GRID_MIN_OFF = (0, 3, 6, 12, 30, 60, 120, 180)
#: Odd widths only (median_filter rejects even ones). 0 keeps the filter off, which is
#: what every result so far has used.
GRID_MEDIAN = (0, 5, 11, 21)
#: Alternating threshold/post-processing fits. Converges in 2 in practice; 4 is a cap.
ROUNDS = 4


def _select(y, state, how):
    if how == "evf1":
        return event_scores(y, state, 0.0, "coverage")["EV_F1"]
    return _timestamp_metrics(y, state)[{"f1": "F1", "ba": "BA"}[how]]


def _search_pp(y, base, how):
    """Best (min_on, min_off, median_k) for the hard states `base`, by `how` against `y`.

    Nested so each shared prefix is computed once: postprocess_state is
    median -> bridge -> drop, so the median filter runs |GRID_MEDIAN| times and the
    bridge |GRID_MEDIAN x GRID_MIN_OFF| times instead of once per full combination.
    """
    best, best_s = None, -np.inf
    for k in GRID_MEDIAN:
        med = median_filter(base, k)
        for mo in GRID_MIN_OFF:
            br = bridge_short_gaps(med, mo)
            for mi in GRID_MIN_ON:
                s = _select(y, drop_short_runs(br, mi), how)
                if s > best_s:
                    best_s, best = s, dict(min_on=mi, min_off=mo, median_k=k)
    return best, float(best_s)


def _fit(yh_src, y_src, rate, floor, how):
    """Alternate threshold and post-processing fits against `y_src`. Returns (thr, pp)."""
    pp = dict(min_on=0, min_off=0, median_k=0)
    thr = _pick_threshold(yh_src, rate, floor, None)
    for _ in range(ROUNDS):
        new_pp, _ = _search_pp(y_src, (yh_src > thr).astype(np.int8), how)
        new_thr = _pick_threshold(yh_src, rate, floor, new_pp)
        if new_pp == pp and new_thr == thr:
            break
        pp, thr = new_pp, new_thr
    return thr, pp


def _fmt(pp):
    return f"{pp['min_on']}/{pp['min_off']}/{pp['median_k']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arms", nargs="+",
                   default=["TCN_KL_scratch", "TCN_KL_cw3", "TCN_KL_act"])
    p.add_argument("--select", default="f1", choices=["f1", "ba", "evf1"],
                   help="validation criterion the post-processing is chosen by")
    p.add_argument("--no-oracle", action="store_true",
                   help="skip the test-side search (halves the runtime, loses the regret)")
    a = p.parse_args()

    rows = []
    for app, ws in DEFAULT_GRID.items():
        cfg = build_config("REDD", app, ws)
        y_valid, y_test, floor = ground_truth_states(cfg)
        rate = float(y_valid.mean())
        kelly = durations_for(app)
        kelly.setdefault("median_k", 0)
        print(f"\n=== {app} / ws{ws} ===  Kelly {_fmt(kelly)} (min_on/min_off/median_k, "
              f"10 s samples), selecting on validation {a.select.upper()}")
        print(f"{'arm':<18}{'Kelly':>10}{'F1':>7}{'evF1':>7} | {'tuned':>10}{'F1':>7}"
              f"{'evF1':>7} | {'oracle':>10}{'F1':>7} | {'gain':>6}{'regret':>8}")
        for arm in a.arms:
            acc = {}
            for seed in SEEDS:
                for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
                    f = os.path.join(a.root, f"REDD_{key}_10s", str(ws),
                                     f"{arm}_{seed}.pt")
                    if os.path.exists(f + ".done"):
                        break
                else:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                yh_v = np.asarray(log["valid_metrics_yhat"], dtype=np.float64).ravel()
                yh_t = np.asarray(log["test_metrics_yhat"], dtype=np.float64).ravel()
                if yh_v.size != y_valid.size or yh_t.size != y_test.size:
                    print(f"  {arm} seed {seed}: size mismatch, skipping", file=sys.stderr)
                    continue
                t0 = time.time()

                # -- Kelly baseline: threshold re-tuned under Kelly's own rules --------
                t_k = _pick_threshold(yh_v, rate, floor, kelly)
                s_k = postprocess_state((yh_t > t_k).astype(np.int8), **kelly)

                # -- tuned on validation, applied to test (label-free end to end) ------
                t_v, pp_v = _fit(yh_v, y_valid, rate, floor, a.select)
                s_v = postprocess_state((yh_t > t_v).astype(np.int8), **pp_v)

                r = dict(
                    F1k=_timestamp_metrics(y_test, s_k)["F1"],
                    evF1k=event_scores(y_test, s_k, 0.0, "coverage")["EV_F1"],
                    F1t=_timestamp_metrics(y_test, s_v)["F1"],
                    evF1t=event_scores(y_test, s_v, 0.0, "coverage")["EV_F1"],
                    on=pp_v["min_on"], off=pp_v["min_off"], med=pp_v["median_k"],
                )
                if not a.no_oracle:
                    # Same procedure, fitted against TEST. Upper bound on what any
                    # validation-side selection could have reached.
                    t_o, pp_o = _fit(yh_t, y_test, float(y_test.mean()), floor, a.select)
                    s_o = postprocess_state((yh_t > t_o).astype(np.int8), **pp_o)
                    r.update(F1o=_timestamp_metrics(y_test, s_o)["F1"],
                             on_o=pp_o["min_on"], off_o=pp_o["min_off"],
                             med_o=pp_o["median_k"])
                r["secs"] = time.time() - t0
                for k, v in r.items():
                    acc.setdefault(k, []).append(v)
            if not acc:
                continue
            m = {k: float(np.nanmean(v)) for k, v in acc.items()}
            # Report the MODAL tuned setting across seeds, not the mean of integers.
            md = {k: max(set(acc[k]), key=acc[k].count) for k in ("on", "off", "med")}
            pp_s = f"{md['on']}/{md['off']}/{md['med']}"
            if a.no_oracle:
                orc, f1o, reg = "-", float("nan"), float("nan")
            else:
                mo = {k: max(set(acc[k]), key=acc[k].count)
                      for k in ("on_o", "off_o", "med_o")}
                orc = f"{mo['on_o']}/{mo['off_o']}/{mo['med_o']}"
                f1o, reg = m["F1o"], m["F1o"] - m["F1t"]
            print(f"{arm:<18}{_fmt(kelly):>10}{m['F1k']:>7.1f}{m['evF1k']:>7.1f} | "
                  f"{pp_s:>10}{m['F1t']:>7.1f}{m['evF1t']:>7.1f} | "
                  f"{orc:>10}{f1o:>7.1f} | {m['F1t']-m['F1k']:>+6.1f}{reg:>+8.1f}")
            rows.append(dict(appliance=app, ws=ws, arm=arm, select=a.select,
                             kelly=_fmt(kelly), tuned=pp_s, oracle=orc, **m))

    if rows:
        out = os.path.join(a.root, f"postprocess_tuning_{a.select}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
        print("Settings are min_on/min_off/median_k in 10 s samples, modal over seeds.\n"
              "gain   = tuned-on-validation minus Kelly, both scored on test.\n"
              "regret = the same search fitted on TEST minus the validation-fitted one;\n"
              "         large regret means validation is too small to choose these three\n"
              "         numbers, not that Kelly was right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
