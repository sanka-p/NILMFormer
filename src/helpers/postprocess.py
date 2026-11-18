"""Temporal post-processing for predicted ON/OFF sequences.

Nothing in this repo has ever enforced duration structure on a prediction. The
detector is a bare per-sample comparison (`y_hat > threshold`, src/helpers/trainer.py
:334-341), so it can emit a one-sample activation for a dishwasher whose ground-truth
status, by construction, can never be shorter than 180 samples (30 min).

That asymmetry is the point. `NILMDataset` derives the ground-truth status with Kelly's
rules (src/helpers/preprocessing.py:406-440): gaps shorter than `min_off_duration` are
bridged and activations shorter than `min_on_duration` are dropped. So a predicted ON
run shorter than `min_on_duration` cannot match ANY true activation -- it is a
guaranteed false positive, removable without looking at the labels. Applying the same
rules to predictions makes the two sequences members of the same family instead of
comparing a filtered signal against an unfiltered one.

Durations are in SAMPLES at the dataset's sampling rate (10 s here), matching
`appliance_param`. Per appliance at 10 s:

    fridge           min_on   6 (60 s)    min_off   1 (10 s)
    washing_machine  min_on 180 (30 min)  min_off  16 (160 s)
    microwave        min_on   1 (no-op)   min_off   3 (30 s)
    dishwasher       min_on 180 (30 min)  min_off 180 (30 min)

Order matters and follows Kelly: bridge short OFF gaps FIRST, then drop short ON runs.
Doing it the other way round would delete two adjacent short runs that, once joined,
would have been long enough to keep.
"""
from __future__ import annotations

import numpy as np

__all__ = ["runs", "bridge_short_gaps", "drop_short_runs", "median_filter",
           "postprocess_state", "APPLIANCE_DURATIONS"]

#: Kelly duration rules in SAMPLES, keyed by the appliance names used in
#: preprocessing.appliance_param. Mirrors preprocessing.py:406-440 -- keep in sync.
APPLIANCE_DURATIONS = {
    "kettle":          {"min_on": 1,   "min_off": 0},
    "fridge":          {"min_on": 6,   "min_off": 1},
    "washing_machine": {"min_on": 180, "min_off": 16},
    "microwave":       {"min_on": 1,   "min_off": 3},
    "dishwasher":      {"min_on": 180, "min_off": 180},
}

#: Directory/appliance spellings used across the repo -> appliance_param keys.
_ALIAS = {
    "WashingMachine": "washing_machine", "WasherDryer": "washing_machine",
    "Dishwasher": "dishwasher", "Microwave": "microwave", "Fridge": "fridge",
    "Kettle": "kettle",
}


def durations_for(appliance: str) -> dict:
    """Duration rules for an appliance written in any of the repo's spellings."""
    key = _ALIAS.get(appliance, appliance).lower()
    if key not in APPLIANCE_DURATIONS:
        raise KeyError(f"no duration rules for appliance {appliance!r} "
                       f"(known: {sorted(APPLIANCE_DURATIONS)})")
    return dict(APPLIANCE_DURATIONS[key])


def runs(state: np.ndarray, value: int = 1):
    """Half-open [start, end) spans where `state` equals `value`.

    Returns an (n, 2) int array. Empty input gives a (0, 2) array, so callers can
    index the result unconditionally.
    """
    s = np.asarray(state).astype(bool, copy=False)
    if value == 0:
        s = ~s
    if s.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    # Pad with False so a run touching either edge still produces one rising and one
    # falling edge; without the pad a sequence that is ON throughout yields neither.
    padded = np.concatenate(([False], s, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return np.stack([starts, ends], axis=1).astype(np.int64)


def _fill(out, spans, value):
    """`out[s:e] = value` for every half-open span, without a Python-level loop.

    Runs are disjoint and sorted, so a +1/-1 difference array marks exactly the covered
    positions after a cumulative sum. The loop this replaces was O(number of runs) in
    Python, and a fragmented prediction has tens of thousands of them -- which made the
    duration rules the bottleneck of any search over their parameters.
    """
    if spans.shape[0] == 0:
        return out
    d = np.zeros(out.size + 1, dtype=np.int64)
    np.add.at(d, spans[:, 0], 1)
    np.add.at(d, spans[:, 1], -1)
    out[np.cumsum(d[:-1]) > 0] = value
    return out


def bridge_short_gaps(state: np.ndarray, min_off: int) -> np.ndarray:
    """Fill OFF runs shorter than `min_off`. Leading/trailing gaps are left alone.

    An OFF run at either end is not a gap BETWEEN activations -- it is the recording
    starting or ending in the OFF state -- so filling it would invent an activation.
    """
    out = np.asarray(state).astype(np.int8, copy=True)
    if min_off is None or min_off <= 1 or out.size == 0:
        return out
    g = runs(out, value=0)
    # A gap touching either edge is not an interior gap, so it is never filled.
    interior = (g[:, 0] != 0) & (g[:, 1] != out.size)
    return _fill(out, g[interior & ((g[:, 1] - g[:, 0]) < min_off)], 1)


def drop_short_runs(state: np.ndarray, min_on: int) -> np.ndarray:
    """Delete ON runs shorter than `min_on`, including ones touching an edge.

    Unlike a gap, a short ON run at an edge is still a short activation: the signal is
    there and it is too brief to be real. Truncation by the window boundary is possible
    but rare, and keeping it would readmit exactly the spurious single-sample firings
    this function exists to remove.
    """
    out = np.asarray(state).astype(np.int8, copy=True)
    if min_on is None or min_on <= 1 or out.size == 0:
        return out
    r = runs(out, value=1)
    return _fill(out, r[(r[:, 1] - r[:, 0]) < min_on], 0)


def median_filter(state: np.ndarray, k: int) -> np.ndarray:
    """Odd-length running median. k <= 1 is a no-op.

    Edges are handled by edge-replication, so the filter never pulls in zeros that are
    not in the signal. Implemented with a strided view: k is small (3-11) and the
    sequences are ~3e5 long, so this stays well inside memory.
    """
    out = np.asarray(state).astype(np.int8, copy=False)
    if k is None or k <= 1 or out.size == 0:
        return out.astype(np.int8, copy=True)
    if k % 2 == 0:
        raise ValueError(f"median_filter needs an odd width, got {k}")
    pad = k // 2
    padded = np.pad(out, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, k)
    return (view.sum(axis=1) > (k // 2)).astype(np.int8)


def postprocess_state(state, min_on=0, min_off=0, median_k=0):
    """Median filter, then bridge short gaps, then drop short runs.

    The median filter runs first because it is a denoiser: it removes isolated
    single-sample flips that would otherwise fragment one activation into several,
    each of which might then survive or fail the duration rules arbitrarily.
    """
    out = median_filter(state, median_k)
    out = bridge_short_gaps(out, min_off)
    return drop_short_runs(out, min_on)
