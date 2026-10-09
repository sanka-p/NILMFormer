#################################################################################################################
#
# @description : NILMFormer - Observation Sliding Window (OSW) construction
#
# Converts raw per-sample power time series into OSW-level blocks, which are the unit of
# measurement for the Aci / Afm / Apd metrics of
#
#     Welikala et al., "Incorporating Appliance Usage Patterns for Non-Intrusive Load
#     Monitoring and Load Forecasting", IEEE Transactions on Smart Grid, 2019.
#
# Those metrics score an appliance *combination* per OSW, never a single sample, so this
# module is the mandatory first stage: it turns (aggregate, per-appliance power) sample
# series into per-OSW block means and per-OSW combination labels.
#
# Deliberately depends on numpy + pandas only -- not on src.helpers.metrics or
# src.helpers.preprocessing, which pull in torch and scikit-learn. The pipeline therefore
# runs on plain arrays from any source (UK-DALE, REDD, Tracebase, ...).
#
#################################################################################################################

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ALL_OFF",
    "DEFAULT_OSW_LENGTH",
    "DEFAULT_ON_THRESHOLDS_W",
    "MAX_BITMASK_APPLIANCES",
    "OSWBlocks",
    "assert_same_label_space",
    "bitmask_to_indices",
    "bitmask_to_names",
    "build_osw",
    "build_osw_from_frame",
    "combination_cardinality",
    "combination_name",
    "concat_osw",
    "mean_power_threshold_rule",
    "names_to_bitmask",
    "states_to_bitmask",
]


# ========================================= Constants ========================================= #

#: Bitmask of the combination in which every appliance is OFF.
ALL_OFF = 0

#: Default number of consecutive samples per OSW (Welikala et al. use N = 10).
DEFAULT_OSW_LENGTH = 10

#: Bit 62 is the largest that stays inside a signed int64 without touching the sign bit.
MAX_BITMASK_APPLIANCES = 62

#: Default per-appliance ON thresholds in watts, applied to the *block mean* power.
#:
#: These values are a deliberate COPY of the ``use_status_from_kelly_paper=False`` branch of
#: ``UKDALE_DataBuilder.appliance_param`` (``src/helpers/preprocessing.py``, ~lines 441-448)
#: rather than an import, so this module stays free of the torch/sklearn import chain. If the
#: builder's values ever change, this dict must be updated by hand.
#:
#: The Kelly thresholds used elsewhere in the repo (kettle 2000, washing_machine 20,
#: dishwasher 10 W) are deliberately NOT the default here: they are calibrated to be paired
#: with the min_on/min_off/min_activation duration filtering of
#: ``UKDALE_DataBuilder._compute_status``, which a block-mean test does not apply. Used raw, a
#: 10 W dishwasher threshold would read ON from standby leakage alone.
DEFAULT_ON_THRESHOLDS_W = {
    "kettle": 500.0,
    "washing_machine": 300.0,
    "dishwasher": 300.0,
    "microwave": 200.0,
    "fridge": 50.0,
    # REDD channel names, same power levels.
    "Kettle": 500.0,
    "WashingMachine": 300.0,
    "WasherDryer": 300.0,
    "Dishwasher": 300.0,
    "Microwave": 200.0,
    "Fridge": 50.0,
}

_NAN_POLICIES = ("drop", "interpolate", "zero")
_PARTIAL_POLICIES = ("drop", "pad", "keep")

#: Fixed key order for ``OSWBlocks.drop_counts``. Reasons are attributed with this
#: precedence, so a block failing two checks is counted exactly once, under the earlier key.
_DROP_KEYS = (
    "partial_tail_blocks",
    "partial_tail_samples",
    "nan_aggregate",
    "nan_true",
    "nan_pred",
    "nan_unfillable",
    "dropped_total",
    "kept",
)

ThresholdFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ========================================= ON/OFF rule ========================================= #


def mean_power_threshold_rule(
    mean_power: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Default ON/OFF rule: an appliance is ON iff its block-mean power >= its threshold.

    Parameters
    ----------
    mean_power : ndarray, shape (n_osw, n_appliances)
        Block-mean active power in watts.
    thresholds : ndarray, shape (n_appliances,)
        Per-appliance ON threshold in watts, aligned to the appliance axis.

    Returns
    -------
    ndarray of bool, shape (n_osw, n_appliances)

    Notes
    -----
    A custom rule passed as ``threshold_fn`` must honour the same contract: pure,
    shape-preserving, boolean output, and it only ever sees the block *means* -- never the
    raw sub-samples. That restriction is what makes the ground-truth and the predicted
    labelling provably symmetric.
    """
    return mean_power >= thresholds[None, :]


# ========================================= Label encoding ========================================= #
#
# The int64 bitmask is the CANONICAL combination encoding: bit i (LSB-first) corresponds to
# appliance_names[i]. It is compact, hashable, orderable, and directly comparable for Aci and
# groupable for Afm. The human-readable form produced by combination_name() is derived from it
# and exists for CSV/reporting only.
#
# Bit meaning depends on the appliance ORDER, so OSWBlocks carries appliance_names and
# assert_same_label_space() must be used before comparing labels from two different builds.


def states_to_bitmask(states: np.ndarray) -> np.ndarray:
    """Encode per-appliance ON/OFF states as int64 bitmasks.

    Parameters
    ----------
    states : ndarray of bool, shape (n, n_appliances)

    Returns
    -------
    ndarray of int64, shape (n,)
    """
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(f"states must be 2-D (n, n_appliances), got shape {states.shape}")
    n_app = states.shape[1]
    if n_app > MAX_BITMASK_APPLIANCES:
        raise ValueError(
            f"cannot bitmask-encode {n_app} appliances; the limit is "
            f"{MAX_BITMASK_APPLIANCES} (signed int64)"
        )
    # np.int64(1) << arange(int64) -- never `1 << i` on a platform-default int, which is
    # 32-bit on some platforms and would silently wrap past 31 appliances.
    weights = np.int64(1) << np.arange(n_app, dtype=np.int64)
    return states.astype(np.int64, copy=False) @ weights


def bitmask_to_indices(mask: int, n_appliances: int) -> tuple[int, ...]:
    """Return the ascending appliance indices that are ON in ``mask``."""
    mask = int(mask)
    return tuple(i for i in range(n_appliances) if mask >> i & 1)


def bitmask_to_names(mask: int, appliance_names: Sequence[str]) -> tuple[str, ...]:
    """Return the appliance names that are ON in ``mask``, in appliance order."""
    return tuple(appliance_names[i] for i in bitmask_to_indices(mask, len(appliance_names)))


def names_to_bitmask(names: Iterable[str], appliance_names: Sequence[str]) -> int:
    """Encode an iterable of ON appliance names as a bitmask."""
    lookup = {name: i for i, name in enumerate(appliance_names)}
    mask = 0
    for name in names:
        if name not in lookup:
            raise KeyError(f"unknown appliance {name!r}; known: {list(appliance_names)}")
        mask |= 1 << lookup[name]
    return mask


def combination_name(
    mask: int,
    appliance_names: Sequence[str],
    all_off_label: str = "OFF",
    sep: str = "+",
) -> str:
    """Human-readable combination label, e.g. ``"Fridge+Kettle"`` or ``"OFF"``."""
    names = bitmask_to_names(mask, appliance_names)
    return sep.join(names) if names else all_off_label


def combination_cardinality(mask):
    """Number of appliances ON -- accepts a scalar mask or an array of masks."""
    arr = np.asarray(mask, dtype=np.int64)
    if hasattr(np, "bitwise_count"):
        out = np.bitwise_count(arr)
    else:  # pragma: no cover - numpy < 2.0 fallback
        out = np.zeros(arr.shape, dtype=np.int64)
        work = arr.copy()
        while np.any(work):
            out += (work & 1).astype(np.int64)
            work >>= 1
    return int(out) if np.isscalar(mask) or arr.ndim == 0 else out.astype(np.int64)


# ========================================= Output container ========================================= #


@dataclass(frozen=True)
class OSWBlocks:
    """OSW-level products: one row per kept observation sliding window.

    All per-OSW arrays share the first axis, of length ``n_osw``.

    Attributes
    ----------
    aggregate_blocks : ndarray, shape (n_osw, osw_length)
        The raw N-sample aggregate block, exposed unchanged for downstream KLE-style
        feature extraction (the extraction itself is out of scope for this module).
    mean_true, mean_pred : ndarray, shape (n_osw, n_appliances)
        Per-appliance block-mean power in watts -- the paper's ``y_t(i)`` and ``yhat_t(i)``.
        ``mean_pred`` is None when no predictions were supplied.
    state_true, state_pred : ndarray of bool, shape (n_osw, n_appliances)
    label_true, label_pred : ndarray of int64, shape (n_osw,)
        Combination bitmasks. ``label_pred`` is None when no predictions were supplied.
    start_times : pd.DatetimeIndex or None
        First timestamp of each kept block.
    start_indices : ndarray of int64, shape (n_osw,)
        Row offset of each kept block into the input arrays.

    Metadata below governs cross-dataset comparability and is preserved for logging:
    an OSW of N=10 samples is ~10 s on 1 Hz Tracebase data but ~30 s on 1/3 Hz REDD
    whole-home data, so scores are only comparable at equal ``osw_duration_s``.
    """

    # --- per-OSW payload ---
    aggregate_blocks: np.ndarray
    mean_true: np.ndarray
    mean_pred: np.ndarray | None
    state_true: np.ndarray
    state_pred: np.ndarray | None
    label_true: np.ndarray
    label_pred: np.ndarray | None
    start_times: pd.DatetimeIndex | None
    start_indices: np.ndarray

    # --- metadata ---
    appliance_names: tuple[str, ...]
    thresholds_w: tuple[float, ...]
    osw_length: int
    sampling_interval_s: float
    osw_duration_s: float
    sampling_interval_source: str
    n_input_samples: int
    n_blocks_total: int
    n_contiguous_runs: int
    drop_counts: Mapping[str, int] = field(default_factory=dict)
    nan_policy: str = "drop"
    partial_policy: str = "drop"
    on_source: str = "threshold"

    # --- convenience ---

    @property
    def n_osw(self) -> int:
        return int(self.label_true.shape[0])

    @property
    def n_appliances(self) -> int:
        return len(self.appliance_names)

    @property
    def has_predictions(self) -> bool:
        return self.label_pred is not None

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(zip(self.appliance_names, self.thresholds_w))

    def active_mask(self) -> np.ndarray:
        """Boolean mask of OSWs whose *ground-truth* combination is not all-OFF."""
        return self.label_true != ALL_OFF

    def combination_names(self, which: str = "true") -> list[str]:
        labels = self.label_true if which == "true" else self.label_pred
        if labels is None:
            raise ValueError("no predicted labels available")
        return [combination_name(int(m), self.appliance_names) for m in labels]

    def metadata(self) -> dict:
        """Flat, CSV-friendly metadata dict."""
        drops = self.drop_counts
        return {
            "OSW_LENGTH": self.osw_length,
            "SAMPLING_INTERVAL_S": self.sampling_interval_s,
            "OSW_DURATION_S": self.osw_duration_s,
            "SAMPLING_INTERVAL_SOURCE": self.sampling_interval_source,
            "N_INPUT_SAMPLES": self.n_input_samples,
            "N_BLOCKS_TOTAL": self.n_blocks_total,
            "N_CONTIGUOUS_RUNS": self.n_contiguous_runs,
            "N_OSW": self.n_osw,
            "N_DROP_PARTIAL_BLOCKS": drops.get("partial_tail_blocks", 0),
            "N_DROP_PARTIAL_SAMPLES": drops.get("partial_tail_samples", 0),
            "N_DROP_NAN": (
                drops.get("nan_aggregate", 0)
                + drops.get("nan_true", 0)
                + drops.get("nan_pred", 0)
                + drops.get("nan_unfillable", 0)
            ),
            "N_DROP_TOTAL": drops.get("dropped_total", 0),
            "NAN_POLICY": self.nan_policy,
            "PARTIAL_POLICY": self.partial_policy,
            "ON_SOURCE": self.on_source,
            "N_APPLIANCES": self.n_appliances,
            "APPLIANCES": "|".join(self.appliance_names),
            "THRESHOLDS_W": "|".join(
                f"{n}={t:g}" for n, t in zip(self.appliance_names, self.thresholds_w)
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """One row per OSW -- for debugging and ad-hoc joins."""
        data = {
            "osw_idx": np.arange(self.n_osw),
            "start_index": self.start_indices,
            "label_true": self.label_true,
            "label_true_name": self.combination_names("true"),
        }
        if self.start_times is not None:
            data["start_time"] = self.start_times
        if self.has_predictions:
            data["label_pred"] = self.label_pred
            data["label_pred_name"] = self.combination_names("pred")
        for i, name in enumerate(self.appliance_names):
            data[f"y_true_{name}"] = self.mean_true[:, i]
            if self.mean_pred is not None:
                data[f"y_pred_{name}"] = self.mean_pred[:, i]
        with np.errstate(invalid="ignore"):
            data["agg_mean"] = np.nanmean(self.aggregate_blocks, axis=1)
            data["agg_max"] = np.nanmax(self.aggregate_blocks, axis=1)
        return pd.DataFrame(data)


def concat_osw(parts: Sequence[OSWBlocks]) -> OSWBlocks:
    """Concatenate OSW blocks built independently over disjoint stretches of time.

    The intended use is one build per house: blocks must never straddle a house boundary
    (two houses' timestamps are unrelated, and can even overlap), so each house is windowed
    on its own timeline and the resulting blocks are pooled here for scoring.

    All parts must share the appliance order, thresholds, ``osw_length`` and policies.
    ``start_indices`` are kept as-is and are therefore only meaningful within their own part;
    ``sampling_interval_s`` is taken from the first part after checking agreement.
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("nothing to concatenate")
    if len(parts) == 1:
        return parts[0]

    head = parts[0]
    for other in parts[1:]:
        assert_same_label_space(head, other)
        if other.osw_length != head.osw_length:
            raise ValueError(
                f"cannot concatenate OSW blocks with different osw_length "
                f"({head.osw_length} vs {other.osw_length})"
            )
        if other.thresholds_w != head.thresholds_w:
            raise ValueError(
                f"cannot concatenate OSW blocks with different thresholds "
                f"({head.thresholds_w} vs {other.thresholds_w})"
            )
        if not np.isclose(other.sampling_interval_s, head.sampling_interval_s):
            raise ValueError(
                f"cannot concatenate OSW blocks sampled at different intervals "
                f"({head.sampling_interval_s}s vs {other.sampling_interval_s}s)"
            )

    has_pred = all(p.has_predictions for p in parts)
    if any(p.has_predictions for p in parts) and not has_pred:
        raise ValueError("cannot concatenate: some parts carry predictions and some do not")
    has_times = all(p.start_times is not None for p in parts)

    def cat(attr):
        return np.concatenate([getattr(p, attr) for p in parts], axis=0)

    drops = dict.fromkeys(_DROP_KEYS, 0)
    for p in parts:
        for key, value in p.drop_counts.items():
            drops[key] = drops.get(key, 0) + value

    return OSWBlocks(
        aggregate_blocks=cat("aggregate_blocks"),
        mean_true=cat("mean_true"),
        mean_pred=cat("mean_pred") if has_pred else None,
        state_true=cat("state_true"),
        state_pred=cat("state_pred") if has_pred else None,
        label_true=cat("label_true"),
        label_pred=cat("label_pred") if has_pred else None,
        start_times=(
            pd.DatetimeIndex(np.concatenate([p.start_times.to_numpy(dtype="datetime64[ns]")
                                             for p in parts]))
            if has_times
            else None
        ),
        start_indices=cat("start_indices"),
        appliance_names=head.appliance_names,
        thresholds_w=head.thresholds_w,
        osw_length=head.osw_length,
        sampling_interval_s=head.sampling_interval_s,
        osw_duration_s=head.osw_duration_s,
        sampling_interval_source=head.sampling_interval_source,
        n_input_samples=sum(p.n_input_samples for p in parts),
        n_blocks_total=sum(p.n_blocks_total for p in parts),
        n_contiguous_runs=sum(p.n_contiguous_runs for p in parts),
        drop_counts=drops,
        nan_policy=head.nan_policy,
        partial_policy=head.partial_policy,
        on_source=head.on_source,
    )


def assert_same_label_space(a: OSWBlocks, b: OSWBlocks) -> None:
    """Raise unless two builds share an appliance order, hence a bitmask meaning."""
    if tuple(a.appliance_names) != tuple(b.appliance_names):
        raise ValueError(
            "combination bitmasks are not comparable: appliance order differs, "
            f"{list(a.appliance_names)} vs {list(b.appliance_names)}"
        )


# ========================================= Input resolution ========================================= #


def _as_2d(obj, name: str, appliance_names: Sequence[str] | None):
    """Coerce a DataFrame/ndarray to a C-contiguous float64 (T, M) array."""
    if isinstance(obj, pd.DataFrame):
        if appliance_names is not None:
            missing = [c for c in appliance_names if c not in obj.columns]
            if missing:
                raise KeyError(f"{name} is missing column(s) {missing}")
            obj = obj[list(appliance_names)]
        arr = obj.to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(obj, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D (T, n_appliances), got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_1d(obj, name: str) -> np.ndarray:
    arr = np.asarray(obj.to_numpy() if isinstance(obj, pd.Series) else obj, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (T,), got shape {np.shape(obj)}")
    return np.ascontiguousarray(arr)


def _resolve_appliance_names(appliance_names, y_true, n_app: int) -> tuple[str, ...]:
    if appliance_names is None:
        if isinstance(y_true, pd.DataFrame):
            appliance_names = list(y_true.columns)
        else:
            raise ValueError(
                "appliance_names is required unless y_true is a DataFrame whose column "
                "order defines the bitmask bit order"
            )
    names = tuple(str(n) for n in appliance_names)
    if not names:
        raise ValueError("appliance_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"appliance_names contains duplicates: {list(names)}")
    if len(names) != n_app:
        raise ValueError(
            f"appliance_names has {len(names)} entries but the power arrays have "
            f"{n_app} appliance columns"
        )
    if len(names) > MAX_BITMASK_APPLIANCES:
        raise ValueError(
            f"cannot bitmask-encode {len(names)} appliances; the limit is "
            f"{MAX_BITMASK_APPLIANCES} (signed int64)"
        )
    return names


def _resolve_thresholds(thresholds, names: Sequence[str]) -> np.ndarray:
    if thresholds is None:
        missing = [n for n in names if n not in DEFAULT_ON_THRESHOLDS_W]
        if missing:
            raise KeyError(
                f"no default ON threshold for appliance(s) {missing}; pass thresholds="
                "{name: watts, ...} explicitly (thresholds are per-appliance because "
                "appliance power ranges differ by orders of magnitude)"
            )
        values = [DEFAULT_ON_THRESHOLDS_W[n] for n in names]
    elif isinstance(thresholds, Mapping):
        missing = [n for n in names if n not in thresholds]
        if missing:
            raise KeyError(f"thresholds is missing entries for {missing}")
        values = [float(thresholds[n]) for n in names]
    else:
        values = [float(v) for v in thresholds]
        if len(values) != len(names):
            raise ValueError(
                f"thresholds has {len(values)} entries but there are {len(names)} appliances"
            )
    return np.asarray(values, dtype=np.float64)


def _validate_alignment(aggregate, y_true, y_pred, status_true, index) -> int:
    """Check every input is sample-aligned; return the common length T."""
    n = aggregate.shape[0]
    if y_true.shape[0] != n:
        raise ValueError(
            f"aggregate has {n} samples but y_true has {y_true.shape[0]} (axis 0); "
            "inputs must be sample-aligned"
        )
    if y_pred is not None and y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred has shape {y_pred.shape} but y_true has {y_true.shape}; "
            "predictions must be sample- and appliance-aligned with the ground truth"
        )
    if status_true is not None and status_true.shape != y_true.shape:
        raise ValueError(
            f"status_true has shape {status_true.shape} but y_true has {y_true.shape}"
        )
    if index is not None and len(index) != n:
        raise ValueError(f"index has length {len(index)} but aggregate has {n} samples")
    if n == 0:
        raise ValueError("inputs are empty (T == 0); nothing to window")
    return n


def _index_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps as int64 nanoseconds, independent of the index's own resolution.

    ``DatetimeIndex.asi8`` returns the index's *native* unit, which since pandas 3.0 is no
    longer always nanoseconds (``pd.date_range`` yields ``datetime64[us]``, a plain
    ``DatetimeIndex`` can be ``datetime64[s]``). Gap arithmetic must not depend on that, so
    everything is normalised through datetime64[ns] here. Integer nanosecond differencing is
    exact for any span these datasets cover.
    """
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _infer_sampling_interval(
    index: pd.DatetimeIndex | None, sampling_interval, tolerance: float
) -> tuple[float, str]:
    """Derive the sampling interval in seconds. Never hardcoded -- see module docstring."""
    explicit = None
    if sampling_interval is not None:
        if isinstance(sampling_interval, str):
            explicit = pd.Timedelta(pd.tseries.frequencies.to_offset(sampling_interval)) \
                .total_seconds()
        elif isinstance(sampling_interval, pd.Timedelta):
            explicit = sampling_interval.total_seconds()
        else:
            explicit = float(sampling_interval)
        if explicit <= 0:
            raise ValueError(f"sampling_interval must be positive, got {explicit}")

    derived = None
    if index is not None and len(index) > 1:
        diffs = np.diff(_index_ns(index))
        if np.any(diffs <= 0):
            first = int(np.argmax(diffs <= 0))
            raise ValueError(
                "index must be strictly increasing (no duplicate or out-of-order "
                f"timestamps); first violation at position {first + 1}: "
                f"{index[first]} -> {index[first + 1]}"
            )
        derived = float(np.median(diffs)) / 1e9

    if explicit is not None:
        if derived is not None and abs(derived - explicit) > explicit * tolerance:
            warnings.warn(
                f"explicit sampling_interval={explicit:g}s disagrees with the index median "
                f"diff of {derived:g}s; using the explicit value",
                UserWarning,
                stacklevel=3,
            )
        return explicit, "explicit"
    if derived is not None:
        return derived, "index_median_diff"
    raise ValueError(
        "cannot determine the sampling interval: pass sampling_interval=... (a pandas freq "
        "string such as '10s', a Timedelta, or seconds) or an index with >= 2 timestamps. "
        "The OSW duration is derived from the data and never assumed, because it governs "
        "cross-dataset comparability."
    )


def _contiguous_runs(
    index: pd.DatetimeIndex | None, n: int, interval_s: float, tolerance: float
) -> list[tuple[int, int]]:
    """Split the timeline into maximal runs of consecutively-sampled rows.

    Without this, a block of N consecutive array *rows* could span a real time gap of
    hours or days -- which is exactly what happens on a timeline built by inner-joining
    several independently-windowed per-appliance datasets.
    """
    if index is None or n < 2:
        return [(0, n)]
    gaps = np.diff(_index_ns(index))
    limit = interval_s * (1.0 + tolerance) * 1e9
    breaks = np.nonzero(gaps > limit)[0] + 1
    bounds = np.concatenate(([0], breaks, [n]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


# ========================================= Blocking ========================================= #


def _block_grid(
    runs: Sequence[tuple[int, int]], osw_length: int, partial_policy: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Lay blocks over each contiguous run.

    Returns ``(starts, lengths, partial_blocks, partial_samples)``. ``lengths`` equals
    ``osw_length`` everywhere except the trailing block of a run under the ``"pad"`` /
    ``"keep"`` policies, where it is the number of real samples available.
    """
    starts: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    partial_blocks = 0
    partial_samples = 0

    for run_start, run_stop in runs:
        run_len = run_stop - run_start
        n_full, remainder = divmod(run_len, osw_length)
        if n_full:
            full = run_start + np.arange(n_full, dtype=np.int64) * osw_length
            starts.append(full)
            lengths.append(np.full(n_full, osw_length, dtype=np.int64))
        if remainder:
            if partial_policy == "drop":
                partial_blocks += 1
                partial_samples += remainder
            else:  # "pad" and "keep" both materialise the short block
                starts.append(np.array([run_start + n_full * osw_length], dtype=np.int64))
                lengths.append(np.array([remainder], dtype=np.int64))

    if not starts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            partial_blocks,
            partial_samples,
        )
    order = np.concatenate(starts)
    sort = np.argsort(order, kind="stable")
    return order[sort], np.concatenate(lengths)[sort], partial_blocks, partial_samples


def _gather_blocks(arr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Gather ``arr`` at the (n_blocks, osw_length) row-index matrix ``rows``."""
    return arr[rows]


def _interpolate_block_column(col: np.ndarray, valid: np.ndarray) -> bool:
    """Fill NaNs inside one block/channel from that block's own valid samples.

    Interpolation never crosses a block boundary -- deliberate, because on an inner-joined
    timeline a boundary may span a real time gap. Returns False if the channel is entirely
    NaN inside the block and therefore unfillable.
    """
    bad = valid & np.isnan(col)
    if not bad.any():
        return True
    good = valid & ~np.isnan(col)
    if not good.any():
        return False
    x = np.arange(col.size, dtype=np.float64)
    # np.interp clamps outside the known range, giving nearest-valid at the block edges.
    col[bad] = np.interp(x[bad], x[good], col[good])
    return True


def _apply_nan_policy(
    blocks: dict[str, np.ndarray | None], valid: np.ndarray, policy: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Resolve NaNs per the documented policy; return per-reason bad-block masks.

    ``blocks`` maps a reason key ("aggregate" / "true" / "pred") to its block array, shaped
    (n_blocks, osw_length) or (n_blocks, osw_length, n_appliances). ``valid`` marks the
    in-use positions of each block (all True unless a short block was padded/kept).
    """
    n_blocks = valid.shape[0]
    bad = {key: np.zeros(n_blocks, dtype=bool) for key in ("aggregate", "true", "pred")}
    unfillable = np.zeros(n_blocks, dtype=bool)

    for key, arr in blocks.items():
        if arr is None:
            continue
        mask = valid if arr.ndim == 2 else valid[:, :, None]
        has_nan = np.isnan(arr) & mask
        if not has_nan.any():
            continue
        if policy == "drop":
            bad[key] = has_nan.reshape(n_blocks, -1).any(axis=1)
        elif policy == "zero":
            arr[has_nan] = 0.0
        else:  # "interpolate"
            rows = np.nonzero(has_nan.reshape(n_blocks, -1).any(axis=1))[0]
            for b in rows:
                if arr.ndim == 2:
                    if not _interpolate_block_column(arr[b], valid[b]):
                        unfillable[b] = True
                else:
                    for c in range(arr.shape[2]):
                        if not _interpolate_block_column(arr[b, :, c], valid[b]):
                            unfillable[b] = True
    bad["unfillable"] = unfillable
    return blocks, bad


def _block_means(
    arr: np.ndarray, valid: np.ndarray, denom: np.ndarray, treat_pad_as_zero: bool
) -> np.ndarray:
    """Mean over each block's samples, accumulated in float64.

    Padded positions contribute 0 to the sum. Under ``partial_policy="pad"`` the divisor is
    the full ``osw_length`` (so padding depresses the mean, as documented); under ``"keep"``
    it is the number of real samples.
    """
    mask = valid if arr.ndim == 2 else valid[:, :, None]
    filled = np.where(mask, arr, 0.0)
    total = filled.sum(axis=1, dtype=np.float64)
    div = denom if arr.ndim == 2 else denom[:, None]
    _ = treat_pad_as_zero  # divisor choice is already encoded in `denom`
    return total / div


# ========================================= Public builders ========================================= #


def build_osw(
    aggregate,
    y_true,
    y_pred=None,
    appliance_names: Sequence[str] | None = None,
    *,
    thresholds: Mapping[str, float] | Sequence[float] | None = None,
    threshold_fn: ThresholdFn = mean_power_threshold_rule,
    osw_length: int = DEFAULT_OSW_LENGTH,
    index: pd.DatetimeIndex | None = None,
    sampling_interval=None,
    nan_policy: str = "drop",
    partial_policy: str = "drop",
    status_true=None,
    contiguity_tolerance: float = 0.5,
) -> OSWBlocks:
    """Build OSW blocks from sample-aligned aggregate and per-appliance power series.

    An OSW is a fixed-length block of ``osw_length`` (default 10) consecutive samples of the
    AGGREGATE whole-house active power signal. The aggregate defines the block grid;
    appliance traces -- true and predicted -- are sliced to the identical spans and are never
    windowed independently. Blocks tile the timeline sequentially and NEVER overlap:
    ``[0:N)``, ``[N:2N)``, ``[2N:3N)``, ... Despite the name "sliding window", the stride
    equals the length: in this paper's evaluation loop each block is consumed once and
    advances state before the next is processed. Each OSW therefore yields exactly one
    ground-truth combination label and at most one predicted combination label.

    Parameters
    ----------
    aggregate : array-like, shape (T,)
        Whole-house active power in watts.
    y_true : array-like or DataFrame, shape (T, n_appliances)
        Measured per-appliance power in watts. Column order defines the bitmask bit order.
    y_pred : array-like, shape (T, n_appliances), optional
        Predicted per-appliance power in watts. When omitted the result carries ground truth
        only, which is the correct mode for dataset characterisation.
    appliance_names : sequence of str, optional
        Required unless ``y_true`` is a DataFrame.
    thresholds : mapping or sequence, optional
        Per-appliance ON threshold in watts. Defaults to :data:`DEFAULT_ON_THRESHOLDS_W`.
        Per-appliance rather than one global constant because appliance power ranges differ
        by orders of magnitude (a lamp versus an HVAC unit).
    threshold_fn : callable, optional
        ON/OFF rule, see :func:`mean_power_threshold_rule`. The paper does not specify the
        rule, so it is configurable.
    osw_length : int, optional
        N, the number of samples per OSW. Default 10.
    index : pd.DatetimeIndex, optional
        Per-sample timestamps. Supplying it enables gap-aware blocking (below) and lets the
        sampling interval be derived from the data.
    sampling_interval : str, Timedelta or float, optional
        Explicit sampling interval (pandas freq string, Timedelta, or seconds). Mandatory
        when ``index`` is None. If both are given the explicit value wins and a mismatch
        beyond ``contiguity_tolerance`` raises a UserWarning.
    nan_policy : {"drop", "interpolate", "zero"}, optional
        See Notes. Default "drop".
    partial_policy : {"drop", "pad", "keep"}, optional
        See Notes. Default "drop".
    status_true : array-like, shape (T, n_appliances), optional
        Alternative ground-truth ON/OFF source (e.g. the dataset's duration-filtered status
        channel). When supplied, the ground-truth state is majority-ON over each block
        instead of a block-mean threshold test; predictions still use ``threshold_fn``.
    contiguity_tolerance : float, optional
        A timestamp gap is a discontinuity when it exceeds
        ``sampling_interval_s * (1 + contiguity_tolerance)``. Default 0.5.

    Returns
    -------
    OSWBlocks

    Notes
    -----
    **Sampling interval and OSW duration.** ``sampling_interval_s`` is derived from the data
    (median of consecutive ``index`` differences) and never hardcoded; ``osw_duration_s`` is
    ``osw_length * sampling_interval_s``. At N=10 that is ~10 s for 1 Hz Tracebase data but
    ~30 s for 1/3 Hz REDD whole-home data, so both values are preserved on the result: OSW
    scores are only comparable across datasets at equal OSW duration. With neither an index
    nor an explicit interval, a ValueError is raised rather than a rate being assumed.

    **Gap-aware blocking.** When ``index`` is supplied the timeline is first split into
    maximal runs of consecutively-sampled rows; blocks are laid inside each run
    independently. A run of length L yields ``L // osw_length`` blocks and its remainder is
    handled by ``partial_policy``. This matters because the input is typically built by
    inner-joining several independently-windowed per-appliance datasets, so N consecutive
    array rows need not be N consecutive real-time samples -- without the split, a block
    could silently span days.

    **partial_policy** -- what to do with a run remainder shorter than ``osw_length``:

    - ``"drop"`` (default): discard it. Chosen as the default because a short block's mean is
      taken over fewer samples and its real duration is not ``N * dt``, which breaks the
      cross-dataset comparability the fixed duration exists to provide. Counted in
      ``drop_counts["partial_tail_blocks"]`` and ``["partial_tail_samples"]``.
    - ``"pad"``: right-pad the block to ``osw_length`` with 0.0 W in the aggregate and every
      appliance channel. Padding depresses the mean, so a padded block can flip an appliance
      to OFF; use only when tail coverage matters more than label fidelity.
    - ``"keep"``: keep the short block, taking means over the samples actually present. The
      unused tail of ``aggregate_blocks`` is filled with NaN so downstream feature code can
      detect it. ``osw_duration_s`` still reports ``N * dt``; these blocks are the documented
      exception.

    **nan_policy** -- how missing samples inside a block are handled. NaN never reaches a
    mean: ``np.nanmean`` is not used anywhere in this module.

    - ``"drop"`` (default): a block with at least one NaN in the aggregate, in any measured
      appliance channel, or in any predicted channel is dropped whole. This matches the
      repo's own window policy (``get_nilm_dataset`` drops any NaN-containing window,
      ``src/helpers/preprocessing.py``), keeping OSW counts comparable with the models' test
      windows.
    - ``"interpolate"``: linear interpolation within the block only, per channel, from that
      block's valid samples, clamping to nearest-valid at the edges. A channel that is
      entirely NaN inside a block is unfillable; that block is dropped and counted under
      ``drop_counts["nan_unfillable"]``.
    - ``"zero"``: NaN becomes 0.0 W (a missing submeter reading read as OFF, as in the
      builders' ``_fill_long_gaps_with_zero``). Biases block means downward.

    **Empty result.** If every block is dropped, a well-formed ``OSWBlocks`` with
    ``n_osw == 0`` and correctly shaped empty arrays is returned rather than an exception;
    ``drop_counts`` explains why.
    """
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    if partial_policy not in _PARTIAL_POLICIES:
        raise ValueError(
            f"partial_policy must be one of {_PARTIAL_POLICIES}, got {partial_policy!r}"
        )
    osw_length = int(osw_length)
    if osw_length < 1:
        raise ValueError(f"osw_length must be >= 1, got {osw_length}")

    # ---- 1. resolve and validate inputs ----
    if index is not None and not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if index is None and isinstance(aggregate, (pd.Series, pd.DataFrame)):
        if isinstance(aggregate.index, pd.DatetimeIndex):
            index = aggregate.index

    agg = _as_1d(aggregate, "aggregate")
    n_app_guess = (
        len(y_true.columns) if isinstance(y_true, pd.DataFrame) else np.shape(y_true)[-1]
    )
    names = _resolve_appliance_names(appliance_names, y_true, n_app_guess)
    y = _as_2d(y_true, "y_true", names)
    yhat = None if y_pred is None else _as_2d(y_pred, "y_pred", names)
    status = None if status_true is None else _as_2d(status_true, "status_true", names)
    n_samples = _validate_alignment(agg, y, yhat, status, index)
    thr = _resolve_thresholds(thresholds, names)

    # ---- 2. sampling interval (derived, never hardcoded) ----
    interval_s, interval_source = _infer_sampling_interval(
        index, sampling_interval, contiguity_tolerance
    )

    # ---- 3. gap-aware block grid ----
    runs = _contiguous_runs(index, n_samples, interval_s, contiguity_tolerance)
    starts, lengths, partial_blocks, partial_samples = _block_grid(
        runs, osw_length, partial_policy
    )
    n_blocks = int(starts.shape[0])

    drop_counts = dict.fromkeys(_DROP_KEYS, 0)
    drop_counts["partial_tail_blocks"] = partial_blocks
    drop_counts["partial_tail_samples"] = partial_samples

    if n_blocks == 0:
        return _empty_blocks(
            names, thr, osw_length, interval_s, interval_source, n_samples, 0,
            len(runs), drop_counts, nan_policy, partial_policy,
            "status" if status is not None else "threshold",
            index is not None, yhat is not None,
        )

    # ---- 4. gather blocks (fancy-index once; padded rows are clipped then masked) ----
    offsets = np.arange(osw_length, dtype=np.int64)[None, :]
    valid = offsets < lengths[:, None]
    rows = np.minimum(starts[:, None] + offsets, n_samples - 1)

    agg_b = _gather_blocks(agg, rows)
    y_b = _gather_blocks(y, rows)
    yhat_b = None if yhat is None else _gather_blocks(yhat, rows)
    status_b = None if status is None else _gather_blocks(status, rows)

    # ---- 5. NaN policy ----
    payload, bad = _apply_nan_policy(
        {"aggregate": agg_b, "true": y_b, "pred": yhat_b}, valid, nan_policy
    )
    agg_b, y_b, yhat_b = payload["aggregate"], payload["true"], payload["pred"]

    # ---- 6. block means, ON/OFF states, combination labels ----
    denom = lengths.astype(np.float64) if partial_policy == "keep" else np.float64(osw_length)
    denom = np.broadcast_to(np.asarray(denom, dtype=np.float64), (n_blocks,)).copy()
    mean_true = _block_means(y_b, valid, denom, partial_policy == "pad")
    mean_pred = (
        None if yhat_b is None else _block_means(yhat_b, valid, denom, partial_policy == "pad")
    )

    if status_b is not None:
        # Majority-ON over the block, using only the samples actually present.
        on_frac = _block_means(status_b, valid, denom, partial_policy == "pad")
        state_true = on_frac >= 0.5
        on_source = "status"
    else:
        state_true = np.asarray(threshold_fn(mean_true, thr), dtype=bool)
        on_source = "threshold"
    state_pred = (
        None if mean_pred is None else np.asarray(threshold_fn(mean_pred, thr), dtype=bool)
    )

    # ---- 7. attribute drops with fixed precedence, then filter once ----
    keep = np.ones(n_blocks, dtype=bool)
    for reason, key in (
        ("aggregate", "nan_aggregate"),
        ("true", "nan_true"),
        ("pred", "nan_pred"),
        ("unfillable", "nan_unfillable"),
    ):
        hit = bad[reason] & keep
        drop_counts[key] = int(hit.sum())
        keep &= ~hit
    drop_counts["dropped_total"] = int(n_blocks - keep.sum())
    drop_counts["kept"] = int(keep.sum())

    if partial_policy == "keep":
        # Documented: the unused tail is NaN so downstream feature code can detect it.
        agg_b = np.where(valid, agg_b, np.nan)

    label_true = states_to_bitmask(state_true)
    label_pred = None if state_pred is None else states_to_bitmask(state_pred)

    return OSWBlocks(
        aggregate_blocks=agg_b[keep],
        mean_true=mean_true[keep],
        mean_pred=None if mean_pred is None else mean_pred[keep],
        state_true=state_true[keep],
        state_pred=None if state_pred is None else state_pred[keep],
        label_true=label_true[keep],
        label_pred=None if label_pred is None else label_pred[keep],
        start_times=None if index is None else index[starts[keep]],
        start_indices=starts[keep],
        appliance_names=names,
        thresholds_w=tuple(float(t) for t in thr),
        osw_length=osw_length,
        sampling_interval_s=interval_s,
        osw_duration_s=osw_length * interval_s,
        sampling_interval_source=interval_source,
        n_input_samples=n_samples,
        n_blocks_total=n_blocks,
        n_contiguous_runs=len(runs),
        drop_counts=drop_counts,
        nan_policy=nan_policy,
        partial_policy=partial_policy,
        on_source=on_source,
    )


def _empty_blocks(
    names, thr, osw_length, interval_s, interval_source, n_samples, n_blocks, n_runs,
    drop_counts, nan_policy, partial_policy, on_source, has_index, has_pred,
) -> OSWBlocks:
    """A well-formed, correctly shaped zero-row result (never raise on total drop-out)."""
    m = len(names)
    empty_2d = np.empty((0, m), dtype=np.float64)
    return OSWBlocks(
        aggregate_blocks=np.empty((0, osw_length), dtype=np.float64),
        mean_true=empty_2d,
        mean_pred=empty_2d.copy() if has_pred else None,
        state_true=np.empty((0, m), dtype=bool),
        state_pred=np.empty((0, m), dtype=bool) if has_pred else None,
        label_true=np.empty(0, dtype=np.int64),
        label_pred=np.empty(0, dtype=np.int64) if has_pred else None,
        start_times=pd.DatetimeIndex([]) if has_index else None,
        start_indices=np.empty(0, dtype=np.int64),
        appliance_names=tuple(names),
        thresholds_w=tuple(float(t) for t in thr),
        osw_length=osw_length,
        sampling_interval_s=interval_s,
        osw_duration_s=osw_length * interval_s,
        sampling_interval_source=interval_source,
        n_input_samples=n_samples,
        n_blocks_total=n_blocks,
        n_contiguous_runs=n_runs,
        drop_counts=drop_counts,
        nan_policy=nan_policy,
        partial_policy=partial_policy,
        on_source=on_source,
    )


def build_osw_from_frame(
    frame: pd.DataFrame,
    appliance_names: Sequence[str],
    *,
    aggregate_col: str = "aggregate",
    true_suffix: str = "_true",
    pred_suffix: str = "_pred",
    status_suffix: str | None = None,
    require_predictions: bool = False,
    **kwargs,
) -> OSWBlocks:
    """Adapter: build OSW blocks from a wide DataFrame indexed by timestamp.

    Expects one aggregate column plus ``f"{appliance}{true_suffix}"`` and, optionally,
    ``f"{appliance}{pred_suffix}"`` columns. When the frame's index is a DatetimeIndex it is
    passed through, enabling gap-aware blocking. Every missing column is reported at once.
    """
    true_cols = [f"{a}{true_suffix}" for a in appliance_names]
    pred_cols = [f"{a}{pred_suffix}" for a in appliance_names]

    missing = [c for c in [aggregate_col, *true_cols] if c not in frame.columns]
    have_pred = all(c in frame.columns for c in pred_cols)
    if require_predictions and not have_pred:
        missing += [c for c in pred_cols if c not in frame.columns]
    if missing:
        raise KeyError(
            f"frame is missing column(s) {missing}; available: {list(frame.columns)}"
        )

    index = frame.index if isinstance(frame.index, pd.DatetimeIndex) else None
    status = None
    if status_suffix is not None:
        status_cols = [f"{a}{status_suffix}" for a in appliance_names]
        absent = [c for c in status_cols if c not in frame.columns]
        if absent:
            raise KeyError(f"frame is missing status column(s) {absent}")
        status = frame[status_cols].to_numpy(dtype=np.float64)

    return build_osw(
        frame[aggregate_col].to_numpy(dtype=np.float64),
        frame[true_cols].to_numpy(dtype=np.float64),
        frame[pred_cols].to_numpy(dtype=np.float64) if have_pred else None,
        appliance_names,
        index=index,
        status_true=status,
        **kwargs,
    )
