#################################################################################################################
#
# @description : Helpers for the TCN ablation arms (TCN / TCN_KL_scratch / TCN_KL_aug)
#
# The pretrained TCN_KL baseline was trained outside this repo on a *curated* UK-DALE
# activation database using *augmented* synthetic aggregate streams. This module supplies
# the two ingredients needed to reproduce those steps on this repo's own (non-curated)
# UK-DALE data, so the two can be ablated independently:
#
#   - fit_kl_basis        : port of KL_FIlter (ukdale_tcn_train/TCN/TCN_Better_Embeddings.py:348)
#   - OffCutoffSampler    : port of ukdale_tcn_train/On_Region_Sampler.py:142
#   - DeviceMultiplexer   : port of ukdale_tcn_train/On_Region_Sampler.py:378
#   - generate_stream     : port of ukdale_tcn_train/On_Region_Sampler.py:323
#
# Nothing here reads the curated database; activation segments are cut out of the status
# channel that UKDALE_DataBuilder already computes from raw UK-DALE.
#
#################################################################################################################

import hashlib
import itertools
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

#: Segments shorter than this are dropped, matching `if len(s) > 10` in TCN_ukdale.py:85.
MIN_SEGMENT_LEN = 11

#: Boxcar width used by OffCutoffSampler to decide what counts as an OFF region.
#: TCN_ukdale.py:144 uses 100 for everything except kettle, which uses 1 because a
#: 100-sample average over a short burst never clears the 2000 W threshold.
DEFAULT_OFF_WINDOW = 100
SHORT_BURST_OFF_WINDOW = 1

#: Real off context (in samples) kept on each side of an extracted activation, so crops
#: have somewhere to start and end. The curated segments were cut the same way.
DEFAULT_OFF_PAD = 20

#: Cap on activation segments kept per appliance. One OffCutoffSampler per segment holds
#: three float arrays the length of the padded signal, and fridge alone yields tens of
#: thousands of activations; this box is memory constrained.
MAX_SEGMENTS_PER_APPLIANCE = 3000

KL_ORDER = 10


# ============================ Karhunen-Loeve basis ============================ #


def fit_kl_basis(segments, order=KL_ORDER):
    """
    Fit the Karhunen-Loeve eigenvector basis from an ensemble of activation segments.

    Verbatim port of KL_FIlter.calculate_autocorrelation_of_ensemble /
    calculate_filters_from_R. Each segment is standardized, its unnormalized
    autocorrelation matrix accumulated over `order`-wide sliding windows, and the total
    divided by the number of SAMPLES (not windows) -- that is what the original does.

    Returns (eigen_values, filters) with columns in descending eigenvalue order.
    """
    R = np.zeros((order, order), dtype=np.float64)
    total_size = 0

    for signal in segments:
        if signal.size < order:
            continue
        std = np.std(signal)
        if not np.isfinite(std) or std == 0:
            # A flat segment carries no correlation structure and would divide by zero.
            continue
        z = (signal - np.mean(signal)) / std
        W = sliding_window_view(z, order)
        R += W.T @ W
        total_size += signal.size

    if total_size == 0:
        raise ValueError("fit_kl_basis: no usable segments (all too short or constant).")

    R /= total_size
    vals, vecs = np.linalg.eigh(R)
    return vals[::-1].copy(), vecs[:, ::-1].copy()


# ============================ Augmentation ============================ #


class OffCutoffSampler:
    """
    Port of On_Region_Sampler.OffCutoffSampler.

    Samples a contiguous slice of one activation segment whose two endpoints both fall in
    OFF regions and which contains at least one ON sample. The slice is real data, taken
    unmodified -- there is no noise, gain or time warping anywhere in this augmentation.
    """

    def __init__(self, signal, threshold, off_window):
        self.signal = np.pad(signal, off_window + 1)
        self.threshold = threshold
        self.off_window = off_window
        self.kernel = np.ones(off_window) / off_window
        self.convolved = np.convolve(self.signal, self.kernel, mode="same")
        self.likelihood = self.convolved < threshold

        n_off = np.sum(self.likelihood)
        self.usable = n_off > 0 and np.any(~self.likelihood)
        if not self.usable:
            return

        pdf = self.likelihood / n_off
        self.cdf = np.cumsum(pdf)
        self.points = np.arange(self.signal.size)
        self.unnormalized_on_cdf = np.cumsum(~self.likelihood)

    def sample_points(self, points):
        return np.round(np.interp(points, self.cdf, self.points)).astype(int)

    def sample_random_interval(self):
        for _ in range(100):
            a, b = np.random.uniform(size=2)
            a, b = min(a, b), max(a, b)
            a, b = self.sample_points(np.array([a, b]))
            if self.unnormalized_on_cdf[b] == self.unnormalized_on_cdf[a]:
                continue
            return self.signal[a:b]
        # Degenerate segment: fall back to the whole padded signal rather than spin.
        return self.signal


class DeviceMultiplexer:
    """Port of On_Region_Sampler.DeviceMultiplexer -- pick a random segment, crop it."""

    def __init__(self, generators):
        self.generators = generators

    def __call__(self):
        return self.generators[np.random.randint(len(self.generators))]()


def generate_stream(sample_generator, length, expected_active_percentage):
    """
    Port of On_Region_Sampler.generate_stream.

    Draws crops until `int(length * expected_active_percentage)` active samples are
    reached (right-trimming the last one), then interleaves them with the pieces of a
    single zero vector split at K-1 random points. Output length is exactly `length`.
    """
    target_active_length = int(length * expected_active_percentage)
    if target_active_length <= 0:
        return np.zeros(length)
    if target_active_length >= length:
        target_active_length = length - 1

    current_length = 0
    samples = []
    while current_length < target_active_length:
        new_sample = sample_generator()
        if new_sample.size == 0:
            continue
        current_length += new_sample.size
        if current_length > target_active_length:
            new_sample = new_sample[: -(current_length - target_active_length)]
        samples.append(new_sample)

    zeros = np.zeros(length - target_active_length)
    random_points = np.sort(np.random.randint(zeros.size, size=len(samples) - 1))
    zero_sectors = np.split(zeros, random_points)
    new_signal = np.concatenate(
        list(itertools.chain.from_iterable(zip(zero_sectors, samples)))
    )
    return new_signal[:length]


# ============================ Segment extraction ============================ #


def extract_activation_segments(
    data, appliance_names, off_pad=DEFAULT_OFF_PAD, min_len=MIN_SEGMENT_LEN,
    max_segments=MAX_SEGMENTS_PER_APPLIANCE, rng=None,
):
    """
    Cut activation segments out of a NILM array's status channels.

    `data` is [N, 1 + len(appliance_names), 2, L] in WATTS (unscaled), laid out as
    UKDALE_DataBuilder.get_nilm_dataset returns it: dim2 index 0 is power, 1 is the
    ON/OFF status already produced by _compute_status from raw UK-DALE.

    Returns (segments, on_fractions):
        segments      {appliance: [1-D float32 arrays]}  one per activation, with up to
                      `off_pad` samples of real off context on each side
        on_fractions  {appliance: float}  empirical duty cycle, used as the target active
                      fraction when generating synthetic streams
    """
    rng = rng if rng is not None else np.random
    segments = {}
    on_fractions = {}

    for j, app in enumerate(appliance_names):
        power = data[:, j + 1, 0, :]
        status = data[:, j + 1, 1, :] > 0

        on_fractions[app] = float(status.mean()) if status.size else 0.0

        found = []
        for i in range(power.shape[0]):
            s = status[i]
            if not s.any():
                continue
            # Contiguous ON runs within this window.
            edges = np.diff(s.astype(np.int8))
            starts = list(np.flatnonzero(edges == 1) + 1)
            ends = list(np.flatnonzero(edges == -1) + 1)
            if s[0]:
                starts.insert(0, 0)
            if s[-1]:
                ends.append(s.size)

            for a, b in zip(starts, ends):
                lo = max(0, a - off_pad)
                hi = min(s.size, b + off_pad)
                seg = power[i, lo:hi]
                if seg.size >= min_len:
                    found.append(np.asarray(seg, dtype=np.float32))

        if len(found) > max_segments:
            idx = rng.choice(len(found), size=max_segments, replace=False)
            found = [found[k] for k in idx]

        segments[app] = found
        logging.info(
            "TCN ablation: %s -> %d activation segments, duty cycle %.4f",
            app, len(found), on_fractions[app],
        )

    return segments, on_fractions


def _build_multiplexer(app, segs, threshold):
    """
    One OffCutoffSampler per segment, dropped if it exposes no ON sample.

    Short-burst appliances need off_window=1: a 100-sample boxcar over a two-sample
    kettle burst never clears 2000 W, which would silently discard every segment.
    Rather than hardcode the appliance list, pick from the median activation length and
    retry per segment on failure.
    """
    if not segs:
        return None

    median_len = float(np.median([s.size for s in segs]))
    off_window = (
        SHORT_BURST_OFF_WINDOW if median_len < DEFAULT_OFF_WINDOW else DEFAULT_OFF_WINDOW
    )

    samplers = []
    n_retry = 0
    for s in segs:
        smp = OffCutoffSampler(signal=s, threshold=threshold, off_window=off_window)
        if not smp.usable and off_window != SHORT_BURST_OFF_WINDOW:
            smp = OffCutoffSampler(
                signal=s, threshold=threshold, off_window=SHORT_BURST_OFF_WINDOW
            )
            n_retry += 1
        if smp.usable:
            samplers.append(smp)

    logging.info(
        "TCN ablation: %s -> %d/%d usable samplers (off_window=%d, %d retried at 1)",
        app, len(samplers), len(segs), off_window, n_retry,
    )
    if not samplers:
        return None
    return DeviceMultiplexer([s.sample_random_interval for s in samplers])


def _calibrate_active_fraction(mux, target_duty, threshold, n_probe=64):
    """
    Convert a real duty cycle into the `expected_active_percentage` generate_stream wants.

    generate_stream counts a crop's full LENGTH as active, but a crop is mostly the off
    padding around one activation, so asking for the measured duty cycle yields a stream
    several times too sparse. Probe the multiplexer for the mean ON fraction of a crop and
    scale by its reciprocal. The original sidestepped this with hand-tuned constants
    (TCN_ukdale.py:167 asks 0.7 for a fridge whose real duty cycle is far lower).
    """
    total = 0
    on = 0
    for _ in range(n_probe):
        crop = mux()
        total += crop.size
        on += int(np.count_nonzero(crop >= threshold))

    if total == 0 or on == 0:
        return float(np.clip(target_duty, 1e-3, 0.9))
    return float(np.clip(target_duty * total / on, 1e-3, 0.9))


def build_synthetic_windows(
    segments, on_fractions, appliance_names, target_appliance,
    appliance_param, compute_status, n_windows, window_size, seed=0,
):
    """
    Generate `n_windows` synthetic windows shaped like get_nilm_dataset's output.

    Per appliance an independent stream is generated by crop-and-remix, the streams are
    summed into a noiseless aggregate -- which is what `synth_aggregate_apps` also does
    for the real data (preprocessing.py:783) -- and the result is sliced into windows.
    The target appliance's own stream supplies the label, with its ON/OFF status
    recomputed by the same _compute_status the real pipeline uses.

    Returns [n_windows, 2, 2, window_size] float64 in WATTS.
    """
    np.random.seed(seed)

    total_len = n_windows * window_size
    streams = []
    for app in appliance_names:
        param = appliance_param.get(app, {})
        threshold = float(param.get("min_threshold", 50))
        mux = _build_multiplexer(app, segments.get(app, []), threshold)
        if mux is None:
            logging.warning(
                "TCN ablation: no usable segments for %s, contributing zeros.", app
            )
            streams.append(np.zeros(total_len))
            continue
        duty = float(on_fractions.get(app, 0.0))
        frac = _calibrate_active_fraction(mux, duty, threshold)
        logging.info(
            "TCN ablation: %s duty cycle %.4f -> active fraction %.4f", app, duty, frac
        )
        streams.append(generate_stream(mux, total_len, frac))

    streams = np.stack(streams, axis=0)
    aggregate = streams.sum(axis=0)

    k = appliance_names.index(target_appliance)
    target = streams[k]
    status = _status_for(target, appliance_param.get(target_appliance, {}), compute_status)

    out = np.empty((n_windows, 2, 2, window_size), dtype=np.float64)
    out[:, 0, 0, :] = aggregate.reshape(n_windows, window_size)
    out[:, 0, 1, :] = (out[:, 0, 0, :] > 0).astype(int)
    out[:, 1, 0, :] = target.reshape(n_windows, window_size)
    out[:, 1, 1, :] = status.reshape(n_windows, window_size)
    return out


def _status_for(power, param, compute_status):
    """ON/OFF labels for a generated stream, using the same rule as _get_dataframe."""
    lo = param.get("min_threshold", 0)
    hi = param.get("max_threshold", np.inf)
    initial = ((power >= lo) & (power <= hi)).astype(int)

    if "min_on_duration" not in param:
        return initial
    try:
        return compute_status(
            initial,
            param["min_on_duration"],
            param["min_off_duration"],
            param["min_activation_time"],
        )
    except (ValueError, IndexError):
        # _compute_status assumes a well-formed event sequence; fall back rather than
        # abort a run over a degenerate generated stream.
        logging.warning("TCN ablation: _compute_status failed, using raw threshold status.")
        return initial


# ============================ Curated activation source ============================ #
#
# The externally pretrained models were trained on a curated activation database instead
# of activations cut from raw UK-DALE. Loading those same activations here lets an in-repo
# arm run with curation as the only difference from TCN_KL_scratch.
#
# Three things must be reconciled with this pipeline:
#   - the curated store is on a 6 s grid, this pipeline runs at 10 s;
#   - a fraction of a percent of segments contain multi-hour gaps that must not be
#     interpolated across;
#   - curated segments carry only ~2 leading / 1 trailing sub-threshold samples, far less
#     off context than extract_activation_segments keeps, so they are padded to match.

CURATED_ROOT = "/new-home/e19/e19275/NILM/ukdale_tcn_train/ukdale/ukdale_curated"

#: One curated store per dataset. Both use the same hive layout, the same 6 s grid and the
#: same 2-leading / 1-trailing border padding, so only the root and the name map differ.
CURATED_ROOTS = {
    "UKDALE": CURATED_ROOT,
    "REDD": "/new-home/e19/e19275/NILM/NILM_datasets/redd_curated",
}

#: The curated stores spell some appliances with spaces, and REDD's repo-side names are
#: capitalised where UK-DALE's are not.
REPO_TO_CURATED_NAME_BY_DATASET = {
    "UKDALE": {
        "kettle": "kettle",
        "fridge": "fridge",
        "microwave": "microwave",
        "washing_machine": "washing machine",
        "dishwasher": "dish washer",
    },
    "REDD": {
        "Fridge": "fridge",
        "Microwave": "microwave",
        "Dishwasher": "dish washer",
        "WashingMachine": "washer dryer",
        "WasherDryer": "washer dryer",
    },
}

#: Back-compat alias; UK-DALE callers that predate the per-dataset split still work.
REPO_TO_CURATED_NAME = REPO_TO_CURATED_NAME_BY_DATASET["UKDALE"]

#: REDD exposes the combined washer/dryer under two repo names pointing at the same meter
#: (preprocessing.py:1258). `app` resolves to WasherDryer while synth_aggregate_apps lists
#: WashingMachine, so the target-stream lookup needs to treat them as one appliance.
SYNTH_APP_ALIASES = {
    "WasherDryer": "WashingMachine",
    "WashingMachine": "WasherDryer",
}


def resolve_synth_app(app, appliance_names):
    """The name under which `app` appears in `appliance_names`, honouring aliases."""
    if app in appliance_names:
        return app
    alias = SYNTH_APP_ALIASES.get(app)
    if alias and alias in appliance_names:
        return alias
    return None

#: Houses restricted to those this repo actually draws activations from, so the comparison
#: isolates extraction quality rather than appliance coverage. washing_machine is the
#: exception: the curated store has it in houses 2 and 4 only while the repo's activations
#: come from house 1, so it CANNOT be house-matched and its cell must be reported flagged.
CURATED_HOUSES_MATCHED = {
    "kettle": [1, 3, 5],
    "fridge": [1],
    "microwave": [1, 5],
    "dishwasher": [1, 5],
    "washing_machine": [4],  # not house-matched
}

#: HOUSE_SPLITS as used by the external training (TCN_ukdale.py:57). Only needed to
#: reproduce its `average_powers` as a fidelity check.
CURATED_HOUSES_EXTERNAL = {
    "kettle": [1, 3, 4, 5],
    "fridge": [1, 4, 5],
    "washing_machine": [4],
    "microwave": [1, 5],
    "dishwasher": [1, 5],
}

CURATED_DT_S = 6
CURATED_CACHE_DIR = "results/tcn_curated_cache"


def _curated_sample_dirs(root, curated_app, houses):
    """`sample=` directories for one appliance -- one activation each.

    Deliberately NOT `glob("sample=*/*.parquet")`: that descends into every sample
    directory from the calling thread, and at ~23 ms per directory on this network
    filesystem 53k of them cost ~21 min of serial stat-ing before a single byte is read.
    Listing only the `Index=` level is one cheap call (~0.05 s for 37k entries); the
    per-directory lookup then happens inside the thread pool alongside the read.
    """
    dirs = []
    for house in houses:
        base = os.path.join(root, f"Appliance={curated_app}", f"Index={house}")
        if not os.path.isdir(base):
            continue
        dirs.extend(
            os.path.join(base, d) for d in os.listdir(base) if d.startswith("sample=")
        )
    return sorted(dirs)


def _process_chunk(payload):
    """Scan one chunk of `sample=` dirs and return finished segments.

    Runs in a worker PROCESS. Reading tiny parquet files over this network filesystem
    costs ~70-85 ms each and polars already parallelises internally, so wrapping
    `read_parquet` in a thread pool made things *worse* (measured 197 ms/segment at 32
    threads vs 73 ms single-process). Processes each running one native multi-file scan
    is what actually scales: ~9.5 min for 53k segments at 8 processes.

    All per-segment work happens here so only finished float32 arrays cross the process
    boundary.
    """
    import polars as pl

    globs, sampling_rate, resample, split_gaps, min_len, off_pad = payload
    try:
        df = (
            pl.scan_parquet(globs, hive_partitioning=True)
            .select(["Time", "Power", "Index", "sample"])
            .sort("Time")
            # (Index, sample) -- `sample` ids restart per house, so grouping on `sample`
            # alone silently merges activations from different houses.
            .group_by(["Index", "sample"], maintain_order=True)
            .agg([pl.col("Time").alias("t"), pl.col("Power").alias("p")])
            .collect()
        )
    except Exception as exc:  # a chunk failing must not abort the whole scan
        return [], 0, 0, f"{type(exc).__name__}: {exc}"

    segs, n_split, n_short = [], 0, 0
    for times, powers in zip(df["t"].to_list(), df["p"].to_list()):
        times = np.asarray(times, dtype="datetime64[ns]")  # tz dropped; only diffs used
        powers = np.asarray(powers, dtype=np.float32)
        pieces = _split_on_gaps(times, powers) if split_gaps else [(times, powers)]
        n_split += len(pieces) - 1
        for t, q in pieces:
            # Apply the length filter on the NATIVE grid, before resampling. The external
            # loader drops `len <= 10` at 6 s; applying the same number after a 6->10 s
            # resample is 1.67x stricter in time and silently discarded 73% of microwave
            # and 25% of kettle activations -- exactly the short-burst appliances -- which
            # would bias those arms toward long events. Post-resample only a tiny floor is
            # needed so downstream array ops stay well defined.
            if q.size < min_len:
                n_short += 1
                continue
            if resample:
                q = _resample_segment(t, q, sampling_rate)
                if q.size < 2:
                    n_short += 1
                    continue
            segs.append(np.pad(np.asarray(q, dtype=np.float32), off_pad)
                        if off_pad else np.asarray(q, dtype=np.float32))
    return segs, n_split, n_short, None


def _scan_parallel(sample_dirs, sampling_rate, resample, split_gaps, min_len, off_pad,
                   workers=8, chunk_size=None):
    """Fan `sample=` dirs across worker processes; concatenate their segments."""
    if not sample_dirs:
        return [], 0, 0
    if chunk_size is None:
        # Several chunks per worker so small appliances parallelise too and large ones
        # stay load-balanced; a fixed 750 put all 691 dishwasher dirs in one process.
        chunk_size = max(25, -(-len(sample_dirs) // (workers * 4)))
    chunks = [sample_dirs[i:i + chunk_size] for i in range(0, len(sample_dirs), chunk_size)]
    payloads = [
        ([f"{d}/*.parquet" for d in c], sampling_rate, resample, split_gaps, min_len, off_pad)
        for c in chunks
    ]
    segs, n_split, n_short = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for got, sp, sh, err in pool.map(_process_chunk, payloads):
            if err:
                logging.warning("Curated: chunk failed (%s)", err)
                continue
            segs.extend(got)
            n_split += sp
            n_short += sh
    return segs, n_split, n_short


def _split_on_gaps(times, powers, dt_s=CURATED_DT_S):
    """Split an activation wherever sample spacing exceeds the native interval.

    A fraction of a percent of curated segments jump by hours; resampling across such a
    jump would fabricate a long ramp between two unrelated events.
    """
    if times.size < 2:
        return [(times, powers)]
    gaps = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    cuts = np.flatnonzero(gaps > dt_s) + 1
    if cuts.size == 0:
        return [(times, powers)]
    return [
        (t, p) for t, p in zip(np.split(times, cuts), np.split(powers, cuts)) if t.size
    ]


def _resample_segment(times, powers, sampling_rate):
    """6 s -> sampling_rate via the same operation the repo applies to raw UK-DALE
    (preprocessing.py:736). 6 s samples into 10 s bins leave every bin populated, so no
    NaN holes appear."""
    s = pd.Series(powers, index=pd.DatetimeIndex(times)).resample(sampling_rate).mean()
    return s.to_numpy().astype(np.float32)


def load_curated_segments(
    appliances, sampling_rate="10s", off_pad=DEFAULT_OFF_PAD, min_len=MIN_SEGMENT_LEN,
    max_segments=None, seed=0, dataset="UKDALE", root=None, houses=None,
    resample=True, split_gaps=True, cache_dir=CURATED_CACHE_DIR, workers=8,
):
    """
    Activation segments from the curated store, shaped exactly like the output of
    `extract_activation_segments` so the two are interchangeable.

    Returns {repo appliance name: [1-D float32 arrays]}.

    `max_segments=None` means no cap. The cap, when set, is applied to the FILE LIST
    before any I/O -- the store is 66k tiny files on a network filesystem, so reading only
    what survives the cap is what keeps this to minutes instead of ~75 min.
    """
    root = root or CURATED_ROOTS[dataset]
    name_map = REPO_TO_CURATED_NAME_BY_DATASET[dataset]
    houses = houses if houses is not None else CURATED_HOUSES_MATCHED
    key = "|".join([
        "v2",  # filter-before-resample
        dataset,
        ",".join(sorted(appliances)), str(sampling_rate if resample else "raw"),
        str(off_pad), str(min_len),
        # A dict cap is applied in memory AFTER loading, so it must not change the key --
        # otherwise every sweep point re-scans tens of thousands of files to thin one
        # appliance. Only a scalar cap changes what is read from disk.
        # A dict cap scans UNCAPPED and thins afterwards, so its cached payload is
        # byte-identical to the uncapped one -- share the entry rather than paying a
        # second full scan for it.
        "None" if isinstance(max_segments, dict) else str(max_segments),
        str(seed) if (max_segments and not isinstance(max_segments, dict)) else "noseed",  # uncapped result is seed-independent
        str(split_gaps),
        ";".join(f"{a}:{houses.get(a, [])}" for a in sorted(appliances)),
    ])
    cache_file = os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest() + ".pkl")
    if os.path.isfile(cache_file):
        logging.info("Curated segments: cache hit (%s)", cache_file)
        with open(cache_file, "rb") as f:
            return _apply_dict_cap(pickle.load(f), max_segments, seed)

    rng = np.random.default_rng(seed)
    out = {}
    for app in appliances:
        curated_app = name_map.get(app)
        if curated_app is None:
            logging.warning("Curated segments: no name mapping for '%s'.", app)
            out[app] = []
            continue

        paths = _curated_sample_dirs(root, curated_app, houses.get(app, []))
        n_avail = len(paths)
        # `max_segments` is either a scalar cap for every appliance or a per-appliance
        # dict (used by the pool-size sweep, which thins only the target appliance so the
        # synthetic aggregate's other channels stay at full realism).
        cap = None if isinstance(max_segments, dict) else max_segments
        if cap and n_avail > cap:
            idx = sorted(rng.choice(n_avail, size=cap, replace=False))
            paths = [paths[i] for i in idx]

        segs, n_split, n_short = _scan_parallel(
            paths, sampling_rate, resample, split_gaps, min_len, off_pad, workers=workers,
        )

        out[app] = segs
        logging.info(
            "Curated: %-16s houses=%-12s %6d/%-6d files -> %6d segments "
            "(%d gap-splits, %d too short)",
            app, str(houses.get(app, [])), len(paths), n_avail, len(segs), n_split, n_short,
        )

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(out, f)
    return _apply_dict_cap(out, max_segments, seed)


def _apply_dict_cap(segs, max_segments, seed):
    """Thin individual appliances in memory (the pool-size sweep).

    Kept out of the cached payload so one uncapped scan serves every sweep point; the
    subsample is seeded, so different seeds draw different subsets -- which is the
    variance a pool-size claim needs to report.
    """
    if not isinstance(max_segments, dict):
        return segs
    rng = np.random.default_rng(seed)
    out = dict(segs)
    for app, cap in max_segments.items():
        pool = out.get(app) or []
        if cap and len(pool) > cap:
            idx = sorted(rng.choice(len(pool), size=cap, replace=False))
            out[app] = [pool[i] for i in idx]
            logging.info(
                "Curated: %s subsampled %d -> %d activations (pool-size sweep).",
                app, len(pool), cap,
            )
    return out
