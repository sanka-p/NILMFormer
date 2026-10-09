#################################################################################################################
#
# @description : Synthesise a multi-appliance aggregate from the isolated DEEE_SmartHome traces.
#
# DEEE has no mains channel -- each trace is one appliance recorded alone -- so an aggregate
# has to be constructed before a disaggregation model has anything to disaggregate. The
# recorded sessions also barely overlap in wall-clock time (6.1 h of data spread over 4
# days), so replaying them at their true timestamps would give a signal that is almost
# always a single appliance against zeros: a trivial task that would flatter any model. The
# timeline is therefore synthesised.
#
# Two construction modes:
#
#   SESSION (primary)  Place WHOLE sessions at random offsets. Non-overlapping within a
#                      category (an appliance cannot run twice at once) and freely
#                      overlapping across categories, which is what creates the
#                      simultaneous-load interference that makes disaggregation hard.
#
#   REMIX (secondary)  Crop-and-remix through the existing OffCutoffSampler/generate_stream
#                      machinery in tcn_ablation.py.
#
# SESSION is primary for two independent reasons. First, washing_machine has
# min_on_duration = 180 (= 1800 s at 10 s), so any crop shorter than that is silently
# labelled OFF by _compute_status and crop-and-remix would destroy the washing-machine
# labels outright. Second, REMIX uses the same generator that produced the curated arm's
# TRAINING data, so a remixed test set structurally resembles what that model was trained
# on and flatters it. REMIX is kept as a robustness check: if it scores far above SESSION,
# that gap is the augmentation-matching bonus rather than a generalisation result.
#
# Both base-load variants are produced, because a noiseless sum of clean single-appliance
# traces has no standby draw, no unmetered load and no meter noise -- all of which make
# real disaggregation harder:
#
#   pure      the exact sum of measured traces, nothing invented
#   baseload  + BASELOAD_W constant + Gaussian noise of sigma = NOISE_SIGMA_W
#
# The injected values are written into the .npz and printed beside every number they
# produced, so a reader can see exactly what was added.
#
#################################################################################################################

import logging
from dataclasses import dataclass, field

import numpy as np

from src.helpers.deee import DEEE_CATEGORY_MAP, sessions_by_category
from src.helpers.tcn_ablation import (
    _build_multiplexer,
    _calibrate_active_fraction,
    _status_for,
    generate_stream,
)

#: Injected base load for the "baseload" variant. 80 W is a modest always-on floor for a
#: single dwelling (router, standby, clocks); sigma = 10 W is light meter noise. Both are
#: recorded in the output so the number is never implicit.
BASELOAD_W = 80.0
NOISE_SIGMA_W = 10.0
BASELOAD_VARIANTS = ("pure", "baseload")

#: Which DEEE categories enter the aggregate.
#:   "all"    all 12 categories -- the 9 with no UK-DALE counterpart (fan, iron, AC, hair
#:            dryer, blender, rice cooker, laptop, light bulb, air fryer) act as realistic
#:            out-of-vocabulary distractor load.
#:   "ukdale" only the 3 categories the UK-DALE models were trained to see.
#: The pair isolates how much of the models' false-positive behaviour is caused by loads
#: they have no vocabulary for, rather than by the target appliances themselves.
SCOPES = ("all", "ukdale")

#: The synthetic stream length must be a multiple of the largest window so 128/256/512 all
#: tile it exactly and the three window sizes score the IDENTICAL signal -- any metric
#: difference between them is then model behaviour, not a different test set.
LCM_WINDOW = 512
DEFAULT_LENGTH = 512 * 360  # 184,320 samples = ~21 days at 10 s

#: Sessions shorter than this are not viable OffCutoffSampler inputs (the sampler pads by
#: off_window+1 and needs an OFF region). The three Blender traces land at 4-6 samples
#: after resampling; in REMIX mode they are contributed verbatim as impulses instead.
MIN_REMIX_LEN = 11

#: MEASURED UK-DALE house-2 duty cycles, read off the OSW ground-truth cache
#: (results/osw_gt_cache/UKDALE_*_10s_w256_*.npz, fraction of timestamps with status == 1).
#: These are the priors the trained models expect, so the three scored appliances are placed
#: at exactly these rates: a test set with a wildly different ON rate would measure the
#: prior mismatch rather than the domain shift.
UKDALE_DUTY = {
    "kettle": 0.00590,
    "microwave": 0.00442,
    "washing_machine": 0.01119,
}

#: ASSUMED duty cycles for the nine DEEE categories with no UK-DALE counterpart. These are
#: not measured -- the recordings are bench captures, so each trace's own duty is ~1.0 and
#: says nothing about how often the appliance runs in a house. They are plausible values for
#: a Sri Lankan dwelling and exist only to make the distractor load realistic; they are
#: stated here rather than buried so a reader can see what was assumed and change it.
DISTRACTOR_DUTY = {
    "Fan": 0.25,
    "Light Bulb": 0.20,
    "Laptop": 0.30,
    "Air Conditioner": 0.10,
    "Rice Cooker": 0.02,
    "Iron": 0.01,
    "Air Fryer": 0.01,
    "Hair Dryer": 0.005,
    "Blender": 0.002,
}

#: The `adapted` threshold regime. Each entry overrides one UK-DALE appliance_param field,
#: with the reason, so the regime is auditable rather than tuned.
#:
#: kettle/min_threshold: UK-DALE uses 2000 W for a ~3 kW UK element; the DEEE kettles are
#:   230 V ~1.4 kW units peaking at 1370 and 1341 W, so the as-trained threshold labels
#:   every sample OFF. 500 W is the repo's own non-Kelly kettle constant
#:   (preprocessing.py:445), not an invented number.
#:
#: washing_machine/min_on_duration: UK-DALE requires 180 samples (1800 s) of continuous ON,
#:   which encodes a European hot-wash cycle. The Singer top-loader's above-threshold phases
#:   are 7 runs with a longest of 73 samples (730 s), separated by OFF gaps of up to 94
#:   samples, so NO run survives and the appliance yields zero activations as-trained. 18
#:   samples (180 s) matches the observed wash phases.
ADAPTED_OVERRIDES = {
    "kettle": {"min_threshold": 500},
    "washing_machine": {"min_on_duration": 18},
}


def build_regimes(appliance_param):
    """{regime: {appliance: param dict}} for the two threshold regimes.

    One source of truth: the same dict drives the ground-truth labels (via _compute_status)
    and `threshold_small_values` at evaluation, so the two cannot silently diverge.
    """
    apps = list(DEEE_CATEGORY_MAP.values())
    astrained = {a: dict(appliance_param[a]) for a in apps}
    adapted = {a: dict(appliance_param[a]) for a in apps}
    for app, overrides in ADAPTED_OVERRIDES.items():
        if app in adapted:
            adapted[app].update(overrides)
    return {"astrained": astrained, "adapted": adapted}


def target_duty(category):
    """The duty cycle a category is placed at, and whether it was measured or assumed."""
    app = DEEE_CATEGORY_MAP.get(category)
    if app and app in UKDALE_DUTY:
        return UKDALE_DUTY[app], "measured(UKDALE h2)"
    if category in DISTRACTOR_DUTY:
        return DISTRACTOR_DUTY[category], "assumed"
    return 0.01, "default"


@dataclass
class DEEEAggregate:
    """A synthesised aggregate plus everything needed to score and trace it."""

    aggregate: np.ndarray  #: (L,) float32 watts
    streams: dict  #: {category: (L,) float32 watts}
    status: dict  #: {appliance: {regime: (L,) int}}
    appliance_power: dict  #: {appliance: (L,) float32 watts} -- ground truth
    src_category_id: np.ndarray  #: (L,) int16, dominant contributor, -1 = none
    src_session_id: np.ndarray  #: (L,) int32, index into `session_names`, -1 = none
    session_names: list
    category_names: list
    mode: str
    variant: str
    scope: str
    seed: int
    baseload_w: float
    noise_sigma_w: float
    length: int
    n_source_samples: int  #: real resampled samples the stream was built from
    duty: dict = field(default_factory=dict)

    @property
    def reuse_factor(self):
        """How many times over the source corpus was reused to fill this stream."""
        return self.length / max(self.n_source_samples, 1)


def _empirical_duty(sessions):
    """Fraction of a category's own recorded samples that are ON (> 5 W)."""
    tot = sum(s.power.size for s in sessions)
    on = sum(int(np.count_nonzero(s.power > 5)) for s in sessions)
    return (on / tot) if tot else 0.0


def _place_sessions(sessions, length, target_duty, rng, cat_id, src_cat, src_sess,
                    session_index):
    """Lay whole sessions onto a zero vector at random non-overlapping offsets.

    Assignment, not accumulation: a single appliance never superimposes on itself.
    Returns the stream. Placement stops when the duty target is met or the timeline is
    too congested to find a free slot.
    """
    stream = np.zeros(length, dtype=np.float32)
    occupied = np.zeros(length, dtype=bool)

    target_on = int(length * target_duty)
    placed_on = 0
    attempts = 0
    max_attempts = 50 * max(1, target_on // max(1, int(np.mean([s.power.size for s in sessions]))))
    max_attempts = max(max_attempts, 200)

    while placed_on < target_on and attempts < max_attempts:
        attempts += 1
        sess = sessions[rng.integers(len(sessions))]
        n = sess.power.size
        if n > length:
            continue
        off = int(rng.integers(0, length - n + 1))
        if occupied[off:off + n].any():
            continue

        stream[off:off + n] = sess.power
        occupied[off:off + n] = True
        placed_on += int(np.count_nonzero(sess.power > 5))

        # Provenance: last writer wins, which is exact here because placements within a
        # category never overlap.
        sid = session_index[sess.name]
        src_cat[off:off + n] = cat_id
        src_sess[off:off + n] = sid

    if placed_on < target_on:
        logging.warning(
            "DEEE aggregate: category id=%d reached duty %.4f of target %.4f after %d "
            "attempts (timeline too congested or sessions too long)",
            cat_id, placed_on / length, target_duty, attempts,
        )
    return stream


def _remix_category(category, sessions, length, target_duty, threshold, rng):
    """Crop-and-remix one category through the tcn_ablation generators."""
    segs = [s.power.astype(np.float64) for s in sessions if s.power.size >= MIN_REMIX_LEN]
    dropped = len(sessions) - len(segs)
    if dropped:
        logging.info(
            "DEEE aggregate: %s -> %d/%d sessions too short for remix (< %d samples)",
            category, dropped, len(sessions), MIN_REMIX_LEN,
        )
    if not segs:
        # Too short to sample from: contribute the sessions verbatim as impulses rather
        # than contributing zeros, which would drop the appliance from the aggregate.
        stream = np.zeros(length, dtype=np.float32)
        for s in sessions:
            off = int(rng.integers(0, max(1, length - s.power.size)))
            stream[off:off + s.power.size] = s.power
        logging.info("DEEE aggregate: %s contributed verbatim (no viable sampler)", category)
        return stream

    mux = _build_multiplexer(category, segs, threshold)
    if mux is None:
        return np.zeros(length, dtype=np.float32)
    frac = _calibrate_active_fraction(mux, target_duty, threshold)
    return generate_stream(mux, length, frac).astype(np.float32)


def build_streams(sessions, appliance_param, mode="session", length=DEFAULT_LENGTH,
                  seed=0, regimes=None, duty_overrides=None, scope="all"):
    """Per-category power streams and their provenance, independent of the base-load variant.

    Built once and shared by every variant, so `pure` and `baseload` differ ONLY by the
    injected load. That is not automatic: the tcn_ablation samplers draw from the LEGACY
    GLOBAL np.random (tcn_ablation.py:124, :141, :170) rather than a passed generator, so
    calling the remix path twice yields different crops even with the same seed. Seeding
    the global RNG here makes a given (mode, seed) reproducible across invocations too.
    """
    if mode not in ("session", "remix"):
        raise ValueError(f"unknown mode {mode!r}")
    if length % LCM_WINDOW:
        raise ValueError(f"length {length} must be a multiple of {LCM_WINDOW}")

    regimes = regimes or sorted(appliance_param)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # the generators in tcn_ablation.py use the global RNG
    by_cat = sessions_by_category(sessions)
    if scope == "ukdale":
        by_cat = {c: v for c, v in by_cat.items() if c in DEEE_CATEGORY_MAP}
    elif scope != "all":
        raise ValueError(f"unknown scope {scope!r}")
    categories = sorted(by_cat)

    session_index = {s.name: i for i, s in enumerate(sessions)}
    src_cat = np.full(length, -1, dtype=np.int16)
    src_sess = np.full(length, -1, dtype=np.int32)

    streams, duty = {}, {}
    for cat_id, category in enumerate(categories):
        cat_sessions = by_cat[category]
        # NOT the recorded duty: these are bench captures made while the appliance ran, so
        # every trace's own duty is ~1.0 and carries no information about household use.
        target, provenance = target_duty(category)
        if duty_overrides and category in duty_overrides:
            target, provenance = duty_overrides[category], "override"
        target = float(np.clip(target, 1e-4, 0.9))
        logging.debug("DEEE aggregate: %s target duty %.5f (%s)", category, target, provenance)

        if mode == "session":
            stream = _place_sessions(
                cat_sessions, length, target, rng, cat_id, src_cat, src_sess, session_index
            )
        else:
            app = DEEE_CATEGORY_MAP.get(category)
            # Remix needs a threshold to find OFF regions. Use the appliance's own where
            # one exists, else a nominal 5 W floor matching the small-value rule.
            thr = appliance_param[regimes[0]].get(app, {}).get("min_threshold", 5) if app else 5
            stream = _remix_category(category, cat_sessions, length, target, thr, rng)

        streams[category] = stream
        duty[category] = float(np.count_nonzero(stream > 5) / length)

    return streams, duty, src_cat, src_sess, categories


def build_aggregate(sessions, appliance_param, compute_status, mode="session",
                    variant="pure", length=DEFAULT_LENGTH, seed=0,
                    regimes=None, duty_overrides=None, prebuilt=None, scope="all"):
    """Synthesise one aggregate and its ground truth.

    `appliance_param` is {regime: {appliance: param dict}} so the two threshold regimes
    (as-trained vs DEEE-adapted) produce labels from one place and cannot diverge.
    `prebuilt` is the tuple returned by build_streams, shared across base-load variants.
    """
    if variant not in BASELOAD_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    regimes = regimes or sorted(appliance_param)
    if prebuilt is None:
        prebuilt = build_streams(
            sessions, appliance_param, mode=mode, length=length, seed=seed,
            regimes=regimes, duty_overrides=duty_overrides, scope=scope,
        )
    streams, duty, src_cat, src_sess, categories = prebuilt
    length = len(next(iter(streams.values())))

    aggregate = np.zeros(length, dtype=np.float32)
    for s in streams.values():
        aggregate += s

    if variant == "baseload":
        # A dedicated generator, so the noise draw cannot shift the stream construction.
        noise = np.random.default_rng(seed + 10_000).normal(
            0.0, NOISE_SIGMA_W, size=length
        ).astype(np.float32)
        aggregate = np.clip(aggregate + BASELOAD_W + noise, 0.0, None)

    # Ground truth is the per-appliance stream, which is unaffected by the base load:
    # the injected floor is unmetered load, not part of any target appliance.
    appliance_power, status = {}, {}
    for category, app in DEEE_CATEGORY_MAP.items():
        if category not in streams:
            continue
        appliance_power[app] = streams[category]
        status[app] = {
            regime: _status_for(streams[category], appliance_param[regime][app], compute_status)
            for regime in regimes
        }

    n_source = int(sum(s.power.size for s in sessions))
    agg = DEEEAggregate(
        aggregate=aggregate,
        streams=streams,
        status=status,
        appliance_power=appliance_power,
        src_category_id=src_cat,
        src_session_id=src_sess,
        session_names=[s.name for s in sessions],
        category_names=categories,
        mode=mode,
        variant=variant,
        scope=scope,
        seed=seed,
        baseload_w=BASELOAD_W if variant == "baseload" else 0.0,
        noise_sigma_w=NOISE_SIGMA_W if variant == "baseload" else 0.0,
        length=length,
        n_source_samples=n_source,
        duty=duty,
    )

    logging.info(
        "DEEE aggregate [%s/%s/%s seed=%d]: L=%d, %d categories, max=%.1f W, mean=%.1f W, "
        "reuse=%.1fx",
        mode, variant, scope, seed, length, len(streams), aggregate.max(), aggregate.mean(),
        agg.reuse_factor,
    )
    for app, per_regime in status.items():
        counts = {r: int(np.count_nonzero(np.diff(np.concatenate([[0], v])) == 1))
                  for r, v in per_regime.items()}
        logging.info("DEEE aggregate:   %s GT activations by regime: %s", app, counts)
    return agg


def save_aggregate(agg, path):
    """Persist to .npz so every model run scores a byte-identical signal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": agg.aggregate,
        "src_category_id": agg.src_category_id,
        "src_session_id": agg.src_session_id,
        "session_names": np.array(agg.session_names, dtype=object),
        "category_names": np.array(agg.category_names, dtype=object),
        "mode": agg.mode,
        "variant": agg.variant,
        "scope": agg.scope,
        "seed": agg.seed,
        "baseload_w": agg.baseload_w,
        "noise_sigma_w": agg.noise_sigma_w,
        "length": agg.length,
        "n_source_samples": agg.n_source_samples,
    }
    for cat, s in agg.streams.items():
        payload[f"stream::{cat}"] = s
    for app, s in agg.appliance_power.items():
        payload[f"power::{app}"] = s
    for app, per_regime in agg.status.items():
        for regime, s in per_regime.items():
            payload[f"status::{app}::{regime}"] = s
    np.savez_compressed(path, **payload)
    logging.info("DEEE aggregate: wrote %s", path)


def load_aggregate(path):
    """Rehydrate a DEEEAggregate saved by save_aggregate."""
    z = np.load(path, allow_pickle=True)
    streams, power, status = {}, {}, {}
    for key in z.files:
        if key.startswith("stream::"):
            streams[key.split("::", 1)[1]] = z[key]
        elif key.startswith("power::"):
            power[key.split("::", 1)[1]] = z[key]
        elif key.startswith("status::"):
            _, app, regime = key.split("::")
            status.setdefault(app, {})[regime] = z[key]
    return DEEEAggregate(
        aggregate=z["aggregate"],
        streams=streams,
        status=status,
        appliance_power=power,
        src_category_id=z["src_category_id"],
        src_session_id=z["src_session_id"],
        session_names=list(z["session_names"]),
        category_names=list(z["category_names"]),
        mode=str(z["mode"]),
        variant=str(z["variant"]),
        scope=str(z["scope"]) if "scope" in z.files else "all",
        seed=int(z["seed"]),
        baseload_w=float(z["baseload_w"]),
        noise_sigma_w=float(z["noise_sigma_w"]),
        length=int(z["length"]),
        n_source_samples=int(z["n_source_samples"]),
    )


def windows_from_streams(aggregate, target_power, target_status, window_size):
    """[n, 2, 2, W] in the layout get_nilm_dataset produces (preprocessing.py:479-553).

    dim1: 0 = aggregate, 1 = the target appliance.  dim2: 0 = power (W), 1 = status.
    Windows tile [0, L) exactly, which requires L % window_size == 0.
    """
    length = aggregate.size
    if length % window_size:
        raise ValueError(f"length {length} is not a multiple of window_size {window_size}")
    n = length // window_size

    out = np.zeros((n, 2, 2, window_size), dtype=np.float32)
    out[:, 0, 0, :] = aggregate.reshape(n, window_size)
    out[:, 0, 1, :] = (aggregate.reshape(n, window_size) > 0).astype(np.float32)
    out[:, 1, 0, :] = np.asarray(target_power, dtype=np.float32).reshape(n, window_size)
    out[:, 1, 1, :] = np.asarray(target_status, dtype=np.float32).reshape(n, window_size)
    return out
