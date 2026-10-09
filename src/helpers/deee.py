#################################################################################################################
#
# @description : Loader for the DEEE_SmartHome dataset.
#
# DEEE_SmartHome is NOT a whole-home NILM recording. It is 35 short single-appliance bench
# traces captured one at a time by a single EKM plug meter, collected deliberately so that
# an aggregate could be synthesised from them afterwards. There is no mains channel, no
# submeter set and no house structure, so it does not fit the *_DataBuilder interface in
# preprocessing.py, which assumes per-house files containing an aggregate plus submeters.
#
# This module does the data layer only: discover the traces, load them, and put them on the
# repo's 10 s grid using the SAME steps the UK-DALE appliance path uses
# (preprocessing.py:688-700), so a DEEE watt value is processed identically to a UK-DALE
# one. Aggregate synthesis lives in src/helpers/deee_aggregate.py.
#
# Three things about the raw files drive the implementation:
#   * `energy` is a cumulative meter register (166.9 -> 168.4 kWh across the whole corpus),
#     not per-session energy, and `pf` is not a power factor (it reaches 2.00). Both are
#     dropped rather than carried as unused columns.
#   * the sampling grid is ~1 Hz but JITTERED -- modal dt is 1.024 s for the 23 `main.csv`
#     sessions and 1.002-1.018 s for the 12 device-named ones -- so time-based resampling
#     is required and reshaping on a fixed stride would silently drift.
#   * the CSV basename is inconsistent (23 `main.csv`, 12 device-specific) and two devices
#     nest one level deeper, so the file is located via each leaf's metadata.json rather
#     than by globbing.
#
#################################################################################################################

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEEE_ROOT = Path("/new-home/e19/e19275/NILM/NILM_datasets/DEEE_SmartHome")

#: DEEE top-level category -> the UK-DALE appliance name it corresponds to.
#: Everything absent from this map is an UNMETERED DISTRACTOR: it contributes to the
#: synthetic aggregate and never carries a label. Only these three of the twelve DEEE
#: categories overlap the UK-DALE target set -- there is no fridge and no dishwasher.
DEEE_CATEGORY_MAP = {
    "Kettle": "kettle",
    "MW": "microwave",
    "Washing Machine": "washing_machine",
}

#: Mirrors UKDALE_DataBuilder.cutoff (preprocessing.py:396). A no-op at DEEE's 1603 W peak,
#: kept so the two paths are byte-identical in intent.
CUTOFF = 6000
#: Mirrors the `appl_data[appl_data < 5] = 0` small-value removal at preprocessing.py:693.
SMALL_VALUE_W = 5
#: A session is cut wherever consecutive samples are further apart than this. The corpus has
#: exactly one such gap (88.9 s, in Laptop/swift3_80_100_light). Splitting is deliberate:
#: _fill_long_gaps_with_zero only fires above 120 s and ffill(limit=6) only spans 60 s, so an
#: unsplit 88.9 s hole would survive as NaN and silently cost a window to _check_anynan.
MAX_GAP_S = 30
#: Pieces shorter than this after resampling are dropped -- too short to be an activation.
MIN_PIECE_LEN = 3


@dataclass
class TraceSession:
    """One contiguous run of one appliance, on the repo's resampled grid."""

    category: str  #: DEEE top-level directory, e.g. "MW"
    variant: str  #: remaining path components, e.g. "abans20l_2water_high"
    appliance: str | None  #: UK-DALE name, or None for an unmetered distractor
    power: np.ndarray  #: float32 watts, on the resampled grid
    source_path: str  #: CSV path relative to the dataset root
    source_start_utc: pd.Timestamp  #: real wall-clock start of this piece
    n_native: int  #: samples before resampling
    piece: int  #: index of this piece within its trace (0 unless the trace was split)

    @property
    def name(self):
        return f"{self.category}/{self.variant}" + (f"#{self.piece}" if self.piece else "")


def discover_traces(root=DEEE_ROOT):
    """[(category, variant, csv_path, metadata)] for every leaf recording.

    Driven by metadata.json rather than a glob: each leaf names its own CSV in
    `devices[0]["csv_file"]`, which absorbs both the inconsistent basename and the extra
    nesting level under `Fan/Sisl_60W/` and `Hair Dryer/Philips_1000W/`.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DEEE root not found: {root}")

    out = []
    for meta_path in sorted(root.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text())
        devices = meta.get("devices") or []
        if not devices:
            logging.warning("DEEE: %s lists no devices, skipping", meta_path)
            continue
        if len(devices) > 1:
            logging.warning(
                "DEEE: %s lists %d devices; only the first is used", meta_path, len(devices)
            )

        device = devices[0]
        csv_path = meta_path.parent / device["csv_file"]
        if not csv_path.is_file():
            # Known defect in the corpus: two of the three Fan/Sisl_60W leaves name the
            # wrong file -- Setting_Low points at "Fan_Sisil_60W.csv" (actual: ..._L.csv)
            # and Setting_High points at "..._L.csv", which is Setting_Low's file. Each
            # leaf holds exactly one CSV, so fall back to it rather than dropping the
            # trace; ambiguity is refused rather than guessed.
            candidates = sorted(meta_path.parent.glob("*.csv"))
            if len(candidates) != 1:
                logging.error(
                    "DEEE: %s names a missing CSV (%s) and its directory holds %d CSVs -- "
                    "cannot disambiguate, skipping",
                    meta_path, device["csv_file"], len(candidates),
                )
                continue
            logging.warning(
                "DEEE: %s names a missing CSV (%s); falling back to the only CSV present "
                "(%s). The metadata csv_file field is wrong for this leaf.",
                meta_path, device["csv_file"], candidates[0].name,
            )
            csv_path = candidates[0]

        rel = meta_path.parent.relative_to(root).parts
        out.append(
            {
                "category": rel[0],
                "variant": "/".join(rel[1:]) or rel[0],
                "csv_path": csv_path,
                "meta": meta,
                "device": device,
            }
        )
    return out


def load_trace(csv_path):
    """Raw trace as a UTC-indexed frame with just the columns that mean anything.

    `energy` (cumulative register) and `pf` (exceeds 1.0, so not a power factor) are
    dropped. The float unix `timestamp` is preferred over the ISO `utc_time` string; the
    two agree, and the float avoids a string parse per row.
    """
    df = pd.read_csv(csv_path, usecols=["timestamp", "power", "voltage"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("time")[["power", "voltage"]].sort_index()


def resample_trace(df, sampling_rate="10s"):
    """Split on gaps, then resample each piece exactly as the UK-DALE appliance path does.

    Returns [(power Series, n_native)], one entry per contiguous piece.
    """
    if df.empty:
        return []

    # Split first, so a long hole becomes two sessions rather than an interpolated ramp.
    gap = df.index.to_series().diff().dt.total_seconds()
    piece_id = (gap > MAX_GAP_S).cumsum()

    out = []
    for _, chunk in df.groupby(piece_id):
        n_native = len(chunk)
        s = chunk["power"].resample(sampling_rate).mean()
        # Same order as preprocessing.py:690-700: resample, drop small values, clip.
        s = s.fillna(0.0)
        s[s < SMALL_VALUE_W] = 0
        s = s.clip(lower=0, upper=CUTOFF)
        if len(s) < MIN_PIECE_LEN:
            continue
        out.append((s.astype(np.float32), n_native))
    return out


def load_all_sessions(root=DEEE_ROOT, sampling_rate="10s"):
    """(sessions, inventory DataFrame) for the whole corpus."""
    traces = discover_traces(root)
    n_csv = len(list(Path(root).rglob("*.csv")))
    logging.info("DEEE: discovered %d traces under %s (%d CSVs on disk)",
                 len(traces), root, n_csv)
    # Silence is the failure mode here: a metadata defect drops a trace with only a log
    # line, and a short corpus looks exactly like a correct one in the results.
    if len(traces) != n_csv:
        raise RuntimeError(
            f"DEEE: resolved {len(traces)} traces but {n_csv} CSVs exist under {root}. "
            "Some metadata.json names a file that could not be resolved -- see the "
            "warnings above. Fix the metadata or the fallback before scoring anything."
        )

    sessions, rows = [], []
    for t in traces:
        df = load_trace(t["csv_path"])
        rel = str(t["csv_path"].relative_to(Path(root)))

        n_written = t["device"].get("samples_written")
        if n_written is not None and n_written != len(df):
            logging.warning(
                "DEEE: %s declares samples_written=%s but the CSV has %d rows",
                rel, n_written, len(df),
            )
        # Known data nit: one 0.0 V sample in Fan/innovexa_fanspeed1_louveron. Logged, not
        # dropped -- it is a voltage glitch and the power reading beside it is usable.
        n_bad_v = int((df["voltage"] <= 0).sum())
        if n_bad_v:
            logging.info("DEEE: %s has %d non-positive voltage sample(s)", rel, n_bad_v)

        dt = df.index.to_series().diff().dt.total_seconds()
        pieces = resample_trace(df, sampling_rate)
        appliance = DEEE_CATEGORY_MAP.get(t["category"])

        for i, (s, n_native) in enumerate(pieces):
            sessions.append(
                TraceSession(
                    category=t["category"],
                    variant=t["variant"],
                    appliance=appliance,
                    power=s.to_numpy(),
                    source_path=rel,
                    source_start_utc=s.index[0],
                    n_native=n_native,
                    piece=i,
                )
            )

        power = df["power"].to_numpy()
        rows.append(
            {
                "category": t["category"],
                "variant": t["variant"],
                "appliance": appliance or "",
                "scored": bool(appliance),
                "csv_path": rel,
                "n_native": len(df),
                "n_pieces": len(pieces),
                "n_resampled": int(sum(len(s) for s, _ in pieces)),
                "peak_w": float(power.max()) if len(power) else 0.0,
                "mean_w": float(power.mean()) if len(power) else 0.0,
                "duty_native": float((power > SMALL_VALUE_W).mean()) if len(power) else 0.0,
                "median_dt_s": float(dt.median()) if len(dt.dropna()) else float("nan"),
                "max_gap_s": float(dt.max()) if len(dt.dropna()) else float("nan"),
                "start_utc": df.index[0].isoformat() if len(df) else "",
                "end_utc": df.index[-1].isoformat() if len(df) else "",
                "n_bad_voltage": n_bad_v,
            }
        )

    inventory = pd.DataFrame(rows).sort_values(["category", "variant"]).reset_index(drop=True)
    logging.info(
        "DEEE: %d traces -> %d sessions, %d resampled samples at %s (%d scored traces)",
        len(traces), len(sessions), inventory["n_resampled"].sum(), sampling_rate,
        inventory["scored"].sum(),
    )
    return sessions, inventory


def sessions_by_category(sessions):
    """{category: [TraceSession]}, preserving discovery order."""
    out = {}
    for s in sessions:
        out.setdefault(s.category, []).append(s)
    return out
