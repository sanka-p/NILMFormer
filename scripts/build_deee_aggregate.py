#################################################################################################################
#
# @description : Build and cache the synthetic DEEE_SmartHome aggregates.
#
# Writes results/deee_aggregate_{mode}_{variant}_seed{N}.npz, one per combination, so every
# model run afterwards scores a byte-identical signal and the reported +-sd is purely model
# variance rather than test-set variance.
#
# Also prints the provenance of every duty cycle used (measured from UK-DALE house 2 for the
# three scored appliances, assumed for the nine distractor categories) and the injected
# base-load/noise values, because those are assumptions a reader has to be able to audit.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.build_deee_aggregate
#     PYTHONPATH=. .venv/bin/python -m scripts.build_deee_aggregate --modes session --variants pure
#
#################################################################################################################

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.helpers.deee import load_all_sessions
from src.helpers.deee_aggregate import (
    BASELOAD_VARIANTS,
    SCOPES,
    DEFAULT_LENGTH,
    build_aggregate,
    build_regimes,
    save_aggregate,
    target_duty,
)
from src.helpers.preprocessing import UKDALE_DataBuilder

OUT_DIR = Path("results")
INVENTORY = OUT_DIR / "deee_trace_inventory.csv"


def aggregate_path(mode, variant, seed, out_dir=OUT_DIR, scope="all"):
    tag = "" if scope == "all" else f"_{scope}"
    return Path(out_dir) / f"deee_aggregate_{mode}_{variant}{tag}_seed{seed}.npz"


def verify(agg, regimes):
    """V5: the aggregate sanity checks from the plan. Raises on anything that must hold."""
    recon = np.zeros_like(agg.aggregate)
    for s in agg.streams.values():
        recon += s
    if agg.variant == "pure":
        assert np.allclose(agg.aggregate, recon), "aggregate is not the exact sum of streams"
    else:
        # baseload adds a constant + noise, so check the residual instead.
        resid = agg.aggregate - recon
        assert resid.mean() > 0, "baseload variant did not raise the floor"
        logging.info("  baseload residual: mean %.2f W, sd %.2f W", resid.mean(), resid.std())

    assert agg.length % 512 == 0, "length must tile every window size"

    # The degenerate cases are expected, so assert them: a silent change here would turn a
    # documented finding into an unexplained number.
    n_kettle = int(agg.status["kettle"]["astrained"].sum())
    assert n_kettle == 0, (
        f"expected 0 as-trained kettle ON samples (DEEE kettles peak at 1370 W, below the "
        f"2000 W UK-DALE threshold) but got {n_kettle}"
    )
    n_wm = int(agg.status["washing_machine"]["astrained"].sum())
    assert n_wm == 0, (
        f"expected 0 as-trained washing-machine ON samples (longest above-threshold run is "
        f"730 s, below the 1800 s min_on_duration) but got {n_wm}"
    )
    for app in agg.status:
        n = int(agg.status[app]["adapted"].sum())
        assert n > 0, f"{app} has no ON samples even under the adapted regime"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", nargs="+", default=["session", "remix"])
    p.add_argument("--variants", nargs="+", default=list(BASELOAD_VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--scopes", nargs="+", default=list(SCOPES),
                   help="'all' = 12 categories; 'ukdale' = only the 3 the models know")
    p.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    builder = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=["kettle"], sampling_rate="10s", window_size=128
    )
    regimes = build_regimes(builder.appliance_param)
    logging.info("Threshold regimes:")
    for name, params in regimes.items():
        logging.info(
            "  %-10s %s", name,
            {app: (prm["min_threshold"], prm["min_on_duration"]) for app, prm in params.items()},
        )

    sessions, inventory = load_all_sessions()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY, index=False)
    logging.info("wrote %s (%d traces)", INVENTORY, len(inventory))

    logging.info("Duty cycles used (target -> provenance):")
    for category in sorted({s.category for s in sessions}):
        t, prov = target_duty(category)
        logging.info("  %-18s %.5f  %s", category, t, prov)

    for scope in a.scopes:
        for mode in a.modes:
            for variant in a.variants:
                for seed in a.seeds:
                    path = aggregate_path(mode, variant, seed, a.out_dir, scope)
                    if path.is_file() and not a.force:
                        logging.info("skip %s (exists)", path)
                        continue
                    agg = build_aggregate(
                        sessions, regimes, builder._compute_status,
                        mode=mode, variant=variant, length=a.length, seed=seed,
                        scope=scope,
                    )
                    verify(agg, regimes)
                    save_aggregate(agg, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
