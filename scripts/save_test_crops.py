import argparse
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.helpers.preprocessing import UKDALE_DataBuilder, split_train_test_nilmdataset
from src.helpers.utils import create_dir


def to_long_df(data, st_date):
    """Convert (N, 2, 2, W) array + st_date DataFrame to long-format DataFrame."""
    n_windows, _, _, window_size = data.shape

    window_idx = np.repeat(np.arange(n_windows), window_size)
    timestep = np.tile(np.arange(window_size), n_windows)
    house_id = np.repeat(st_date.index.values, window_size)
    start_date = np.repeat(st_date["start_date"].values, window_size)

    agg_power = data[:, 0, 0, :].ravel()
    agg_state = data[:, 0, 1, :].ravel()
    app_power = data[:, 1, 0, :].ravel()
    app_state = data[:, 1, 1, :].ravel()

    return pd.DataFrame({
        "window_idx": window_idx,
        "timestep": timestep,
        "house_id": house_id,
        "start_date": start_date,
        "agg_power": agg_power,
        "agg_state": agg_state,
        "app_power": app_power,
        "app_state": app_state,
    })


def print_summary(results, ukdale_cfg):
    for (sr, ws), apps in results.items():
        print(f"\nsr={sr}  ws={ws}")
        print(f"{'Appliance':<20} {'Train Houses':<16} {'Valid Houses':<16} {'Test Houses':<14} {'Train N':>8} {'Valid N':>8} {'Test N':>8}")
        print("-" * 92)
        for app_key, entry in apps.items():
            cfg = ukdale_cfg[app_key]
            train_h = str(cfg["ind_house_train"])
            valid_h = str(cfg.get("ind_house_valid", []))
            test_h  = str(cfg["ind_house_test"])
            print(f"{app_key:<20} {train_h:<16} {valid_h:<16} {test_h:<14}"
                  f" {entry['n_train']:>8} {entry['n_valid']:>8} {entry['n_test']:>8}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/")
    parser.add_argument("--output_path", default="cropped/")
    parser.add_argument("--sampling_rates", nargs="+", default=["10s"])
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    with open("configs/datasets.yaml") as f:
        datasets_cfg = yaml.safe_load(f)

    ukdale_cfg = datasets_cfg["UKDALE"]

    results = {}

    for sr in args.sampling_rates:
        for ws in args.window_sizes:
            results[(sr, ws)] = {}
            for app_key, app_cfg in ukdale_cfg.items():
                tag = f"UKDALE_{app_key}_{sr}"
                if not args.summary:
                    print(f"\n{tag} / ws={ws}")

                builder = UKDALE_DataBuilder(
                    data_path=f"{args.data_path}/UKDALE/",
                    mask_app=app_cfg["app"],
                    sampling_rate=sr,
                    window_size=ws,
                )

                data_train_all, st_train_all = builder.get_nilm_dataset(
                    house_indicies=app_cfg["ind_house_train"]
                )
                data_train, st_train, data_valid, st_valid = split_train_test_nilmdataset(
                    data_train_all, st_train_all, perc_house_test=0.2, seed=args.seed
                )
                data_test, st_test = builder.get_nilm_dataset(
                    house_indicies=app_cfg["ind_house_test"]
                )

                results[(sr, ws)][app_key] = {
                    "data_train": data_train, "st_train": st_train,
                    "data_valid": data_valid, "st_valid": st_valid,
                    "data_test":  data_test,  "st_test":  st_test,
                    "n_train": len(data_train),
                    "n_valid": len(data_valid),
                    "n_test":  len(data_test),
                    "tag": tag,
                }

                if not args.summary:
                    print(f"  train: {len(data_train):>6} windows")
                    print(f"  valid: {len(data_valid):>6} windows")
                    print(f"  test:  {len(data_test):>6} windows")

    if args.summary:
        print_summary(results, ukdale_cfg)

    for (sr, ws), apps in results.items():
        for app_key, entry in apps.items():
            tag = entry["tag"]
            out_dir = create_dir(os.path.join(args.output_path, tag, str(ws)))

            splits = [
                ("train_data.csv", entry["data_train"], entry["st_train"]),
                ("valid_data.csv", entry["data_valid"], entry["st_valid"]),
                ("test_data.csv",  entry["data_test"],  entry["st_test"]),
            ]
            for fname, data, st_date in splits:
                path = os.path.join(out_dir, fname)
                to_long_df(data, st_date).to_csv(path, index=False)
                print(f"  saved {fname}")


if __name__ == "__main__":
    main()
