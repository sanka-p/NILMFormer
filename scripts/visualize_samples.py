import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
)
from src.helpers.utils import create_dir

SAMPLING_RATE = "10s"
WINDOW_SIZE = 128
DATA_PATH = "data/"
OUT_BASE = "plots/data_samples"
SEED = 0
TIME_AXIS = np.arange(WINDOW_SIZE) * 10  # seconds


def plot_sample(agg, app_power, app_state, title, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(TIME_AXIS, agg, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Aggregate (W)")
    axes[0].set_title(title)

    axes[1].plot(TIME_AXIS, app_power, color="darkorange", linewidth=0.8)
    axes[1].fill_between(TIME_AXIS, 0, app_power, where=app_state > 0, alpha=0.3, color="red", label="Active")
    axes[1].set_ylabel("Appliance (W)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(handles=[mpatches.Patch(color="red", alpha=0.3, label="Active")], loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def save_split(data, split_name, dataset_name, appliance, st_date=None):
    out_dir = os.path.join(OUT_BASE, f"{dataset_name}_{appliance}", split_name)
    create_dir(out_dir)
    n = len(data)
    print(f"  {split_name}: {n} samples → {out_dir}")
    for i in range(n):
        agg = data[i, 0, 0, :]
        app_power = data[i, 1, 0, :]
        app_state = data[i, 1, 1, :]
        date_str = ""
        if st_date is not None:
            try:
                date_str = f" | {st_date.iloc[i].values[0]}"
            except Exception:
                pass
        title = f"{dataset_name} | {appliance} | {split_name} | sample {i:05d}{date_str}"
        out_path = os.path.join(out_dir, f"sample_{i:05d}.png")
        plot_sample(agg, app_power, app_state, title, out_path)


def process_ukdale(datasets_cfg):
    appliances = datasets_cfg["UKDALE"]
    for app_key, app_cfg in appliances.items():
        app_name = app_cfg["app"]
        ind_train = app_cfg["ind_house_train"]
        ind_test = app_cfg["ind_house_test"]

        print(f"\nUKDALE | {app_name}")
        builder = UKDALE_DataBuilder(
            data_path=f"{DATA_PATH}UKDALE/",
            mask_app=app_name,
            sampling_rate=SAMPLING_RATE,
            window_size=WINDOW_SIZE,
        )

        data_train, st_date_train = builder.get_nilm_dataset(house_indicies=ind_train)
        data_test, st_date_test = builder.get_nilm_dataset(house_indicies=ind_test)

        data_train, st_date_train, data_valid, st_date_valid = split_train_test_nilmdataset(
            data_train, st_date_train, perc_house_test=0.2, seed=SEED
        )

        save_split(data_train, "train", "UKDALE", app_name, st_date_train)
        save_split(data_valid, "valid", "UKDALE", app_name, st_date_valid)
        save_split(data_test, "test", "UKDALE", app_name, st_date_test)


def process_refit(datasets_cfg):
    appliances = datasets_cfg["REFIT"]
    for app_key, app_cfg in appliances.items():
        app_name = app_cfg["app"].strip()
        houses = app_cfg["house_with_app_i"]

        print(f"\nREFIT | {app_name}")
        builder = REFIT_DataBuilder(
            data_path=f"{DATA_PATH}REFIT/RAW_DATA_CLEAN/",
            mask_app=app_name,
            sampling_rate=SAMPLING_RATE,
            window_size=WINDOW_SIZE,
        )

        data, st_date = builder.get_nilm_dataset(house_indicies=houses)

        data_train, st_date_train, data_test, st_date_test = split_train_test_pdl_nilmdataset(
            data.copy(), st_date.copy(), nb_house_test=2, seed=SEED
        )
        data_train, st_date_train, data_valid, st_date_valid = split_train_test_pdl_nilmdataset(
            data_train, st_date_train, nb_house_test=1, seed=SEED
        )

        save_split(data_train, "train", "REFIT", app_name, st_date_train)
        save_split(data_valid, "valid", "REFIT", app_name, st_date_valid)
        save_split(data_test, "test", "REFIT", app_name, st_date_test)


def main():
    with open("configs/datasets.yaml", "r") as f:
        datasets_cfg = yaml.safe_load(f)

    create_dir(OUT_BASE)
    process_ukdale(datasets_cfg)
    process_refit(datasets_cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
