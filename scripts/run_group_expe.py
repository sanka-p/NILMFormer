#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Group Experiments (process data once per dataset/appliance/window/seed)
#
#################################################################################################################

import argparse
import copy
import os
import yaml
import logging
import numpy as np

from omegaconf import OmegaConf

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training

NILM_MODELS = [
    "BiLSTM", "FCN", "CNN1D", "UNetNILM", "DAResNet", "BERT4NILM",
    "DiffNILM", "TSILNet", "Energformer", "BiGRU", "STNILM", "NILMFormer",
]
TSER_MODELS = ["ConvNet", "ResNet", "Inception"]


def prepare_data(base_config):
    """
    Load, split, and scale data once for a given (dataset, appliance, window_size, seed).
    Returns scaled data splits, scaler, cutoff, and threshold.
    Mutates base_config["window_size"] if it was a string (day/week/month).
    """
    np.random.seed(seed=base_config["seed"])

    if base_config["dataset"] == "UKDALE":
        data_builder = UKDALE_DataBuilder(
            data_path=f"{base_config['data_path']}/UKDALE/",
            mask_app=base_config["app"],
            sampling_rate=base_config["sampling_rate"],
            window_size=base_config["window_size"],
        )

        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        if isinstance(base_config["window_size"], str):
            base_config["window_size"] = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=base_config["ind_house_train"]
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=base_config["ind_house_test"]
        )

        data_train, st_date_train, data_valid, st_date_valid = split_train_test_nilmdataset(
            data_train, st_date_train, perc_house_test=0.2, seed=base_config["seed"]
        )

    elif base_config["dataset"] == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path=f"{base_config['data_path']}/REFIT/RAW_DATA_CLEAN/",
            mask_app=base_config["app"],
            sampling_rate=base_config["sampling_rate"],
            window_size=base_config["window_size"],
        )

        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=base_config["house_with_app_i"]
        )

        if isinstance(base_config["window_size"], str):
            base_config["window_size"] = data_builder.window_size

        data_train, st_date_train, data_test, st_date_test = split_train_test_pdl_nilmdataset(
            data.copy(), st_date.copy(), nb_house_test=2, seed=base_config["seed"]
        )

        data_train, st_date_train, data_valid, st_date_valid = split_train_test_pdl_nilmdataset(
            data_train, st_date_train, nb_house_test=1, seed=base_config["seed"]
        )

    else:
        raise ValueError(f"Dataset {base_config['dataset']} unknown. Only 'UKDALE' and 'REFIT' available.")

    scaler = NILMscaler(
        power_scaling_type=base_config["power_scaling_type"],
        appliance_scaling_type=base_config["appliance_scaling_type"],
    )
    data = scaler.fit_transform(data)

    cutoff = float(scaler.appliance_stat2[0])
    threshold = data_builder.appliance_param[base_config["app"]]["min_threshold"]

    data_train = scaler.transform(data_train)
    data_valid = scaler.transform(data_valid)
    data_test = scaler.transform(data_test)

    return (
        data_train, data_valid, data_test, data,
        st_date_train, st_date_valid, st_date_test, st_date,
        scaler, cutoff, threshold,
    )


def run_model(name_model, tuple_data, scaler, cutoff, threshold,
              base_config, models_config, result_base_path):
    result_path = f"{result_base_path}{name_model}_{base_config['seed']}"

    if os.path.isfile(f"{result_path}.pt"):
        logging.info("Skipping (already done): %s.pt", result_path)
        return

    if name_model not in models_config:
        raise ValueError(
            f"Model {name_model} unknown. Available: {list(models_config.keys())}"
        )

    logging.info("---- Running model: %s ----", name_model)

    expes_config = copy.deepcopy(base_config)
    expes_config.update(models_config[name_model])
    expes_config["name_model"] = name_model
    expes_config["cutoff"] = cutoff
    expes_config["threshold"] = threshold
    expes_config["result_path"] = result_path

    expes_config = OmegaConf.create(expes_config)
    launch_models_training(tuple_data, scaler, expes_config)


def main(dataset, sampling_rate, window_size, appliance, seed, model_group):
    try:
        window_size = int(window_size)
    except ValueError:
        pass

    with open("configs/expes.yaml") as f:
        base_config = yaml.safe_load(f)

    with open("configs/datasets.yaml") as f:
        datasets_config = yaml.safe_load(f)
        if dataset not in datasets_config:
            raise ValueError(f"Dataset {dataset} unknown. Only 'UKDALE' and 'REFIT' available.")
        datasets_config = datasets_config[dataset]
        if appliance not in datasets_config:
            raise ValueError(
                f"Appliance {appliance} unknown for dataset {dataset}. "
                f"Available: {list(datasets_config.keys())}"
            )
        base_config.update(datasets_config[appliance])

    with open("configs/models.yaml") as f:
        models_config = yaml.safe_load(f)

    base_config["dataset"] = dataset
    base_config["appliance"] = appliance
    base_config["window_size"] = window_size
    base_config["sampling_rate"] = sampling_rate
    base_config["seed"] = seed

    # Build result path using the original window_size label (string or int) as directory name,
    # matching the behaviour of run_one_expe.py which creates this path before resolving "day"/"week"/"month".
    result_path = create_dir(base_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset}_{appliance}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    logging.info(
        "Processing data for %s / %s / win=%s / seed=%s ...",
        dataset, appliance, window_size, seed,
    )
    (
        data_train, data_valid, data_test, data,
        st_date_train, st_date_valid, st_date_test, st_date,
        scaler, cutoff, threshold,
    ) = prepare_data(base_config)
    logging.info("Data ready. window_size resolved to %s.", base_config["window_size"])

    if model_group == "nilm":
        models = NILM_MODELS
        tuple_data = (
            data_train, data_valid, data_test, data,
            st_date_train, st_date_valid, st_date_test, st_date,
        )
    else:
        models = TSER_MODELS
        X, y = nilmdataset_to_tser(data)
        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)
        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    for name_model in models:
        run_model(
            name_model, tuple_data, scaler, cutoff, threshold,
            base_config, models_config, result_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NILMFormer Group Experiments — process data once, run all models."
    )
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (UKDALE or REFIT).")
    parser.add_argument("--sampling_rate", required=True, type=str, help="Sampling rate, e.g. '1min'.")
    parser.add_argument("--window_size", required=True, type=str, help="Window size, e.g. '128' or 'day'.")
    parser.add_argument("--appliance", required=True, type=str, help="Appliance name, e.g. 'Kettle'.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    parser.add_argument(
        "--model_group", required=True, choices=["nilm", "tser"],
        help="'nilm' for seq2seq models, 'tser' for regression models."
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        seed=args.seed,
        model_group=args.model_group,
    )
