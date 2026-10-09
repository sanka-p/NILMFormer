#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from omegaconf import OmegaConf

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    REDD_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import TCN_ABLATION_MODELS, launch_models_training
from src.helpers.tcn_ablation import (
    build_synthetic_windows,
    extract_activation_segments,
    fit_kl_basis,
    load_curated_segments,
    resolve_synth_app,
)

#: Activation segments are shared by every appliance and arm of a given
#: (window_size, seed), but rebuilding them costs a full 5-appliance dataset pass.
TCN_AUG_CACHE = Path("results/tcn_aug_cache")


def _tcn_ablation_segments(expes_config):
    """Activation segments + duty cycles for the synthetic aggregate.

    `segments_source: curated` swaps the activation WAVEFORMS for the curated store's,
    while keeping this repo's empirical duty cycles -- curated duty is 0.73-0.99 by
    construction (its segments *are* on-regions), so reusing it would generate absurdly
    dense streams and change far more than the thing being ablated.
    """
    segments, on_fractions = _repo_segments(expes_config)

    source = str(expes_config.model_kwargs.get("segments_source", "repo"))
    if source == "repo":
        return segments, on_fractions
    if source != "curated":
        raise ValueError(f"Unknown segments_source '{source}'. Use 'repo' or 'curated'.")

    apps = list(expes_config.synth_aggregate_apps)
    # REDD's splits are per-appliance, and the repo arm extracts from the run's own train
    # houses, so draw every appliance's curated activations from those same houses --
    # that keeps the curated arm matched to the arm it is compared against.
    houses = (
        {a: list(expes_config.ind_house_train) for a in apps}
        if expes_config.dataset == "REDD"
        else None  # UK-DALE uses the static house-matched map
    )
    # Pool-size sweep: cap ONLY the target appliance, so the measured effect is the
    # diversity of its own activations rather than a thinner synthetic aggregate.
    cap = expes_config.model_kwargs.get("max_segments")
    cap_target = expes_config.model_kwargs.get("max_segments_target")
    if cap_target:
        tgt = resolve_synth_app(expes_config.app, apps) or expes_config.app
        cap = {tgt: int(cap_target)}
        logging.info(
            "TCN ablation: curated pool for '%s' capped at %d activations.",
            tgt, int(cap_target),
        )

    curated = load_curated_segments(
        apps,
        sampling_rate=expes_config.sampling_rate,
        max_segments=cap,
        seed=expes_config.seed,
        dataset=expes_config.dataset,
        houses=houses,
    )
    return curated, on_fractions


def _repo_segments(expes_config):
    """
    Activation segments and duty cycles for every appliance of the synthetic aggregate,
    cut from the training houses of this repo's own (non-curated) UK-DALE data.

    UK-DALE only: the seeded 80/20 split is applied so most of what the real validation
    set contains is held out of the augmentation pool. It cannot be held out exactly --
    this array carries every appliance, so its NaN-dropped window grid does not line up
    with the single-appliance grid `data_train` comes from (the misalignment documented in
    scripts/score_osw.py:15). Residual overlap only affects which of the three epochs is
    selected, not the reported test metrics.

    REDD holds out a whole validation *house* instead (run_one_expe's REDD branch), so
    there is nothing to carve out of the training windows and the split is skipped.
    """
    houses = list(expes_config.ind_house_train)
    apps = list(expes_config.synth_aggregate_apps)
    key = "{}_h{}_{}_w{}_s{}".format(
        expes_config.dataset,
        "-".join(str(h) for h in houses),
        expes_config.sampling_rate,
        expes_config.window_size,
        expes_config.seed,
    )
    cache_file = TCN_AUG_CACHE / f"{key}.pkl"

    if cache_file.is_file():
        logging.info("TCN ablation: loading cached segments from %s", cache_file)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logging.info(
        "TCN ablation: extracting %s activation segments (houses %s) ...",
        expes_config.dataset, houses,
    )
    if expes_config.dataset == "REDD":
        builder = REDD_DataBuilder(
            data_path=f"{expes_config.data_path}/REDD/redd.h5",
            mask_app=apps,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            synth_aggregate_apps=apps,
        )
    else:
        builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=apps,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            synth_aggregate_apps=apps,
        )
    arr, st = builder.get_nilm_dataset(house_indicies=houses)
    if expes_config.dataset != "REDD":
        arr, st, _, _ = split_train_test_nilmdataset(
            arr, st, perc_house_test=0.2, seed=expes_config.seed
        )
    result = extract_activation_segments(
        arr, apps, rng=np.random.default_rng(expes_config.seed)
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def _dummy_st_date(template, n_rows):
    """Placeholder timestamps for synthetic windows.

    The TCN arms get a NILMDataset with no exogenous channels (expes.py:155), so st_date
    is never read for them -- but its length must track the data array.
    """
    stamp = template["start_date"].iloc[0]
    return pd.DataFrame(
        data=[stamp] * n_rows, index=[-1] * n_rows, columns=["start_date"]
    )


def _prepare_tcn_ablation(expes_config, data_builder, data_train, st_date_train):
    """
    Fit the KL basis and, for the augmented arm, build the synthetic training set.

    Runs on unscaled watts and after the train/valid split, so validation and test stay
    real and the scaler is still fit on real data alone -- scaling is therefore identical
    across all arms.
    """
    mk = expes_config.model_kwargs
    if not (mk.get("use_kl") or mk.get("augment")):
        return data_train, st_date_train

    segments, on_fractions = _tcn_ablation_segments(expes_config)

    if mk.get("use_kl"):
        pooled = [seg for segs in segments.values() for seg in segs]
        _, basis = fit_kl_basis(pooled, order=int(mk.get("kl_order", 10)))
        mk.kl_basis = basis.tolist()
        logging.info(
            "TCN ablation: KL basis fitted on %d pooled segments.", len(pooled)
        )

    if mk.get("augment"):
        apps = list(expes_config.synth_aggregate_apps)
        target = resolve_synth_app(expes_config.app, apps)
        if target is None:
            raise ValueError(
                f"Target appliance '{expes_config.app}' is not in synth_aggregate_apps "
                f"{apps}; the augmented arm cannot build its label channel."
            )
        if target != expes_config.app:
            # REDD: `app` is WasherDryer while synth_aggregate_apps lists WashingMachine.
            # Same physical meter (preprocessing.py:1258).
            logging.info(
                "TCN ablation: target '%s' resolved to synth appliance '%s'.",
                expes_config.app, target,
            )

        mode = str(mk.get("aug_mode", "replace"))
        ratio = float(mk.get("aug_ratio", 1.0))
        n_real = len(data_train)
        n_aug = n_real if mode == "replace" else max(1, int(round(n_real * ratio)))

        synth = build_synthetic_windows(
            segments=segments,
            on_fractions=on_fractions,
            appliance_names=apps,
            target_appliance=target,
            appliance_param=data_builder.appliance_param,
            compute_status=data_builder._compute_status,
            n_windows=n_aug,
            window_size=int(expes_config.window_size),
            seed=expes_config.seed,
        )
        synth_st = _dummy_st_date(st_date_train, n_aug)

        if mode == "replace":
            data_train, st_date_train = synth, synth_st
        elif mode == "mix":
            data_train = np.concatenate((data_train, synth), axis=0)
            st_date_train = pd.concat([st_date_train, synth_st], axis=0)
        else:
            raise ValueError(f"Unknown aug_mode '{mode}'. Use 'replace' or 'mix'.")

        logging.info(
            "TCN ablation: aug_mode=%s, %d real -> %d training windows.",
            mode, n_real, len(data_train),
        )

    return data_train, st_date_train


def launch_one_experiment(expes_config: OmegaConf):
    np.random.seed(seed=expes_config.seed)

    logging.info("Process data ...")
    if expes_config.dataset == "UKDALE":
        data_builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            synth_aggregate_apps=expes_config.synth_aggregate_apps,
        )

        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_nilmdataset(
                data_train,
                st_date_train,
                perc_house_test=0.2,
                seed=expes_config.seed,
            )
        )

    elif expes_config.dataset == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            synth_aggregate_apps=expes_config.synth_aggregate_apps,
        )

        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config.house_with_app_i
        )

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config.seed
            )
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config.seed
            )
        )

    elif expes_config.dataset == "REDD":
        data_builder = REDD_DataBuilder(
            data_path=f"{expes_config.data_path}/REDD/redd.h5",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
            synth_aggregate_apps=expes_config.synth_aggregate_apps,
        )

        ind_house_train = list(expes_config.ind_house_train)
        ind_house_valid = list(expes_config.ind_house_valid)
        ind_house_test = list(expes_config.ind_house_test)
        all_houses = sorted(set(ind_house_train + ind_house_valid + ind_house_test))

        data, st_date = data_builder.get_nilm_dataset(house_indicies=all_houses)

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=ind_house_train
        )
        data_valid, st_date_valid = data_builder.get_nilm_dataset(
            house_indicies=ind_house_valid
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=ind_house_test
        )

    logging.info("             ... Done.")

    if expes_config.name_model in TCN_ABLATION_MODELS:
        data_train, st_date_train = _prepare_tcn_ablation(
            expes_config, data_builder, data_train, st_date_train
        )

    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)

    expes_config.cutoff = float(scaler.appliance_stat2[0])
    expes_config.threshold = data_builder.appliance_param[expes_config.app][
        "min_threshold"
    ]

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        X, y = nilmdataset_to_tser(data)

        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)

        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    else:
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        tuple_data = (
            data_train,
            data_valid,
            data_test,
            data,
            st_date_train,
            st_date_valid,
            st_date_test,
            st_date,
        )

    launch_models_training(tuple_data, scaler, expes_config)


def main(dataset, sampling_rate, window_size, appliance, name_model, seed, result_path=None):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (UKDALE or REFIT).
        sampling_rate (int): Selected sampling rate.
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance.
        name_model (str): Name of the model to use for the experiment.
        seed (int): Random seed for reproducibility.
        result_path (str): Optional output root, overriding configs/expes.yaml. Lets an
            ablation write to its own directory without editing the global config.
    """

    # Attempt to convert window_size to int
    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    # Load configurations
    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)

        # Dataset name check
        if dataset in datasets_config:
            datasets_config = datasets_config[dataset]
        else:
            raise ValueError(
                "Dataset {} unknown. Only 'UKDALE', 'REFIT', and 'REDD' available.".format(
                    dataset
                )
            )

    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        # Selected baseline check
        if name_model in baselines_config:
            expes_config.update(baselines_config[name_model])
        else:
            raise ValueError(
                "Model {} unknown. List of implemented baselines: {}".format(
                    name_model, list(baselines_config.keys())
                )
            )

    # Merge dataset-level keys (non-appliance entries) — overrides model-level config.
    # Needed so REFIT/REDD supply their own synth_aggregate_apps names.
    dataset_level = {k: v for k, v in datasets_config.items() if not isinstance(v, dict)}
    expes_config.update(dataset_level)

    # Selected appliance check
    if appliance in datasets_config:
        expes_config.update(datasets_config[appliance])
    else:
        logging.error("Appliance '%s' not found in datasets_config.", appliance)
        raise ValueError(
            "Appliance {} unknown. List of available appliances (for selected {} dataset): {}, ".format(
                appliance, dataset, list(datasets_config.keys())
            )
        )

    # Display experiment config with passed parameters
    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance)
    logging.info("      Model: %s", name_model)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    # Update experiment config with passed parameters
    expes_config["dataset"] = dataset
    expes_config["appliance"] = appliance
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = name_model

    if result_path is not None:
        expes_config["result_path"] = (
            result_path if result_path.endswith("/") else result_path + "/"
        )

    # Create directories for results
    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset}_{appliance}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    # Cast to OmegaConf
    expes_config = OmegaConf.create(expes_config)

    # Define the path to save experiment results
    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}"
    )

    # Launch experiments
    launch_one_experiment(expes_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NILMFormer Experiments.")
    parser.add_argument(
        "--dataset", required=True, type=str, help="Dataset name (UKDALE, REFIT, or REDD)."
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate, e.g. '30s', '1min', '10min', etc.).",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help="Selected appliance, e.g., 'WashingMachine'.",
    )
    parser.add_argument(
        "--name_model", required=True, type=str, help="Name of the model for training."
    )
    parser.add_argument(
        "--seed", required=True, type=int, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--result_path",
        required=False,
        type=str,
        default=None,
        help="Output root, overriding result_path in configs/expes.yaml.",
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        seed=args.seed,
        result_path=args.result_path,
    )
