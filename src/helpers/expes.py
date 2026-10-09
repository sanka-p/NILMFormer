#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Experiments Helpers
#
#################################################################################################################

import csv
import os
import torch
import logging

import torch.nn as nn
from omegaconf import OmegaConf

from src.helpers.trainer import SeqToSeqTrainer, TserTrainer
from src.helpers.dataset import NILMDataset, TSDatasetScaling
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation


# ==== SotA NILM baselines ==== #
# Recurrent-based
from src.baselines.nilm.bilstm import BiLSTM
from src.baselines.nilm.bigru import BiGRU

# Conv-based
from src.baselines.nilm.fcn import FCN
from src.baselines.nilm.cnn1d import CNN1D
from src.baselines.nilm.unetnilm import UNetNiLM
from src.baselines.nilm.dresnets import DAResNet, DResNet
from src.baselines.nilm.diffnilm import DiffNILM
from src.baselines.nilm.tsilnet import TSILNet
from src.baselines.nilm.tcn import TCN_NILM

# Transformer-based
from src.baselines.nilm.bert4nilm import BERT4NILM
from src.baselines.nilm.stnilm import STNILM
from src.baselines.nilm.energformer import Energformer


# ==== SotA TSER baselines ==== #
from src.baselines.tser.convnet import ConvNet
from src.baselines.tser.resnet import ResNet
from src.baselines.tser.inceptiontime import Inception

# ==== NILMFormer ==== #
from src.nilmformer.congif import NILMFormerConfig
from src.nilmformer.model import NILMFormer


#: Ablation arms sharing the trainable TCN. Their extra model_kwargs steer data
#: preparation in scripts/run_one_expe.py and are stripped before reaching the
#: constructor by launch_models_training.
TCN_ABLATION_MODELS = (
    "TCN", "TCN_KL_scratch", "TCN_KL_aug", "TCN_KL_aug_curated", "TCN_aug_curated",
    # curated-pool-size sweep: identical to TCN_KL_aug_curated but with the target
    # appliance's activation pool capped, to locate where curation stops helping.
    "TCN_KL_aug_curated_n20", "TCN_KL_aug_curated_n50", "TCN_KL_aug_curated_n100", "TCN_KL_aug_curated_n200",
)

#: Externally pretrained TCNs loaded from a Lightning checkpoint and never trained
#: here. TCN_KL is the 8-output model; TCN_KL_perapp is the matching single-output
#: checkpoint trained per appliance, so it compares at the same head width as the
#: in-repo ablation arms.
PRETRAINED_TCN_MODELS = ("TCN_KL", "TCN_KL_perapp", "TCN_KL_perapp_nilmformerlike")
TCN_ABLATION_DATA_KWARGS = (
    "kl_order", "augment", "aug_mode", "aug_ratio", "segments_source", "max_segments", "max_segments_target",
)


def get_model_instance(name_model, c_in, window_size, **kwargs):
    """
    Get model instances
    """
    if name_model == "BiGRU":
        inst = BiGRU(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "BiLSTM":
        inst = BiLSTM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "CNN1D":
        inst = CNN1D(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "UNetNILM":
        inst = UNetNiLM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "FCN":
        inst = FCN(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "BERT4NILM":
        inst = BERT4NILM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "STNILM":
        inst = STNILM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "BERT4NILM":
        inst = BERT4NILM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "DResNet":
        inst = DResNet(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "DAResNet":
        inst = DAResNet(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "DiffNILM":
        inst = DiffNILM(**kwargs)
    elif name_model == "TSILNet":
        inst = TSILNet(c_in=1, window_size=window_size, **kwargs)
    elif name_model == "Energformer":
        inst = Energformer(c_in=1, **kwargs)
    elif name_model == "ConvNet":
        inst = ConvNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "ResNet":
        inst = ResNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "Inception":
        inst = Inception(in_channels=1, nb_class=1, **kwargs)
    elif name_model == "NILMFormer":
        inst = NILMFormer(NILMFormerConfig(c_in=1, c_embedding=c_in - 1, **kwargs))
    elif name_model in TCN_ABLATION_MODELS:
        inst = TCN_NILM(window_size=window_size, c_in=1, **kwargs)
    else:
        raise ValueError("Model name {} unknown".format(name_model))

    return inst


_TIMING_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "timing.csv")


def _append_timing_csv(experiment_name, train_time, eval_time):
    write_header = not os.path.exists(_TIMING_CSV)
    with open(_TIMING_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["experiment", "train_time", "eval_time"])
        writer.writerow([experiment_name, train_time, eval_time])


def nilm_model_training(inst_model, tuple_data, scaler, expes_config):
    if expes_config.name_model == "NILMFormer":
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=expes_config.list_exo_variables,
            freq=expes_config.sampling_rate,
        )

        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=expes_config.list_exo_variables,
            freq=expes_config.sampling_rate,
        )

        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=expes_config.list_exo_variables,
            freq=expes_config.sampling_rate,
        )
    elif expes_config.name_model == "DiffNILM":
        train_dataset = NILMDataset(
            tuple_data[0],
            st_date=tuple_data[4],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        valid_dataset = NILMDataset(
            tuple_data[1],
            st_date=tuple_data[5],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
        test_dataset = NILMDataset(
            tuple_data[2],
            st_date=tuple_data[6],
            list_exo_variables=["hour", "dow", "month"],
            freq=expes_config.sampling_rate,
            cosinbase=False,
            newRange=(-0.5, 0.5),
        )
    else:
        train_dataset = NILMDataset(tuple_data[0])
        valid_dataset = NILMDataset(tuple_data[1])
        test_dataset = NILMDataset(tuple_data[2])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=expes_config.batch_size, shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_trainer = SeqToSeqTrainer(
        inst_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        learning_rate=expes_config.model_training_param.lr,
        weight_decay=expes_config.model_training_param.wd,
        criterion=nn.MSELoss(),
        f_metrics=NILMmetrics(),
        training_in_model=expes_config.model_training_param.training_in_model,
        patience_es=expes_config.p_es,
        patience_rlr=expes_config.p_rlr,
        n_warmup_epochs=expes_config.n_warmup_epochs,
        verbose=True,
        plotloss=False,
        save_fig=False,
        path_fig=None,
        device=expes_config.device,
        all_gpu=expes_config.all_gpu,
        save_checkpoint=True,
        path_checkpoint=expes_config.result_path,
    )

    if expes_config.name_model in PRETRAINED_TCN_MODELS:
        logging.info("Skipping training (pretrained %s)", expes_config.name_model)
    else:
        logging.info("Model training...")
        model_trainer.train(expes_config.epochs)
        model_trainer.restore_best_weights()

    logging.info("Eval model...")
    model_trainer.evaluate(
        valid_loader,
        scaler=scaler,
        threshold_small_values=expes_config.threshold,
        save_outputs=True,
        mask="valid_metrics",
    )
    model_trainer.evaluate(
        test_loader,
        scaler=scaler,
        threshold_small_values=expes_config.threshold,
        save_outputs=True,
        mask="test_metrics",
    )

    # TODO: Update eval_win_energy_aggregation to support variable exogene data
    if expes_config.name_model == "DiffNILM":
        eval_win_energy_aggregation(
            tuple_data[2],
            tuple_data[6],
            model_trainer,
            scaler=scaler,
            metrics=NILMmetrics(round_to=5),
            window_size=expes_config.window_size,
            freq=expes_config.sampling_rate,
            list_exo_variables=["hour", "dow", "month"],
            cosinbase=False,
            newRange=(-0.5, 0.5),
            threshold_small_values=expes_config.threshold,
        )
    else:
        eval_win_energy_aggregation(
            tuple_data[2],
            tuple_data[6],
            model_trainer,
            scaler=scaler,
            metrics=NILMmetrics(round_to=5),
            window_size=expes_config.window_size,
            freq=expes_config.sampling_rate,
            list_exo_variables=expes_config.list_exo_variables
            if expes_config.name_model == "NILMFormer"
            else [],
            threshold_small_values=expes_config.threshold,
        )

    model_trainer.save()
    _append_timing_csv(
        expes_config.result_path,
        model_trainer.train_time,
        model_trainer.log.get("test_metrics_time"),
    )
    logging.info(
        "Training and eval completed! Model weights and log save at: {}.pt".format(expes_config.result_path)
    )


def tser_model_training(inst_model, tuple_data, scaler, expes_config):
    train_dataset = TSDatasetScaling(tuple_data[0][0], tuple_data[0][1])
    valid_dataset = TSDatasetScaling(tuple_data[1][0], tuple_data[1][1])
    test_dataset = TSDatasetScaling(tuple_data[2][0], tuple_data[2][1])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=expes_config.batch_size, shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_trainer = TserTrainer(
        inst_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        learning_rate=expes_config.model_training_param.lr,
        weight_decay=expes_config.model_training_param.wd,
        criterion=nn.MSELoss(),
        f_metrics=NILMmetrics(),
        patience_es=expes_config.p_es,
        patience_rlr=expes_config.p_rlr,
        n_warmup_epochs=expes_config.n_warmup_epochs,
        verbose=True,
        plotloss=False,
        save_fig=False,
        path_fig=None,
        device=expes_config.device,
        all_gpu=expes_config.all_gpu,
        save_checkpoint=True,
        path_checkpoint=expes_config.result_path,
    )

    logging.info("Train model...")
    model_trainer.train(expes_config.epochs)

    logging.info("Eval model...")
    model_trainer.restore_best_weights()
    model_trainer.evaluate(
        valid_loader,
        scaler=scaler,
        save_outputs=True,
        mask="valid_metrics",
    )

    model_trainer.evaluate(
        test_loader,
        scaler=scaler,
        save_outputs=True,
        mask="test_metrics",
    )

    _append_timing_csv(
        expes_config.result_path,
        model_trainer.train_time,
        model_trainer.log.get("test_metrics_time"),
    )
    logging.info(
        "Training and eval completed! Model weights and log save at: {}".format(
            expes_config.result_path
        )
    )


def launch_models_training(data_tuple, scaler, expes_config):
    if "cutoff" in expes_config.model_kwargs:
        expes_config.model_kwargs.cutoff = expes_config.cutoff

    if "threshold" in expes_config.model_kwargs:
        expes_config.model_kwargs.threshold = expes_config.threshold

    if expes_config.name_model in TCN_ABLATION_MODELS:
        # These drive data preparation upstream, not the module signature.
        for key in TCN_ABLATION_DATA_KWARGS:
            expes_config.model_kwargs.pop(key, None)

    if expes_config.name_model in PRETRAINED_TCN_MODELS:
        from src.baselines.nilm.tcn_kl import load_pretrained, TCN_KL_NILMFormerAdapter

        def _resolve_dataset_path(path_config, field_name):
            if OmegaConf.is_dict(path_config):
                if expes_config.dataset in path_config and path_config[expes_config.dataset]:
                    return path_config[expes_config.dataset]
                if "default" in path_config and path_config["default"]:
                    return path_config["default"]
                raise ValueError(
                    f"TCN_KL {field_name} is not configured for dataset '{expes_config.dataset}'."
                )
            if not path_config:
                raise ValueError(
                    f"TCN_KL {field_name} is not configured for dataset '{expes_config.dataset}'."
                )
            return path_config

        # Per-appliance checkpoints live in one directory per appliance, so the configured
        # path may carry an {app} placeholder. Dataset-keyed paths still work unchanged.
        # The checkpoint directory name does not always equal the repo's appliance key
        # (repo "dishwasher" vs directory "dish_washer"), so allow an explicit mapping.
        app_dir_map = dict(expes_config.model_kwargs.get("app_dir_map") or {})
        app_dir = app_dir_map.get(expes_config.app, expes_config.app)

        weights_path = _resolve_dataset_path(expes_config.model_kwargs.weights_path, "weights_path")
        weights_path = str(weights_path).format(app=app_dir)
        meta_path = expes_config.model_kwargs.get("meta_path")
        if meta_path is not None:
            meta_path = _resolve_dataset_path(meta_path, "meta_path")
            meta_path = str(meta_path).format(app=app_dir)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"{expes_config.name_model}: no checkpoint for appliance "
                f"'{expes_config.app}' at {weights_path}"
            )

        core, klf, meta = load_pretrained(
            weights_path=weights_path,
            meta_path=meta_path,
        )
        model_instance = TCN_KL_NILMFormerAdapter(
            core=core,
            kl_filter=klf,
            tcn_appliance_names=meta["appliance_names"],
            tcn_average_powers=meta["average_powers"],
            target_appliance=expes_config.app,
            nilmformer_scaler=scaler,
            appliance_name_map=dict(expes_config.model_kwargs.get("appliance_name_map") or {}),
            input_norm_const=float(meta.get("input_norm_const", 5000.0)),
        )
    else:
        model_instance = get_model_instance(
            name_model=expes_config.name_model,
            c_in=(1 + 2 * len(expes_config.list_exo_variables)),
            window_size=expes_config.window_size,
            **expes_config.model_kwargs,
        )

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        tser_model_training(model_instance, data_tuple, scaler, expes_config)
    else:
        nilm_model_training(model_instance, data_tuple, scaler, expes_config)

    del model_instance
