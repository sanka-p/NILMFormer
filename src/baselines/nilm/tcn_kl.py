import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLFilter(nn.Module):
    """
    PyTorch port of KL_FIlter.__call__ from TCN_Better_Embeddings.py.
    Applies KL eigenvector basis via sliding window of size `order`.
    Padding is right-side only (mirrors np.pad(signal, (0, order-1))).
    """

    def __init__(self, basis: np.ndarray):
        super().__init__()
        self.order = basis.shape[0]
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (B, order, T)
        x_padded = F.pad(x, (0, self.order - 1))
        x_unfold = x_padded.unfold(-1, self.order, 1)  # (B, 1, T, order)
        kl = x_unfold.squeeze(1) @ self.basis           # (B, T, order)
        return kl.permute(0, 2, 1)                      # (B, order, T)


class TCN_KL_Core(nn.Module):
    """
    Mirrors TCN_Model_Output_Conditioning with device_conv1/2 bypassed
    (forward returns tcn output directly, matching the saved-weight forward path).
    """

    def __init__(self, kl_order: int, n_appliances: int):
        super().__init__()
        from pytorch_tcn import TCN
        self.tcn = TCN(
            num_inputs=kl_order + 1,
            num_channels=[32] * 4,
            kernel_size=4,
            input_shape="NCL",
            output_projection=n_appliances,
            output_activation=None,
        )
        # device_conv1/2 present to match saved state_dict keys but never called
        self.device_conv1 = nn.Conv1d(n_appliances, n_appliances, bias=False,
                                       kernel_size=5, padding="same", groups=n_appliances)
        self.device_conv2 = nn.Conv1d(n_appliances, n_appliances, bias=False,
                                       kernel_size=5, padding="same", groups=n_appliances)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


class TCN_KL_NILMFormerAdapter(nn.Module):
    """
    Drop-in replacement for `pred = model(ts_agg)` in SeqToSeqTrainer.evaluate.

    Undoes NILMFormer's aggregate normalization, runs TCN+KL inference,
    multiplies by AVERAGE_POWERS, slices the target appliance channel,
    then re-applies NILMFormer's appliance normalization so the output
    lives in the same scaled space as `target`.
    """

    def __init__(
        self,
        core: TCN_KL_Core,
        kl_filter: KLFilter,
        tcn_appliance_names: list,
        tcn_average_powers: list,
        target_appliance: str,
        nilmformer_scaler,
        input_norm_const: float = 5000.0,
        appliance_name_map: dict = None,
    ):
        super().__init__()
        self.core = core
        self.kl_filter = kl_filter
        self.input_norm_const = input_norm_const

        mapped = (appliance_name_map or {}).get(target_appliance, target_appliance)
        if mapped not in tcn_appliance_names:
            raise ValueError(
                f"Appliance '{target_appliance}' (mapped to '{mapped}') not in TCN appliances: "
                f"{tcn_appliance_names}. Add an entry to appliance_name_map in configs/models.yaml."
            )
        self.appliance_idx = tcn_appliance_names.index(mapped)

        self.register_buffer(
            "avg_powers",
            torch.tensor(tcn_average_powers, dtype=torch.float32),
        )

        # Capture scaler stats as plain floats to avoid device mismatches
        self.power_stat1 = float(nilmformer_scaler.power_stat1)
        self.power_stat2 = float(nilmformer_scaler.power_stat2)
        self.appliance_stat1 = float(nilmformer_scaler.appliance_stat1[0])
        self.appliance_stat2 = float(nilmformer_scaler.appliance_stat2[0])

    def forward(self, ts_agg: torch.Tensor) -> torch.Tensor:
        # ts_agg: (B, c_in, T) — channel 0 is NILMFormer-scaled aggregate
        agg_scaled = ts_agg[:, 0:1, :]
        agg_watts = agg_scaled * self.power_stat2 + self.power_stat1
        x = agg_watts / self.input_norm_const
        x_kl = torch.cat([self.kl_filter(x), x], dim=1)   # (B, kl_order+1, T)
        pred_norm = F.relu(self.core(x_kl))                # (B, K, T)
        pred_watts = pred_norm * self.avg_powers.unsqueeze(0).unsqueeze(-1)
        k = self.appliance_idx
        pred_watts = pred_watts[:, k : k + 1, :]           # (B, 1, T)
        return (pred_watts - self.appliance_stat1) / self.appliance_stat2


def load_pretrained(weights_path: str, meta_path: str = None):
    """
    Load TCN_KL_Core and KLFilter from a Lightning checkpoint plus sidecar metadata.

    Metadata lookup order:
      1. meta_path argument (JSON or .pt)
      2. ckpt["meta"]
      3. ckpt["hyper_parameters"]

    Required metadata keys:
      kl_basis, appliance_names, average_powers, kl_order, n_appliances
    Optional:
      input_norm_const (default 5000.0)

    Returns (core, kl_filter, meta_dict).
    """
    import json

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    meta = None
    if meta_path is not None:
        if str(meta_path).endswith(".json"):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    elif "meta" in ckpt:
        meta = ckpt["meta"]
    elif "hyper_parameters" in ckpt:
        meta = ckpt["hyper_parameters"]

    if meta is None:
        raise ValueError(
            "No metadata found in checkpoint and no meta_path provided. "
            "Create a sidecar .json or .pt file with keys: "
            "kl_basis, appliance_names, average_powers, kl_order, n_appliances"
        )

    required = {"kl_basis", "appliance_names", "average_powers", "kl_order", "n_appliances"}
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"Metadata missing required keys: {missing}")

    kl_order = int(meta["kl_order"])
    n_appliances = int(meta["n_appliances"])

    kl_filter = KLFilter(np.array(meta["kl_basis"]))
    core = TCN_KL_Core(kl_order=kl_order, n_appliances=n_appliances)

    state_dict = ckpt.get("state_dict", ckpt)
    stripped = {}
    for k, v in state_dict.items():
        key = k[len("model."):] if k.startswith("model.") else k
        stripped[key] = v

    missing_keys, unexpected_keys = core.load_state_dict(stripped, strict=False)
    if missing_keys:
        logging.warning("TCN_KL missing keys: %s", missing_keys)
    if unexpected_keys:
        logging.debug("TCN_KL unexpected keys (likely device_conv): %s", unexpected_keys)

    logging.info(
        "TCN_KL loaded. n_appliances=%d, kl_order=%d, appliances=%s",
        n_appliances, kl_order, meta["appliance_names"],
    )

    core.eval()
    kl_filter.eval()

    return core, kl_filter, dict(meta)
