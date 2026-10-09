#################################################################################################################
#
# @description : Trainable TCN baseline for the curation/augmentation/KL ablation
#
# Same architecture as the externally pretrained TCN_KL (src/baselines/nilm/tcn_kl.py),
# but trained inside this repo's pipeline like every other baseline: single appliance
# head, aggregate already scaled by NILMscaler, (B, 1, T) in and out.
#
#################################################################################################################

import numpy as np
import torch
import torch.nn as nn

from src.baselines.nilm.tcn_kl import KLFilter


# ======================= Temporal Convolutional Network (Bai et al., 2018) =======================#
class TCN_NILM(nn.Module):
    def __init__(
        self,
        window_size,
        c_in=1,
        use_kl=False,
        kl_basis=None,
        num_channels=(32, 32, 32, 32),
        kernel_size=4,
    ):
        """
        TCN seq2seq NILM model, optionally fed a Karhunen-Loeve expansion of the input.

        Architecture matches TCN_KL_Core: pytorch_tcn defaults are kept (causal=True,
        use_norm="weight_norm", dropout=0.1), as in the external training script which
        also passes none of them.

        - use_kl   : prepend `kl_order` Karhunen-Loeve channels to the raw aggregate
        - kl_basis : (order, order) basis fitted upstream by tcn_ablation.fit_kl_basis
        """
        super().__init__()
        from pytorch_tcn import TCN

        self.use_kl = use_kl

        if use_kl:
            if kl_basis is None:
                raise ValueError("TCN_NILM: use_kl=True requires a fitted kl_basis.")
            # model_kwargs arrives as an OmegaConf ListConfig, which np.asarray
            # would turn into an object array.
            basis = np.array(
                [[float(v) for v in row] for row in kl_basis], dtype=np.float32
            )
            self.kl_filter = KLFilter(basis)
            num_inputs = basis.shape[0] + 1
        else:
            self.kl_filter = None
            num_inputs = c_in

        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            input_shape="NCL",
            output_projection=1,
            output_activation=None,
        )

    def forward(self, x):
        # x: (B, 1, T) — NILMscaler-scaled aggregate, no exogenous channels
        if self.use_kl:
            x = torch.cat([self.kl_filter(x), x], dim=1)
        return self.tcn(x)
