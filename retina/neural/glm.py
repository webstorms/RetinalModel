import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brainbox import models


class GLM(models.BBModel):

    MODEL_HZ = 240

    def __init__(self, n_in, n_out, h, w, rf_len_ms, pad=False, clamp=False):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._h = h
        self._w = w
        self._rf_len_ms = rf_len_ms
        self._pad = pad
        self._clamp = clamp

        frame_dt = 1000 / GLM.MODEL_HZ
        self.rf_len = round(rf_len_ms / frame_dt)
        self._rf_weight = nn.Parameter(torch.rand(n_out, n_in, self.rf_len, h, w), requires_grad=True)
        self._rf_bias = nn.Parameter(-2*torch.ones(n_out), requires_grad=True)

        self.init_weight(self._rf_weight, "uniform", a=-1 / np.sqrt(n_in * self.rf_len * h * w), b=1 / np.sqrt(n_in * self.rf_len * h * w))

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "h": self._h, "w": self._w, "rf_len_ms": self._rf_len_ms, "pad": self._pad}

    def get_params(self):
        return [self._rf_weight]

    def forward(self, x):
        if len(x.shape) == 4:
            # x: b x c x t x h (where h is n)
            x = x.unsqueeze(-1)  # Add w dim

        if self._pad:
            x = F.pad(x, (0, 0, 0, 0, self.rf_len - 1, 0))

        x = F.conv3d(x, self._rf_weight, bias=self._rf_bias)
        x = x[..., 0, 0]
        x = x.permute(0, 2, 1)
        x = torch.exp(x)

        if self._clamp:
            x = torch.clamp(x, 0, 1)

        return x
