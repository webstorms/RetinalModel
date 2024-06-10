import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import devtorch

from .snn import SNN


class RetinaModel(devtorch.DevModel):
    
    def __init__(self, params):
        super().__init__()
        self.params = params

        self._encoder_bias = nn.Parameter(torch.rand(params.n_in))
        self._decoder_bias = nn.Parameter(torch.rand(1))
        self._encoder_weight = nn.Parameter(torch.rand(params.n_in, 1, params.encoder_span, params.rf_size, params.rf_size), requires_grad=True)
        self._decoder_weight = nn.Parameter(torch.rand(params.n_in, params.decoder_span, params.rf_size, params.rf_size), requires_grad=True)

        init_beta = np.exp(-params.dt / params.mem_tc)
        self._neurons = SNN(params.n_in, init_beta)

        # Initialise weights
        k_encoder = params.encoder_span * params.rf_size * params.rf_size
        self.init_weight(self._encoder_weight, "uniform", a=-1 / np.sqrt(k_encoder), b=1 / np.sqrt(k_encoder))
        self.init_weight(self._encoder_bias, "constant", val=0)

        k_decoder = params.n_in * params.decoder_span
        self.init_weight(self._decoder_weight, "uniform", a=-1 / np.sqrt(k_decoder), b=1 / np.sqrt(k_decoder))
        self.init_weight(self._decoder_bias, "constant", val=0)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "params": self.params.hyperparams}

    def forward(self, x, mode="train", stride=1):
        # x: b x n x t x h x w
        if mode in ["train", "val"]:
            assert x.shape[-1] == x.shape[-2]
            assert x.shape[-1] == self.params.rf_size

        # Add noise to input (photo receptors are noisy)
        x = x + torch.normal(0, self.params.photo_noise, size=x.shape).to(x.device)

        # Ganglion transformation
        # Compute input current
        input_current = F.conv3d(x, self._encoder_weight, self._encoder_bias, stride=(1, stride, stride))
        abs_input_current = F.conv3d(x.abs(), self._encoder_weight.abs(), self._encoder_bias.abs(), stride=(1, stride, stride))

        # Add noise to input current
        noise = self.params.ganglion_noise * torch.normal(0, 1, size=input_current.shape).to(x.device)
        if self.params.noise_type == "+":
            input_current = input_current + noise
        elif self.params.noise_type == "*":
            input_current = input_current * (1 + noise)

        # Obtain neuron outputs
        if mode == "just_spikes":
            # This will return spikes over all spatial locations
            return self._neurons(input_current[:, :, :], mode)[0]

        neuron_outputs = self._neurons(input_current[:, :, :, 0, 0], mode)
        spikes = neuron_outputs[0]

        # Decode neuron spikes to output frame
        output = self._spikes_to_predicted_clip(spikes, self._decoder_weight)
        abs_output = self._spikes_to_predicted_clip(spikes, self._decoder_weight.abs())

        if mode == "train":
            abs_graded_current = abs_input_current.mean()
            abs_rec_current = neuron_outputs[1]
            abs_spiking_current = abs_output.mean() + abs_rec_current.mean()

            return output, abs_graded_current, abs_spiking_current
        elif mode == "val":
            mem = neuron_outputs[1]
            abs_rec = neuron_outputs[2]

            return output, spikes, mem, abs_rec, input_current

    def _spikes_to_predicted_clip(self, spikes, decoder_weight):
        _, _, sim_length = spikes.shape
        output_list = []

        for t in range(sim_length - self.params.decoder_span):
            output = torch.einsum("bnt, nthw -> bhw", spikes[:, :, t:t+self.params.decoder_span], decoder_weight) + self._decoder_bias
            output_list.append(output)

        return torch.stack(output_list, dim=1).unsqueeze(1)
