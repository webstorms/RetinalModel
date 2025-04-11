import torch
import torch.nn as nn
import numpy as np
import devtorch


class FastSigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, scale=100):
        ctx.scale = scale
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2

        return grad, None


class SNN(devtorch.DevModel):

    def __init__(self, n_out, init_beta):
        super().__init__()

        # Membrane
        self._beta = nn.Parameter(data=torch.clamp(torch.normal(init_beta, 0.01, (n_out,)), 0, 1), requires_grad=True)

        # Recurrent
        self._no_self_connection_mask = nn.Parameter(self._build_no_self_connection_mask(n_out), requires_grad=False)
        self._recurrent_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=True)

        # Initialise weights
        self.init_weight(self._recurrent_weight, "uniform", a=-0.1 / np.sqrt(n_out), b=0.1 / np.sqrt(n_out))

    @property
    def beta(self):
        return torch.clamp(self._beta, min=0.001, max=0.999)

    @property
    def recurrent_weight(self):
        return self._recurrent_weight * self._no_self_connection_mask

    def forward(self, x, mode="train", ablate_recurrence=False, ablate_neurons=False):
        # x: b x n x t
        mem_list = []
        spike_list = []
        abs_rec_current_list = []

        spikes = torch.zeros_like(x).to(x.device)[:, :, 0]
        mem = torch.zeros_like(x).to(x.device)[:, :, 0]

        for t in range(x.shape[2]):
            input_current = x[:, :, t]

            # Get recurrent currents
            rec_current = torch.einsum("ij, bj... -> bi...", self.recurrent_weight, spikes)
            abs_rec_current = torch.einsum("ij, bj... -> bi...", self.recurrent_weight.abs(), spikes)
            abs_rec_current_list.append(abs_rec_current)

            if not ablate_recurrence:
                input_current = input_current + rec_current

            # Update membrane potentials
            new_mem = torch.einsum("bn..., n -> bn...", mem, self.beta) + input_current

            # Output spikes
            spikes = FastSigmoid.apply(new_mem - 1)
            mem = new_mem * (1 - spikes.detach())
            spike_list.append(spikes)

            # Validation mode variables
            if mode == "val":
                mem_list.append(new_mem)

        if mode == "train" or mode == "val_curr":
            return torch.stack(spike_list, dim=2), torch.stack(abs_rec_current_list, dim=2)
        elif mode == "val":
            return torch.stack(spike_list, dim=2), torch.stack(mem_list, dim=2), torch.stack(abs_rec_current_list, dim=2)
        elif mode == "just_spikes":
            return torch.stack(spike_list, dim=2),

    def _build_no_self_connection_mask(self, n_out):
        no_self_connection_mask = torch.ones(n_out, n_out)
        for i in range(n_out):
            no_self_connection_mask[i, i] = 0

        return no_self_connection_mask