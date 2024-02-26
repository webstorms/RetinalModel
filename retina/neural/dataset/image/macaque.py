import torch

from ..base import NeuralDataset


class MacaqueImage(NeuralDataset):

    _Y_NAMES = ["off parasol1",
                  "off parasol2",
                  "off parasol6",
                  "off parasol7",
                  "off parasol8",
                  "off parasol9",
                  "off parasol10",
                  "on parasol1",
                  "on parasol2",
                  "on parasol3",
                  "on parasol4",
                  "on parasol5",
                  "on parasol6",
                  "on parasol7",
                  "on parasol8",
                  "on parasol9"
                  ]

    def __init__(self, root, train, spatial_scale, max_dim, subclip_ms=None, pad=0, luminance=1, cell_idx=None):
        super().__init__(root, train, spatial_scale, max_dim, subclip_ms, pad, luminance, cell_idx, x_hz=60, y_hz=10000)

    @property
    def cc_max(self):
        return torch.Tensor([0.9105, 0.9044, 0.9105, 0.9108, 0.9087, 0.9087, 0.9103, 0.9042, 0.9073,
                             0.9066, 0.9009, 0.8994, 0.9014, 0.9120, 0.9060, 0.9089])

    def _get_train_mean_and_std(self):
        return 40.9658, 10.3062

    def _slice_out_fold(self):
        if self._train == "all":
            return

        if self._train:
            # First 38 images are training (80%)
            self._transformed_x = self._transformed_x[:38]
            self._transformed_y = self._transformed_y[:38]
        else:
            # Last 10 images are testing (20%)
            self._transformed_x = self._transformed_x[38:]
            self._transformed_y = self._transformed_y[38:]

    def _load_x(self):
        # Slice out centre of movie (which was projected onto neuron)
        x = torch.load(f"{self._root}/data/neural/macaque_image/tensors/x.pt").float()
        x[x == 41.6646] = 40.9658

        return x

    def _load_y(self):
        y_list = []

        for y_name in MacaqueImage._Y_NAMES:
            y_tensor = torch.load(f"{self._root}/data/neural/macaque_image/tensors/{y_name}.pt").float()
            y_list.append(y_tensor)

        y_tensor = torch.stack(y_list).float()
        y_tensor = y_tensor.permute(1, 3, 2, 0)
        y_tensor[:, :2500] = 0  # Remove spikes from first 250ms (not representative of stimulus response)

        return y_tensor
