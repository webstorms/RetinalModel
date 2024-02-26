import torch

from ..base import NeuralDataset


class MacaqueMovie(NeuralDataset):
    
    _Y_NAMES = ['OFF Parasol 1_2020-10-20.pt',
                'OFF Parasol 1_2020-10-27.pt',
                'OFF Parasol 1_2020-10-28.pt',
                'OFF Parasol 1_2020-11-10.pt',
                'OFF Parasol 1_2020-11-18.pt',
                'OFF Parasol 1_2020-12-17.pt',
                'OFF Parasol 2_2020-10-20.pt',
                'OFF Parasol 2_2020-10-27.pt',
                'OFF Parasol 2_2020-11-10.pt',
                'OFF Parasol 2_2020-12-17.pt',
                'OFF Parasol 3_2020-10-27.pt',
                'OFF Parasol 4_2020-10-27.pt',
                'ON Parasol 1_2020-10-20.pt',
                'ON Parasol 1_2020-10-27.pt',
                'ON Parasol 1_2020-10-28.pt',
                'ON Parasol 1_2020-11-10.pt',
                'ON Parasol 1_2020-12-17.pt',
                'ON Parasol 2_2020-10-20.pt',
                'ON Parasol 2_2020-10-27.pt',
                'ON Parasol 2_2020-11-10.pt',
                'ON Parasol 3_2020-10-20.pt',
                'ON Parasol 3_2020-10-27.pt',
                'ON Parasol 3_2020-11-10.pt',
                'ON Parasol 4_2020-10-27.pt',
                ]

    def __init__(self, root, train, spatial_scale, max_dim, subclip_ms=None, pad=0, luminance=1, cell_idx=None):
        super().__init__(root, train, spatial_scale, max_dim, subclip_ms, pad, luminance, cell_idx, x_hz=60, y_hz=10000)

    @property
    def cc_max(self):
        return torch.Tensor([0.8895, 0.8910, 0.8897, 0.8893, 0.8873, 0.8909, 0.8904, 0.8906, 0.8895,
                             0.8916, 0.8906, 0.8882, 0.8902, 0.8878, 0.8819, 0.8802, 0.8921, 0.8892,
                             0.8884, 0.8673, 0.8894, 0.8894, 0.8760, 0.8888])

    def _get_train_mean_and_std(self):
        return 40.9699, 30.3615

    def _slice_out_fold(self):
        if self._train:
            # First 5 clips are training (72%)
            self._transformed_x = self._transformed_x[:5]
            self._transformed_y = self._transformed_y[:5]
        else:
            # Last 2 clips are testing (28%)
            self._transformed_x = self._transformed_x[5:]
            self._transformed_y = self._transformed_y[5:]

    def _load_x(self):
        # Slice out centre of movie (which was projected onto neuron)
        x = torch.load(f"{self._root}/data/neural/macaque_movie/tensors/x.pt")[:, :, 109:-110, 159:-160].float()
        x[x == 41.6646] = 40.9699

        return x

    def _load_y(self):
        y_list = []

        for y_name in MacaqueMovie._Y_NAMES:
            y_tensor = torch.load(f"{self._root}/data/neural/macaque_movie/tensors/{y_name}")
            y_list.append(y_tensor[:, :4])  # Constrain to 4 trials to build a valid tensors

        y_tensor = torch.stack(y_list).float()
        y_tensor = y_tensor.permute(1, 3, 2, 0)

        return y_tensor
