import h5py
import torch
import numpy as np
import pandas as pd
from brainbox import transforms

from ..base import NeuralDataset


class SalamanderImage(NeuralDataset):

    def __init__(self, root, train, spatial_scale, max_dim, subclip_ms=None, pad=0, luminance=1, cell_idx=None):
        data = h5py.File(f"{root}/data/neural/salamander_image/responses2naturalimages.h5", "r")
        trial_numbers = pd.read_csv(f"{root}/data/neural/salamander_image/trialnumbers.txt", header=None)
        trial_query = (trial_numbers >= 12).values.flatten()
        images = np.array(data["images"])
        images = np.rot90(images, axes=(1, 2))
        images = torch.from_numpy(images.copy()).float()[:300]  # Only natural images
        images = torch.clamp(images, min=-1, max=1)

        spikes = np.array(data["spikes"])
        spikes = torch.from_numpy(spikes).float()[:, :300]  # Only natural images
        spikes = spikes.permute(1, 3, 2, 0)  # Get into order clips x time x repeat x neuron
        spikes = spikes[..., trial_query][:, :, :12]  # Constrain to 12 trials to build a valid tensors

        self._images = images
        self._spikes = spikes
        self._img_to_clip_transform = transforms.ImgToClip(0, 200, 100, c=0)

        super().__init__(root, train, spatial_scale, max_dim, subclip_ms, pad, luminance, cell_idx, x_hz=1000, y_hz=1000)

    @property
    def cc_max(self):
        return torch.Tensor([0.9524, 0.9518, 0.9514, 0.9503, 0.9470, 0.9533, 0.9557, 0.9528, 0.9546,
                             0.9543, 0.9557, 0.9414, 0.9490, 0.9530, 0.9524, 0.9473, 0.9504, 0.9430,
                             0.9491, 0.9516, 0.9528, 0.9566, 0.9539, 0.9463, 0.9560, 0.9568, 0.9534,
                             0.9563, 0.9474, 0.9291, 0.9514, 0.9512, 0.9504, 0.9525, 0.9550, 0.9482,
                             0.9558, 0.9534, 0.9527, 0.9489, 0.9538, 0.9569, 0.9536, 0.9588, 0.9531,
                             0.9547, 0.9558, 0.9580, 0.9420, 0.9543, 0.9548, 0.9564, 0.9563, 0.9499,
                             0.9586, 0.9539, 0.9531, 0.9555, 0.9544, 0.9570, 0.9460, 0.9487, 0.9571,
                             0.9539, 0.9532, 0.9520, 0.9572, 0.9515, 0.9555, 0.9549, 0.9567, 0.9575,
                             0.9567, 0.9578, 0.9567, 0.9564, 0.9514, 0.9563, 0.9557, 0.9559, 0.9553,
                             0.9560, 0.9575, 0.9560, 0.9570, 0.9571, 0.9558, 0.9546, 0.9566, 0.9569,
                             0.9500, 0.9575, 0.9548, 0.9557, 0.9520, 0.9550, 0.9560, 0.9574, 0.9576,
                             0.9553, 0.9559, 0.9561, 0.9580, 0.9531, 0.9567, 0.9557, 0.9553, 0.9551,
                             0.9478, 0.9571, 0.9557, 0.9558, 0.9562, 0.9550, 0.9574, 0.9581, 0.9563,
                             0.9571, 0.9537, 0.9561])

    def _get_train_mean_and_std(self):
        return -0.0131, 0.4705

    def _slice_out_fold(self):
        if self._train == "all":
            return

        if self._train:
            # First 240 images are training (80%)
            self._transformed_x = self._transformed_x[:240]
            self._transformed_y = self._transformed_y[:240]
        else:
            # Last 60 images are testing (20%)
            self._transformed_x = self._transformed_x[240:]
            self._transformed_y = self._transformed_y[240:]

    def _load_x(self):
        clips = self._img_to_clip_transform(self._images.unsqueeze(1))[:, 0]
        clips[:, -100:] = -0.0131

        return clips

    def _load_y(self):
        return self._spikes
