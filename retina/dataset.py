import random

import h5py
import torch
import numpy as np
from brainbox.datasets.transforms import ClipRandomHorizontalFlip, ClipExtend


class PatchNaturalDataset(torch.utils.data.Dataset):

    LENGTH = 100000

    def __init__(self, root, train=True, temp_len=30, kernel=20, flip=True, n_frame_ext=1):
        self._root = root
        self._temp_len = temp_len
        self._kernel = kernel
        self._flip = flip
        self._n_frame_ext = n_frame_ext

        self._flip_transform = ClipRandomHorizontalFlip() if flip else None
        self._clip_extender_transform = ClipExtend(n_frame_ext) if n_frame_ext > 1 else None

        # Load and normalise dataset
        self._dataset = self._load_dataset(train=train)
        self._dataset.sub_(115.97)
        self._dataset.divide_(63.60)

        # Build idxs
        self._idxs = self._build_index()

    def __getitem__(self, i):
        b_idx, t_idx, h_idx, w_idx = self._idxs[random.randint(0, len(self._idxs)-1)]

        clip = self._dataset[b_idx,
               :,
               t_idx*self._temp_len: (t_idx + 1) * self._temp_len,
               h_idx: h_idx + self._kernel,
               w_idx: w_idx + self._kernel,
               ]

        if self._flip:
            clip = self._flip_transform(clip)

        if self._clip_extender_transform is not None:
            clip = self._clip_extender_transform(clip)

        return clip, clip

    def __len__(self):
        return PatchNaturalDataset.LENGTH

    @property
    def hyperparams(self):
        return {"dt": self._temp_len, "kernel": self._kernel, "flip": self._flip, "ext_frames": self._n_frame_ext}

    def _load_dataset(self, train):
        hf = h5py.File(f"{self._root}/natural.hdf5", "r")
        dataset_name = "train" if train else "test"
        dataset = np.array(hf.get(dataset_name))
        hf.close()

        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset

    def _build_index(self):
        n_batch = self._dataset.shape[0]
        n_steps = int(self._dataset.shape[2] / self._temp_len)
        n_h = self._dataset.shape[3] - self._kernel + 1
        n_w = self._dataset.shape[4] - self._kernel + 1

        self._idxs = []

        for b_idx in range(n_batch):
            for t_idx in range(n_steps):
                for h_idx in range(n_h):
                    for w_idx in range(n_w):
                        self._idxs.append((b_idx, t_idx, h_idx, w_idx))

        return self._idxs
