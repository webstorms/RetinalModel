import os
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from brainbox.transforms import GaussianKernel

from retina.neural.dataset import loader, base


class PCBuilder:

    def __init__(self, root, lam, pred_ms, model, train_dataset, test_dataset, n_pca, pad_t_len=29, stride=4, repeats=5):
        self._root = root
        self._lam = lam
        self._pred_ms = pred_ms
        self._model = model.cpu()
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._n_pca = n_pca
        self._pad_t_len = pad_t_len
        self._stride = stride
        self._repeats = repeats

        self._pca = IncrementalPCA(n_components=self._n_pca)

    @staticmethod
    def model_to_str(lam, noise_type, photo_noise, ganglion_noise, pred_ms, decoder_span):
        return f"{lam}_{noise_type}_{photo_noise}_{ganglion_noise}_{pred_ms}_{decoder_span}"

    @staticmethod
    def dataset_to_str(name, spatial_scale, max_dim, luminance):
        return f"{name}_{spatial_scale}_{max_dim}_{luminance}"

    @property
    def output_path(self):
        model_name = PCBuilder.model_to_str(self._lam, self._model.params.noise_type, self._model.params.photo_noise, self._model.params.ganglion_noise, self._pred_ms, self._model.params.decoder_span)
        dataset_name = PCBuilder.dataset_to_str(self._train_dataset.__class__.__name__, self._train_dataset._spatial_scale, self._train_dataset._max_dim, self._train_dataset._luminance)

        return f"{self._root}/data/fits_{self._n_pca}/{dataset_name}/{model_name}"

    def fit_and_transform(self):
        if os.path.exists(self.output_path):
            print(f"Already built for {self.output_path}")
            return

        Path(self.output_path).mkdir(parents=True, exist_ok=False)

        print(f"Building activity...")
        train_activity = self.generate_model_outputs(self._train_dataset, self._pad_t_len)
        test_activity = self.generate_model_outputs(self._test_dataset, self._pad_t_len)

        print(f"PCA fitting...")
        self.fit_pca(train_activity)

        print("Building PCs...")
        train_pcs = self.get_pcs(train_activity)
        test_pcs = self.get_pcs(test_activity)

        print("Saving PCs...")
        torch.save(train_pcs, f"{self.output_path}/train_pc.pt")
        torch.save(test_pcs, f"{self.output_path}/test_pc.pt")

    def generate_model_outputs(self, dataset, pad_t_len):
        activity_list = []
        n_samples = dataset._transformed_x.shape[0]

        for sample_idx in range(n_samples):
            print(f"Sampling for clip {sample_idx}...")

            clip = dataset._transformed_x[sample_idx]
            clip = F.pad(clip, (0, 0, 0, 0, pad_t_len, 0))
            clip = clip.unsqueeze(0).unsqueeze(0)   # Add channel dim

            clip_activity_list = []

            with torch.no_grad():
                for trial_idx in range(self._repeats):
                        activity = self._model(clip, mode="just_spikes", stride=self._stride)
                        activity = activity.permute(0, 2, 1, 3, 4)
                        activity = activity.flatten(2, 4)
                        clip_activity_list.append(activity)
                clip_activity = torch.cat(clip_activity_list)
                clip_activity = clip_activity.mean(0)  # Average over trials
                activity_list.append(clip_activity)

        activity = torch.stack(activity_list, dim=0)

        return activity

    def fit_pca(self, dataset):
        dataset = dataset.flatten(0, 1)

        n_samples = int(dataset.shape[0] / self._n_pca)

        for i in range(n_samples):
            if i == n_samples - 1:
                subset = dataset[i * self._n_pca:]
            else:
                subset = dataset[i * self._n_pca: (i+1) * self._n_pca]
            self._pca.partial_fit(subset)

    def get_pcs(self, dataset):
        frame_list = []

        for t in range(dataset.shape[1]):
            frame = dataset[:, t]
            pc = self._pca.transform(frame)
            pc = torch.from_numpy(pc)
            frame_list.append(pc)

        return torch.stack(frame_list, dim=1)


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, root, train, n_pca, cell_idx, subclip_ms):
        self._root = root
        self._train = train
        self._n_pca = n_pca
        self._cell_idx = cell_idx
        self._subclip_ms = subclip_ms

        self._gk = GaussianKernel(9, 101)  # 9 bins is approx 37ms
        self._dataset_name = None
        self._idxs = None

        self.cc_max = None

    def __getitem__(self, i):
        clip_idx, t_start_idx, t_end_idx = self._idxs[i]

        x = self._x[clip_idx, t_start_idx: t_end_idx]
        y = self._y[clip_idx, t_start_idx: t_end_idx]

        x = x.unsqueeze(0)  # Add channel dimension
        y = y.mean(1)  # Average over trials to get spike probability

        if self._cell_idx is not None:
            y = y[:, self._cell_idx].unsqueeze(1)

        return x, y

    def __len__(self):
        return len(self._idxs)

    @property
    def n_neurons(self):
        return self._y.shape[-1]

    @property
    def hyperparams(self):
        return {}

    def load_model(self, lam, noise_type, photo_noise, ganglion_noise, pred_ms, decoder_span):
        assert self._dataset_name is not None, "Need to call load_responses before calling load_model"
        model_name = PCBuilder.model_to_str(lam, noise_type, photo_noise, ganglion_noise, pred_ms, decoder_span)
        output_path = f"{self._root}/data/fits_{self._n_pca}/{self._dataset_name}/{model_name}"

        # Load x
        train_pc = torch.load(f"{output_path}/train_pc.pt").float()
        test_pc = torch.load(f"{output_path}/test_pc.pt").float()

        # Normalise
        mean, std = train_pc.mean(), train_pc.std()
        train_pc = (train_pc - mean) / train_pc.std()
        test_pc = (test_pc - mean) / train_pc.std()

        self._x = train_pc if self._train else test_pc
        assert self._x.shape[0] == self._y.shape[0]  # Same number of clips
        #assert self._x.shape[1] == self._y.shape[1]  # Same number of time steps

        self._idxs = base.NeuralDataset.build_idxs(self._subclip_ms, self._x.shape[0], self._x.shape[1])

    def load_responses(self, dataset, spatial_scale, max_dim, luminance):
        self._dataset_name = PCBuilder.dataset_to_str(dataset, spatial_scale, max_dim, luminance)

        # Load y
        standard_dataset = loader.load(self._root, self._train, dataset, spatial_scale, max_dim, luminance, None)
        self.cc_max = standard_dataset.cc_max
        self._raw_y = standard_dataset._transformed_y

        self._y = self._raw_y.mean(2)
        self._y = self._gk(self._y.unsqueeze(1))[:, 0]
        self._y = self._y.unsqueeze(-2)
