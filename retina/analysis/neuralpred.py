import torch
import numpy as np

import retina.train as trainer
import retina.neural.dataset.loader as dataset_loader
import retina.neural.analysis as readout_loader


class ResponseSampler:

    def __init__(self, root, train, dataset_name, clip_idx, spatial_scale=0.4, max_dim=80.0, toff=64, rf_len_ms=21):
        self._clip_idx = clip_idx

        self._model = trainer.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{toff}_8")
        self._readout = readout_loader.FitMetricsBuilder(root, dataset_name, is_train=train, pred_ms=toff, spatial_scale=spatial_scale, max_dim=max_dim, luminance=1, rf_len_ms=rf_len_ms)
        self._dataset = dataset_loader.load(root, train, dataset_name, spatial_scale, max_dim, luminance=1, cell_idx=None)

        # Get spike responses
        # Note, here we just slice out a 20x20 patch, but actually convolve over the input when computing the responses
        # and building the PCs for predicting the real neural responses
        try:
            clip = self._dataset._transformed_x[clip_idx, :, 40:60, 40:60]
            torch.manual_seed(42)  # Make that reproducible
            self._model_raster = ResponseSampler.get_raster(self._model, clip, n_trials=8)
        except:
            pass

    def get_clip_frame(self, t):
        return self._dataset._transformed_x[self._clip_idx, t]

    def get_neuron_raster(self, unit_idx):
        return ResponseSampler.spike_tensor_to_points(self._dataset._transformed_y[self._clip_idx, :, :, unit_idx].T)

    def get_neuron_firing_rate(self, neuron_idx):
        return self._readout.target_y[self._clip_idx, :, neuron_idx]

    def get_model_raster(self, unit_idx):
        return ResponseSampler.spike_tensor_to_points(self._model_raster[:, unit_idx])

    def get_model_pcs(self):
        return self._readout.x[self._clip_idx].cpu()

    def get_model_predicted_firing_rate(self, neuron_idx):
        return self._readout.pred_y[self._clip_idx, :, neuron_idx]

    @staticmethod
    def get_raster(model, clip, n_trials=8):
        warmup_period = 10

        with torch.no_grad():
            spikes = model(clip.unsqueeze(0).unsqueeze(0).cuda().repeat(n_trials, 1, 1, 1, 1), mode="just_spikes")

        return spikes.cpu()[:, :, warmup_period:, 0, 0].cpu()

    @staticmethod
    def spike_tensor_to_points(spike_tensor):
        x = np.array([p[1].item() for p in torch.nonzero(spike_tensor.cpu())])
        y = np.array([p[0].item() for p in torch.nonzero(spike_tensor.cpu())])

        return x, y


class PredictionResults:

    def __init__(self, root, is_train=False):
        self._root = root
        self._is_train = is_train
        self._pred_ms_list = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 136, 144, 152]

        # Results
        self.macaque_movie_df = self._get_results_df("MacaqueMovie", self._pred_ms_list, [(0.19634, 20.0)])
        self.macaque_image_df = self._get_results_df("MacaqueImage", self._pred_ms_list, [(0.19634, 20.0)])
        self.marmoset_movie_df = self._get_results_df("MarmosetMovie", self._pred_ms_list, [(0.4, 80.0)])
        self.salamander_image_df = self._get_results_df("SalamanderImage", self._pred_ms_list, [(0.1171875, 30.0)])
        self.mouse_movie_df = self._get_results_df("MouseMovie", self._pred_ms_list, [(0.4, 80.0)], rf_len_ms=46)

        # Compression of spatial dimension search (supplementary)
        self.supp_macaque_movie_df = self._get_results_df("MacaqueMovie", [0], [(0.122, 20.0), (0.19634, 20.0), (0.2927, 24.0)])
        self.supp_macaque_image_df = self._get_results_df("MacaqueImage", [0], [(0.122, 20.0), (0.19634, 20.0), (0.2927, 24.0)])
        self.supp_marmoset_movie_df = self._get_results_df("MarmosetMovie", [0], [(0.18, 36.0), (0.27, 54.0), (0.4, 80.0)])
        self.supp_salamander_image_df = self._get_results_df("SalamanderImage", [0], [(0.078125, 20.0), (0.1171875, 30.0), (0.1796875, 46.0)])
        self.supp_mouse_movie_df = self._get_results_df("MouseMovie", [0], [(0.18, 36.0), (0.27, 54.0), (0.4, 80.0)], rf_len_ms=46)

    def _get_results_df(self, dataset_name, pred_ms_list, spatial_args, rf_len_ms=21):
        metric_summary = readout_loader.FitMetricsSummary(self._root, dataset_name, is_train=self._is_train, pred_ms_list=pred_ms_list, spatial_args_list=spatial_args, luminance=1, rf_len_ms=rf_len_ms)

        return metric_summary.build_df()

    def get_neuron_cc_for_best_model(self, query_df):
        summary_df = query_df.groupby(["pred_ms", "dim"]).mean()
        pred_ms = summary_df.loc[summary_df.idxmax()].index[0][0]
        return query_df[query_df["pred_ms"] == pred_ms]["cc"].values

    def get_neuron_cc_for_compression_model(self, query_df):
        return query_df[query_df["pred_ms"] == 0]["cc"].values

    def best_offsets(self):
        names = ["macaque_movie", "macaque_image", "marmoset_movie", "salamander_image", "mouse_movie"]
        dataset_dfs = [self.macaque_movie_df, self.macaque_image_df, self.marmoset_movie_df, self.salamander_image_df, self.mouse_movie_df]

        for name, dataset_df in zip(names, dataset_dfs):
            summary_df = dataset_df.groupby(["pred_ms", "dim"]).mean()
            offset = summary_df.loc[summary_df.idxmax()].index[0][0]
            print(f"name={name} offset={offset}")
