import gc

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from brainbox import spiking as spiking_phys

from retina import train
import retina.dataset as retina_dataset
from retina.neural.dataset import loader
from retina.analysis import rf


class AllDatasetSpikeStats:

    def __init__(self, root):
        self.sal_img_stats = DatasetSpikeStats(root, istrain=False, dataset_name="SalamanderImage", spatial_scale=0.1171875, max_dim=30.0, get_model_responses=True)
        torch.cuda.empty_cache()
        self.macaque_img_stats = DatasetSpikeStats(root, istrain=False, dataset_name="MacaqueImage", spatial_scale=0.19634, max_dim=20.0, get_model_responses=True)
        torch.cuda.empty_cache()
        self.macaque_movie_stats = DatasetSpikeStats(root, istrain=False, dataset_name="MacaqueMovie", spatial_scale=0.19634, max_dim=20.0, get_model_responses=False)
        self.marmoset_movie_stats = DatasetSpikeStats(root, istrain=False, dataset_name="MarmosetMovie", spatial_scale=0.4, max_dim=80.0, get_model_responses=False)
        self.mouse_movie_stats = DatasetSpikeStats(root, istrain=False, dataset_name="MouseMovie", spatial_scale=0.4, max_dim=80.0, get_model_responses=True)
        torch.cuda.empty_cache()

        self.model_img_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=128, repeats=5, img=True)
        torch.cuda.empty_cache()
        self.model_movie_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=128, repeats=5, img=False)
        torch.cuda.empty_cache()
        self.norec_model_img_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=128, repeats=5, img=True, ablate_recurrence=True)
        torch.cuda.empty_cache()
        self.norec_model_movie_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=128, repeats=5, img=False, ablate_recurrence=True)
        torch.cuda.empty_cache()
        self.compression_model_img_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=0, repeats=5, img=True)
        torch.cuda.empty_cache()
        self.compression_model_movie_stats = ModelDatasetSpikeStats(root, min_cv_spikes=3, pred_offset=0, repeats=5, img=False)
        torch.cuda.empty_cache()

    def get_spike_rate_df(self, img=False):
        return self._build_metric_comparison_df("spike_rate", img)

    def get_cv_df(self, img=False):
        return self._build_metric_comparison_df("cv", img)

    def get_fano_df(self, img=False):
        return self._build_metric_comparison_df("fano", img)

    def get_image_CC(self, pairs=2000, bin_dt=24):
        lag_list, correlation_list, label_list, colour_list = [], [], [], []
        lag, correlation = self.sal_img_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Salamander")
        colour_list.append("gray")
        lag, correlation = self.model_img_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Prediction model")
        colour_list.append("#1f78b4")
        lag, correlation = self.norec_model_img_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Pred. model (no rec.)")
        colour_list.append("deepskyblue")
        lag, correlation = self.compression_model_img_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Encoding model")
        colour_list.append("#ff7f00")

        return lag_list, correlation_list, label_list, colour_list

    def get_movie_CC(self, pairs=2000, bin_dt=24):
        lag_list, correlation_list, label_list, colour_list = [], [], [], []
        lag, correlation = self.mouse_movie_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Mouse")
        colour_list.append("gray")
        lag, correlation = self.model_movie_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Prediction model")
        colour_list.append("#1f78b4")
        lag, correlation = self.norec_model_movie_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Pred. model (no rec.)")
        colour_list.append("deepskyblue")
        lag, correlation = self.compression_model_movie_stats.exp_stats.get_synchronization(pairs=pairs, seed=0, bin_dt=bin_dt)
        lag_list.append(lag)
        correlation_list.append(correlation)
        label_list.append("Encoding model")
        colour_list.append("#ff7f00")

        return lag_list, correlation_list, label_list, colour_list

    def _build_metric_comparison_df(self, metric_func, img):
        list_data = []

        if img:
            sources = [
                ("sal_img_stats", self.sal_img_stats),
                ("macaque_img_stats", self.macaque_img_stats),
                ("model_img_stats", self.model_img_stats),
                ("norec_model_img_stats", self.norec_model_img_stats),
                ("compression_model_img_stats", self.compression_model_img_stats),
            ]
        else:
            sources = [
                ("macaque_movie_stats", self.macaque_movie_stats),
                ("marmoset_movie_stats", self.marmoset_movie_stats),
                ("mouse_movie_stats", self.mouse_movie_stats),
                ("model_movie_stats", self.model_movie_stats),
                ("norec_model_movie_stats", self.norec_model_movie_stats),
                ("compression_model_movie_stats", self.compression_model_movie_stats)
            ]

        for name, dataset in sources:
            exp_data = getattr(dataset.exp_stats, metric_func)

            for v in exp_data:
                list_data.append({"animal": name, "data": "exp", "y": v.item()})

        # Rename for plots
        data_df = pd.DataFrame(list_data)
        rename_mapping = {"sal_img_stats": "Salamander",
                          "macaque_img_stats": "Macaque",
                          "macaque_movie_stats": "Macaque",
                          "marmoset_movie_stats": "Marmoset",
                          "mouse_movie_stats": "Mouse",
                          "model_img_stats": "Prediction model",
                          "model_movie_stats": "Prediction model",
                          "norec_model_img_stats": "Pred. model (no rec.)",
                          "norec_model_movie_stats": "Pred. model (no rec.)",
                          "compression_model_img_stats": "Encoding model",
                          "compression_model_movie_stats": "Encoding model"}
        data_df["animal"] = data_df["animal"].map(rename_mapping)
        data_df["data"] = data_df["data"].map({"exp": "Data"})

        return data_df


class ModelDatasetSpikeStats:

    def __init__(self, root, min_cv_spikes=3, pred_offset=128, repeats=5, img=True, dataset_path="/home/datasets/natural", ablate_recurrence=False, patchify=False):
        if img:
            dataset = loader.load(root, False, "SalamanderImage", 0.1171875, 30.0, 1.0, cell_idx=None)
            self.x = dataset._transformed_x.unsqueeze(1)
            if patchify:
                self.x = self.x[:, :, :, 5:-5, 5:-5]
        else:
            import random
            random.seed(42)
            torch.manual_seed(42)
            test_dataset = retina_dataset.PatchNaturalDataset(root=dataset_path, train=False, temp_len=600, kernel=20, flip=True, n_frame_ext=2)
            self.x = torch.stack([test_dataset[i][0] for i in range(20)])

        prediction_model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
        pred_model_spikes = DatasetSpikeStats.get_raster(prediction_model, self.x, repeats, ablate_recurrence)
        self.pred_model_spikes = pred_model_spikes.permute(0, 3, 1, 2, 4, 5).flatten(3, 5)
        self.exp_stats = SpikeStats(self.pred_model_spikes, min_cv_spikes)

    @property
    def n_unresponsive_units(self):
        activity = self.pred_model_spikes.mean((0, 1, 2))
        return activity[activity == 0].shape[0]

    @property
    def active_unit_idxs(self):
        activity = self.pred_model_spikes.mean((0, 1, 2))
        return torch.arange(400)[activity > 0]


class DatasetSpikeStats:

    def __init__(self, root, istrain, dataset_name, spatial_scale, max_dim, min_cv_spikes=3, pred_offset=128, repeats=5, luminance=1.0, get_model_responses=False):
        dataset = loader.load(root, istrain, dataset_name, spatial_scale, max_dim, luminance=luminance, cell_idx=None)
        self.x = dataset._transformed_x.unsqueeze(1)
        self.spike_tensor = dataset._transformed_y
        del dataset
        gc.collect()
        self.exp_stats = SpikeStats(self.spike_tensor, min_cv_spikes)

        if get_model_responses:
            prediction_model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
            pred_model_spikes = DatasetSpikeStats.get_raster(prediction_model, self.x, repeats)
            del prediction_model
            self.unperm_pred_model_spikes = pred_model_spikes
            self.pred_model_spikes = pred_model_spikes.permute(0, 3, 1, 2, 4, 5).flatten(3, 5)
            offset = self.spike_tensor.shape[1] - self.pred_model_spikes.shape[1]
            self.spike_tensor = self.spike_tensor[:, offset:]
            self.pred_model_stats = SpikeStats(self.pred_model_spikes, min_cv_spikes)
            gc.collect()
            torch.cuda.empty_cache()

    def get_exp_spike_times(self):
        return DatasetSpikeStats.spike_tensor_to_points(self.spike_tensor[:, :, 0].flatten(0, 1).T)

    def get_pred_spike_times(self, idxs, n, seed=42):
        torch.manual_seed(seed)
        spikes = self.unperm_pred_model_spikes[:, 0, idxs]
        spikes = spikes.permute(0, 2, 1, 3, 4)
        spikes = spikes.flatten(0, 1)
        spikes = spikes.flatten(1, 3)

        total_neurons = spikes.shape[1]
        idx = torch.randperm(total_neurons)[:n]
        spikes = spikes[:, idx]

        return DatasetSpikeStats.spike_tensor_to_points(spikes.T)

    @staticmethod
    def get_raster(model, clips, n_trials=8, ablate_recurrence=False):
        unit_raster_list = []

        for i in range(clips.shape[0]):
            trial_unit_raster_list = []
            for _ in range(n_trials):
                with torch.no_grad():
                    spikes = model(clips[i:i+1].cuda(), mode="just_spikes", stride=4, ablate_recurrence=ablate_recurrence)
                    trial_unit_raster_list.append(spikes.detach().cpu())
                    del spikes

            unit_raster_list.append(torch.cat(trial_unit_raster_list))

        torch.cuda.empty_cache()
        return torch.stack(unit_raster_list)

    @staticmethod
    def spike_tensor_to_points(spike_tensor):
        x = np.array([p[1].item() for p in torch.nonzero(spike_tensor.cpu())])
        y = np.array([p[0].item() for p in torch.nonzero(spike_tensor.cpu())])

        return x, y


class SpikeStats:

    def __init__(self, spike_tensor, min_cv_spikes=3):
        self.spike_tensor = spike_tensor
        self.duration = (self.spike_tensor.shape[1] / 240) * 1000
        self.min_cv_spikes = min_cv_spikes

    @property
    def spike_count(self):
        return self.spike_tensor.sum(1).mean(1).mean(0)

    @property
    def spike_variance(self):
        return self.spike_tensor.sum(1).var(1).mean(0)

    @property
    def spike_rate(self):
        return 1000 * (self.spike_count / self.duration)

    @property
    def cv(self):
        reshaped_spikes = self.spike_tensor[:, :, 0, :].permute(0, 2, 1)
        isi_tensor = spiking_phys.compute_isis_tensor(reshaped_spikes)

        # Returns clips x neurons
        cvs = spiking_phys.compute_isi_cvs(isi_tensor, self.min_cv_spikes)
        # -1 are invalid cv from min_cv_spikes filtering
        return torch.Tensor([neuron_cvs[neuron_cvs != -1].mean() for neuron_cvs in cvs.permute(1, 0)])

    @property
    def synchronization(self):
        return self.get_synchronization(pairs=1000, seed=0, bin_dt=33)[1]

    def get_synchronization(self, pairs=1000, seed=0, bin_dt=33, from_idxs=None, to_idxs=None):
        torch.manual_seed(seed)
        dt = 4
        reshaped_spikes = self.spike_tensor[:, :, 0, :].permute(0, 2, 1)
        cross_covariance_tensor = spiking_phys.compute_synchronization(reshaped_spikes, pairs, dt=dt, bin_dt=bin_dt, normalize=False, from_idxs=from_idxs, to_idxs=to_idxs)
        synchronization_df = spiking_phys.compute_synchronization_df(cross_covariance_tensor.mean(0), dt=bin_dt)

        mean_synchronization = synchronization_df.groupby("lag").mean()
        lag = mean_synchronization.index / 1000
        # brainbox >=0.0.9 normalizes the binning kernel in bin_spikes (kernel /= kernel.sum()),
        # which scales the unnormalized cross-covariance by 1/(bin_dt/dt)**2. Compensate to
        # reproduce the paper's published values, which were generated against the older binning.
        kernel_length = bin_dt / dt
        correlation = mean_synchronization["correlation"].values * (kernel_length ** 2)

        return lag.values, correlation

    @property
    def fano(self):
        fano_factors = self.spike_variance / self.spike_count

        return fano_factors


class UnitResponseCorrelation:

    def __init__(self, root, model_states):
        self.model_states = model_states
        model = train.Trainer.load_model(f"{root}/results", "0.0031622776601683794_*_0.01_0.6_128_8")
        self.rf_query = rf.RFQuery(root, model)
        self.cc_matrix = self._build_correlation_matrix()

    @property
    def unit_cc_df(self):
        data = []
        for i in range(4):
            data.append({"type": "Same", "v": self.cc_matrix[i, i].item()})
        for i in range(4):
            for j in range(4):
                if i != j:
                    data.append({"type": "Different", "v": self.cc_matrix[i, j].item()})

        return pd.DataFrame(data)

    def _build_correlation_matrix(self):
        cc_matrix = torch.zeros(4, 4)
        torch.manual_seed(1)
        for i in range(4):
            for j in range(4):
                from_idxs = self._get_unit_type_idxs(i)
                to_idxs = self._get_unit_type_idxs(j)
                cc = self.model_states.exp_stats.get_synchronization(pairs=1000, seed=0, bin_dt=33, from_idxs=from_idxs, to_idxs=to_idxs)
                cc_matrix[i, j] = cc[1][int(len(cc[1])//2)]
        return cc_matrix

    def _get_unit_type_idxs(self, unit_type):
        query = self.rf_query.params_df["type"] == unit_type
        return torch.Tensor(self.rf_query.params_df[query].og_index.values)
