import os

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import brainbox.physiology.rfs as rfs
from brainbox.physiology.rfs.gaussian import fit, query


class RFQuery:

    PARASOL_Y = 3
    PARASOL_X = 0.5
    MIDGET_Y = 1.5
    MIDGET_X = 0.1

    def __init__(self, root, model, min_cc=0.7, min_env=0.5):
        self.og_strfs = self._build_strfs(model)
        self.rfs = self._get_all_highest_power_spatial_rf(self.og_strfs)

        if not os.path.exists(f"{root}/data/rf/gaus.csv"):
            fit.GaussianFitter().fit_spatial(f"{root}/data/rf/gaus.csv", self.rfs, 200, n_spatial_iterations=4000, spatial_lr=1e-1)

        self.fit_query = query.GaussianQuery(f"{root}/data/rf/gaus.csv", self.og_strfs)
        self.params_df, self.spatial_rfs, self.gaussians, self.strfs = self.fit_query.validate(min_cc=min_cc, min_env=min_env)
        self.params_df = self.params_df.copy()
        self.params_df["size"] = (self.params_df["sigmax"] * self.params_df["sigmay"]) ** 0.5
        self.params_df["index"] = list(range(len(self.params_df)))
        self.params_df = self.params_df.reset_index()
        self.params_df = self.params_df.set_index("index")
        self.params_df["first_pc"], self.params_df["type"] = self._get_cell_type()
        self.params_df = self.params_df.rename(columns={"level_0": "og_index"})

    def _build_strfs(self, model, rf_len=36, t_len=100, noise_var=10, samples=200, batch_size=50, rf_h=20, rf_w=20, device="cuda", **kwargs):

        def model_output(noise):
            with torch.no_grad():
                return model(noise.unsqueeze(1), mode="val", **kwargs)[1]

        return rfs.sta(model_output, rf_len, rf_h, rf_w, t_len, noise_var, samples, batch_size, device)

    def _get_highest_power_spatial_rf(self, spatiotemporal_rf):
        # spatiotemporal_rf: rf_len, rf_shape, rf_shape
        power_at_timesteps = torch.pow(spatiotemporal_rf, 2).mean(dim=(1, 2))
        t = power_at_timesteps.argmax().item()
        spatial_rf = spatiotemporal_rf[t]

        return spatial_rf

    def _get_all_highest_power_spatial_rf(self, spatiotemporal_rfs):
        # spatiotemporal_rfs: n_units, rf_len, rf_shape, rf_shape
        rfs = []

        for i in range(len(spatiotemporal_rfs)):
            spatial_rf = self._get_highest_power_spatial_rf(spatiotemporal_rfs[i].detach().cpu().float())
            rfs.append(spatial_rf)
        rfs = torch.stack(rfs)

        return rfs

    def _get_cell_type(self):
        # Get first PC from rf temporal profile
        temp_profiles = self.strfs.mean((2, 3))
        first_pc = PCA(n_components=1).fit_transform(temp_profiles)[:, 0]

        # K-mean clustering
        data = [(x, y) for x, y in zip(first_pc, self.params_df["size"])]
        kmeans = KMeans(n_clusters=4, n_init=1, max_iter=1, random_state=42)
        kmeans.fit(data)
        init_centres = np.array([
            [-RFQuery.PARASOL_X, RFQuery.PARASOL_Y],
            [RFQuery.PARASOL_X, RFQuery.PARASOL_Y],
            [-RFQuery.MIDGET_X, RFQuery.MIDGET_Y],
            [RFQuery.MIDGET_X, RFQuery.MIDGET_Y]]).astype(np.float64)
        kmeans.cluster_centers_ = init_centres  # Hard code those centroids
        cell_type = kmeans.predict(data)

        return first_pc, cell_type


class FlashQuery:

    WARMUP = 10

    def __init__(self, model, pre_ms=100, light_ms=400, off_ms=400, lum=0.3, model_n_frames=29):
        self.model_n_frames = model_n_frames
        dt = 1000/240
        self._flash_clip = self._generate_flash_sequence(pre_ms, light_ms, off_ms, dt=dt, frame_size=20, lum=lum, begin_pad=model_n_frames + FlashQuery.WARMUP)
        self._raster = self._get_raster(model, self._flash_clip)

    @property
    def flash_clip(self):
        return self._flash_clip[FlashQuery.WARMUP + self.model_n_frames:]

    def get_spike_points(self, unit_idx):
        unit_idx = int(unit_idx)
        x = np.array([p[1].item() for p in torch.nonzero(self._raster[:, unit_idx])])
        y = np.array([p[0].item() for p in torch.nonzero(self._raster[:, unit_idx])])

        return x, y

    def _get_raster(self, model, clip, n_trials=8):
        torch.manual_seed(42)

        with torch.no_grad():
            spikes = model(clip.unsqueeze(0).unsqueeze(0).repeat(n_trials, 1, 1, 1, 1).cuda(), mode="just_spikes")

        return spikes[:, :, FlashQuery.WARMUP:, 0, 0].cpu()

    def _generate_flash_sequence(self, pre_ms, light_ms, off_ms, dt=4.16, frame_size=20, lum=1, begin_pad=29):
        duration_ms = pre_ms + light_ms + off_ms

        n_frame = int(duration_ms / dt)
        flash_clip = -lum * torch.ones(n_frame, frame_size, frame_size)

        flash_clip[int(pre_ms/dt):int((pre_ms + light_ms)/dt)] = lum

        flash_clip = F.pad(flash_clip, (0, 0, 0, 0, begin_pad, 0), value=lum)

        return flash_clip