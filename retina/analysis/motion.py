import os

import torch
import numpy as np
import pandas as pd

from brainbox import tuning, spiking
from brainbox.transforms import GaussianKernel


class GratingQuery:

    def __init__(self, root, model, ablate_recurrence=False):#Nicol
        self._root = root
        self._model = model
        self._ablate_recurrence = ablate_recurrence#Nicol

        if not os.path.exists(f"{self.tuning_path}/probe.csv"):
            self.probe(ablate_recurrence=ablate_recurrence)

        self.query = tuning.TuningQuery(self.tuning_path)
        self.all_tuning_results = self.query.validate(response_threshold=0.0, fit_threshold=-1)
        self.filtered_tuning_results = self.query.validate(response_threshold=0.01, fit_threshold=-1)

    @property
    def tuning_path(self):
        return f"{self._root}/data/tuning"

    def probe(self, n_trials=8, ablate_recurrence=False):
        torch.manual_seed(42)

        probe_ms = 3000  # Probe for 3s
        dt = 1000 / 240
        warmup_period = 10  # Same as used for training

        def input_to_spikes(data):
            with torch.no_grad():
                rate_trains_list = []
                data = data.unsqueeze(1).cuda()
                data = (data - data.mean()) / data.std()

                b, _, _, _, _ = data.shape
                data = data.repeat(n_trials, 1, 1, 1, 1)
                spike_trains = self._model(data, mode="just_spikes", ablate_recurrence=self._ablate_recurrence)[..., 0, 0]#Nicol
                _, n, t = spike_trains.shape
                spike_trains = spike_trains.view(n_trials, b, n, t)
                spike_trains = spike_trains.mean(0)

                rate_trains = spiking.rate.bin_spikes(spike_trains, dt, 400, pad_input=True, gaussian=True, sigma=9)  # Same as neural prediciton - 9 bins is approx 37ms
                rate_trains_list.append(rate_trains[:, :, warmup_period:])

                mean_rate_trains = torch.stack(rate_trains_list).mean(dim=0)
                return mean_rate_trains

        thetas = np.linspace(0, np.pi*2, 72)
        spatial_freqs = np.around(np.linspace(0.01, 0.2, 10), 4)
        temporal_freqs = [1, 2, 4, 8]

        gratings = tuning.GratingsProber(input_to_spikes, amplitude=1, rf_w=20, rf_h=20, duration=probe_ms+warmup_period*dt, dt=dt, thetas=thetas, spatial_freqs=spatial_freqs, temporal_freqs=temporal_freqs)
        gratings.probe_and_fit(self.tuning_path, probe_batch=256, response_batch=32)


class TextureMotion:

    def __init__(self, model, grating_query, unit_idx, sf=None, tf=None, ablate_recurrence=False):
        self._grating_query = grating_query
        self._unit_idx = unit_idx

        self.optimal_grating = self._generate_grating(sf=sf, tf=tf)
        self.fixed_grating = self._generate_grating(sf=sf, tf=0)

        self.optimal_raster_v = TextureMotion.get_raster(model, unit_idx, self.optimal_grating, ablate_recurrence=ablate_recurrence)
        self.fixed_raster_v = TextureMotion.get_raster(model, unit_idx, self.fixed_grating, ablate_recurrence=ablate_recurrence)

        self.optimal_raster_x, self.optimal_raster_y = TextureMotion.spike_tensor_to_points(self.optimal_raster_v)
        self.fixed_raster_x, self.fixed_raster_y = TextureMotion.spike_tensor_to_points(self.fixed_raster_v)

    @property
    def optimal_raster(self):
        return self.optimal_raster_x, self.optimal_raster_y

    @property
    def fixed_raster(self):
        return self.fixed_raster_x, self.fixed_raster_y

    @property
    def orientation_tf_tuning(self):
        theta, tf, response = self._grating_query.query.orientation_tf_tuning_curve(self._unit_idx)

        # Not a good coding-practice, but hard coded 4, as 4 different tf values were probed
        return theta.reshape(-1, 4), tf.reshape(-1, 4), response.reshape(-1, 4)

    @property
    def orientation_sf_tuning(self):
        theta, sf, response = self._grating_query.query.orientation_sf_tuning_curve(self._unit_idx)

        # Not a good coding-practice, but hard coded 10, as 10 different sf values were probed
        return theta.reshape(-1, 10), sf.reshape(-1, 10), response.reshape(-1, 10)

    def _generate_grating(self, sf=None, tf=None):
        theta = self._grating_query.query.preferred_direction(self._unit_idx)

        if sf is None:
            sf = self._grating_query.query.preferred_spatial_frequency(self._unit_idx)

        if tf is None:
            tf = self._grating_query.query.preferred_temporal_frequency(self._unit_idx)

        probe_ms = 3000
        dt = 1000 / 240
        warmup_period = 10

        return tuning.GratingsProber.generate_grating(1, 20, 20, theta, sf, tf, duration=probe_ms + warmup_period * dt, dt=dt)

    @staticmethod
    def get_raster(model, unit_idx, grating, n_trials=8,ablate_recurrence=False):
        #print(ablate_recurrence)
        warmup_period = 10

        with torch.no_grad():
            spikes = model(grating.unsqueeze(0).unsqueeze(0).repeat(n_trials, 1, 1, 1, 1).cuda(), mode="just_spikes",ablate_recurrence=ablate_recurrence)

        return spikes[:, unit_idx, warmup_period:, 0, 0].cpu()

    @staticmethod
    def spike_tensor_to_points(spike_tensor):
        x = np.array([p[1].item() for p in torch.nonzero(spike_tensor.cpu())])
        y = np.array([p[0].item() for p in torch.nonzero(spike_tensor.cpu())])

        return x, y

    @staticmethod
    def count_texture_tuned_units(model, grating_query, ablate_recurrence=False):

        def smooth(signal):
            gk = GaussianKernel(3, 101)  # 3 bins is approx 12ms
            firing_rate = gk(signal.unsqueeze(0).unsqueeze(-1))
            return firing_rate[0, :, 0]

        count_data = []
        for i in range(400):
            texture_motion = TextureMotion(model, grating_query, i, sf=0.3, ablate_recurrence=ablate_recurrence)
            opt_count = texture_motion.optimal_raster_v.mean(0)
            fixed_count = texture_motion.fixed_raster_v.mean(0)

            opt_count = smooth(opt_count).max().item()
            fixed_count = smooth(fixed_count).max().item()

            count_data.append({"opt_count": opt_count, "fixed_count": fixed_count})

        df = pd.DataFrame(count_data)
        q = (df["opt_count"] > df["fixed_count"]) & (df["fixed_count"] < 0.1) & (df["opt_count"] > 0.1)

        return df[q].shape[0]


class DifferentialMotion:

    def __init__(self, model, unit_idx, theta, spatial_freq, temporal_freq, y0, x0, r, lum=1, moving_background=False, ablate_recurrence=False):
        #print("making grating")
        self.grating = self._jittered_grating(theta, spatial_freq, temporal_freq, lum)
        #self.grating = self.grating * lum
        #self.grating_shift = self.grating_shift * lum
        #print("making grating 2")
        self.grating2 = self._jittered_grating(theta, spatial_freq, temporal_freq, lum) #was spatial_freq*1.5 and 1.*temporal_freq
        #self.grating2 = self.grating2 * lum
        self.masked_grating_local = self._mask_grating(y0, x0, r, moving_background=True)
        self.masked_grating_global = self._mask_grating(y0, x0, r, moving_background=False)
        self.global_raster_x, self.global_raster_y = TextureMotion.spike_tensor_to_points(TextureMotion.get_raster(model, unit_idx, self.masked_grating_global, ablate_recurrence=ablate_recurrence))#eye
        self.local_raster_x, self.local_raster_y = TextureMotion.spike_tensor_to_points(TextureMotion.get_raster(model, unit_idx, self.masked_grating_local, ablate_recurrence=ablate_recurrence))#eye+object

        self.global_fwd_current, self.global_rec_current = self._get_current(model, unit_idx, self.masked_grating_global, n_trials=8, ablate_recurrence=ablate_recurrence)
        self.local_fwd_current, self.local_rec_current = self._get_current(model, unit_idx, self.masked_grating_local, n_trials=8, ablate_recurrence=ablate_recurrence)

    @property
    def global_raster(self):
        return self.global_raster_x, self.global_raster_y

    @property
    def local_raster(self):
        return self.local_raster_x, self.local_raster_y

    def _generate_grating(self, theta, spatial_freq, temporal_freq):
        probe_ms = 3000
        dt = 1000 / 240
        warmup_period = 10

        return tuning.GratingsProber.generate_grating(1, 20, 20, theta, spatial_freq, temporal_freq, duration=probe_ms+warmup_period*dt, dt=dt)

    def _jittered_grating(self,theta, spatial_freq, temporal_freq, lum):
        amplitude = lum
        rf_w = 20
        rf_h = 20
        probe_ms = 50000
        dt = 1000 / 240
        warmup_period = 10
        duration = probe_ms + warmup_period * dt

        y, x = torch.meshgrid(
            [
                torch.arange(rf_h, dtype=torch.float32),
                torch.arange(rf_w, dtype=torch.float32),
            ],
            indexing="ij",
        )
        theta = torch.Tensor([theta]).expand_as(y)
        spatial_freq = torch.Tensor([spatial_freq]).expand_as(y)

        fps = int(1000 / dt)
        n_timesteps = int(duration / dt)

        jitter = 2 * torch.randint(0, 2, (n_timesteps,)) - 1
        #jitter[1::4] = 0
        #jitter[2::4] = 0
        #jitter[3::4] = 0
        #print(jitter)
        randomwalk = torch.cumsum(jitter,dim=0)
        #print(randomwalk.shape)
        randomwalkgrid = (
            randomwalk.view(n_timesteps, 1, 1).repeat(1, rf_h, rf_w)
        )


        rotx = x * torch.cos(theta) - y * torch.sin(theta)
        initial_phase = 2 * np.pi * torch.rand(1)
        #initial_phase2 = 2 * np.pi * torch.rand(1)
        grating = amplitude * torch.sin(
            2 * np.pi * (spatial_freq * rotx - temporal_freq * randomwalkgrid / fps) + initial_phase
        )
        #grating_shift = amplitude * torch.sin(
        #    2 * np.pi * (spatial_freq * rotx - temporal_freq * randomwalkgrid / fps) + initial_phase2
        #)

        return grating#, grating_shift


    def _mask_grating(self, x0, y0, r, moving_background):
        mask1 = torch.zeros_like(self.grating)
        mask2 = torch.zeros_like(self.grating)

        for i in range(20):
            for j in range(20):
                d = np.sqrt((x0 - i) ** 2 + (y0 - j) ** 2)
                if d <= r:
                    mask1[:, i, j] = 1

        for i in range(20):
            for j in range(20):
                d = np.sqrt((x0 - i) ** 2 + (y0 - j) ** 2)
                if d <= r+1.1:
                    mask2[:, i, j] = 1

        if not moving_background:
            return mask1 * self.grating + (1-mask2) * self.grating#, Nicol
        else:
            return mask1 * self.grating + (1-mask2) * self.grating2

    def _get_current(self, model, unit_idx, grating, n_trials=8, ablate_recurrence=False):
        warmup_period = 10

        with torch.no_grad():
            output, spikes, mem, abs_rec, input_current = model(grating.unsqueeze(0).unsqueeze(0).repeat(n_trials, 1, 1, 1, 1).cuda(), mode="val",ablate_recurrence=ablate_recurrence)
            abs_rec = abs_rec[:, unit_idx, warmup_period:].cpu()
            input_current = input_current[:, unit_idx, warmup_period:, 0, 0].cpu()

        return abs_rec, input_current


def load_anticipation_df(root, file_name):

    def load_anticipation_csv(file_name):
        data_df = pd.read_csv(f"{root}/data/figures/{file_name}", header=None)
        data_df = data_df.rename(columns={0: "x", 1: "y", 2: "index", 3: "type"})
        data_df = data_df.set_index(["index", "type"])
        data_df = data_df.unstack(level='type')
        data_df.columns = ['_'.join([c.strip() for c in col]).strip() for col in data_df.columns.values]
        data_df["se"] = abs(data_df["y_mean"] - data_df["y_std"])
        data_df = data_df.drop(columns=["x_std", "y_std"])

        return data_df

    data_df = load_anticipation_csv(file_name)

    data_list = []
    for x, y, err in zip(data_df["x_mean"], data_df["y_mean"], data_df["se"]):
        for _ in range(10):
            e = (np.random.rand(1)[0]-0.5) * 10 * err
            data_list.append({"t": x, "r": y + e})

    data_list.append({"t": -400, "r": 0})
    data_list.append({"t": 400, "r": 0})

    return pd.DataFrame(data_list)
