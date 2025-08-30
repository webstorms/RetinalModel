import torch
import pandas as pd
import torch.nn.functional as F
from scipy.signal import find_peaks
from brainbox.transforms import GaussianKernel

from retina import train


class OSRQuerySet:

    def __init__(self, root, pred_offset=128, ablate_recurrence=False):
        self.ors_query_8hz = OSRQuery(root, pred_offset=pred_offset, start_ms=150, duration_ms=1300, end_ms=300, hz=8, ablate_recurrence=ablate_recurrence)
        self.ors_query_12hz = OSRQuery(root, pred_offset=pred_offset, start_ms=150, duration_ms=1300, end_ms=300, hz=12, ablate_recurrence=ablate_recurrence)
        self.ors_query_16hz = OSRQuery(root, pred_offset=pred_offset, start_ms=150, duration_ms=1300, end_ms=300, hz=16, ablate_recurrence=ablate_recurrence)
        self.ors_query_20hz = OSRQuery(root, pred_offset=pred_offset, start_ms=150, duration_ms=1300, end_ms=300, hz=20, ablate_recurrence=ablate_recurrence)
        self.ors_idx = self._find_intersection([self.ors_query_12hz.OSR_idx, self.ors_query_16hz.OSR_idx, self.ors_query_20hz.OSR_idx])

    def get_latency_df(self):
        data_list = []
        osr_query_list = [self.ors_query_8hz, self.ors_query_12hz, self.ors_query_16hz, self.ors_query_20hz]

        for osr_query in osr_query_list:
            for i, latency in enumerate(self._get_latencies(osr_query.firing_rate[self.ors_idx])):
                data_list.append({"i": i, "period": int(f"{1000/osr_query.hz:.0f}"), "latency": latency})

        return pd.DataFrame(data_list)

    def get_input_current_latency_df(self):#Nicol
        data_list = []
        osr_query_list = [self.ors_query_8hz, self.ors_query_12hz, self.ors_query_16hz, self.ors_query_20hz]

        for osr_query in osr_query_list:
            for i, latency in enumerate(self._get_latencies(osr_query.smooth_input_current[self.ors_idx])):
                data_list.append({"i": i, "period": int(f"{1000/osr_query.hz:.0f}"), "latency": latency})

        return pd.DataFrame(data_list)

    def _get_latencies(self, firing_rates):
        latency_list = []
        #penult_peak_list = []#Nicol
        #last_peak_list = []#Nicol
        for i in range(firing_rates.shape[0]):
            peaks, _ = find_peaks(firing_rates[i], prominence=0.04)#0.041, was 0.03
            latency_list.append((peaks[-1]-peaks[-2])*4.333)
            #penult_peak_list.append(peaks[-2] * 4.333)#Nicol
            #last_peak_list.append(peaks[-1] * 4.333)#Nicol
        return latency_list#, last_peak_list, penult_peak_list#Nicol

    def _get_peaks_and_latencies(self, firing_rates):
        #latency_list = []
        #penult_peak_list = []#Nicol
        #last_peak_list = []#Nicol
        #for i in range(firing_rates.len):
        peaks, _ = find_peaks(firing_rates, prominence=0.03)#0.041
        try:
            latency = (peaks[-1]-peaks[-2])*4.333
        except:
            latency = 0
        try:
            penultimate_peak = peaks[-2] * 4.333#Nicol
        except:
            penultimate_peak = 0
        try:
            last_peak = peaks[-1] * 4.333#Nicol
        except:
            last_peak = 0
        return latency, last_peak, penultimate_peak, peaks*4.333#Nicol


    def _find_intersection(self, lists):
        intersection_set = set(lists[0])
        for lst in lists[1:]:
            intersection_set.intersection_update(lst)

        return list(intersection_set)


class OSRQuery:

    def __init__(self, root, pred_offset=128, consume_pad=29, start_ms=200, duration_ms=1400, end_ms=200, flash_duration=40, n_trials=8, hz=12, middle_omission=False, ablate_recurrence=False):
        self._model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
        self._consume_pad = consume_pad
        self._start_ms = start_ms
        self._duration_ms = duration_ms
        self._end_ms = end_ms
        self._flash_duration = flash_duration
        self._n_trials = n_trials
        self._middle_omission = middle_omission
        self._ablate_recurrence = ablate_recurrence
        self.hz = hz

        self._flash_clip = self._generate_flash_sequence(hz)
        self._spikes = self._get_raster_responses()
        self._input_current = self._get_input_current()  # Nicol

    @property
    def flash_clip(self):
        return self._flash_clip[0, 0, self._consume_pad:].mean((1, 2)).cpu()

    @property
    def spikes(self):
        return self._spikes[..., 0, 0]

    @property
    def firing_rate(self):
        spike_prob = self._spikes.mean(0)[:, :, 0, 0]   # Mean over trials
        gk = GaussianKernel(1.5, 101)  # 3 bins is approx 12ms#Nicol, was 2

        firing_rate = gk(spike_prob.permute(1, 0).unsqueeze(0))
        firing_rate = firing_rate[0].permute(1, 0)

        return firing_rate

    @property
    def smooth_input_current(self):#Nicol
        mean_input_current = self._input_current.mean(0)[:, :, 0, 0]   # Mean over trials
        gk = GaussianKernel(1.5, 101)  # 3 bins is approx 12ms #Nicol, was 2

        smooth_input_current = gk(mean_input_current.permute(1, 0).unsqueeze(0))
        smooth_input_current = smooth_input_current[0].permute(1, 0)
        #print("smooth input current")
        #print(smooth_input_current.min())
        #print(smooth_input_current.max())
        return smooth_input_current

    @property
    def responsive_idx(self):
        return torch.arange(400)[self.spikes.mean(0).mean(1) > 0.05 * self.spikes.mean(0).mean(1).mean()]

    @property
    def OSR_idx(self):
        return [idx.item() for idx in self.responsive_idx if self._is_OSR(self.firing_rate[idx])]

    def _get_raster_responses(self):
        with torch.no_grad():
            torch.manual_seed(42)
            spikes = self._model(self._flash_clip, "just_spikes", ablate_recurrence=self._ablate_recurrence).cpu()

        return spikes

    def _get_input_current(self):#Nicol
        with torch.no_grad():
            torch.manual_seed(42)
            input_current = self._model(self._flash_clip, "just_input_current", ablate_recurrence=self._ablate_recurrence).cpu()
            #print("input current")
            #print(input_current.min())
            #print(input_current.max())
        return input_current

    def _generate_flash_sequence(self, hz):
        light = 0#Nicol
        darkness = -1.5#Nicol
        dt = 1000 / 240
        n_frames = int(self._duration_ms / dt)
        clip = light*torch.ones(1, 1, n_frames, 20, 20).cuda()#Nicol

        # Add off pulses - 12Hz
        t = 0
        n_flash_frames = int(self._flash_duration / dt)

        while t < n_frames:
            clip[:, :, t: t+n_flash_frames] = darkness  # 40ms, Nicol
            t += int(240 / hz)

        clip = F.pad(clip, (0, 0, 0, 0, self._consume_pad + int(self._start_ms / dt), int(self._end_ms / dt)), value=light)#Nicol
        clip = clip.repeat(self._n_trials, 1, 1, 1, 1)

        if self._middle_omission:
            assert hz == 16
            clip[:, :, 228:241+13] = light#Nicol

        return clip

    def _is_OSR(self, signal):
        dt = 1000 / 240
        start_len = int(self._start_ms / dt)
        duration_len = int(self._duration_ms / dt)

        default_fr = signal[:start_len].max()
        initial_fr = signal[start_len:start_len+duration_len].max()
        post_fr = signal[start_len+duration_len:start_len+duration_len+20].max()

        return ((post_fr / initial_fr) > 1.05) and ((post_fr / default_fr) > 1.2) and (post_fr > 0.2)
