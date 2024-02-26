import torch
import torch.nn.functional as F
from brainbox.datasets.transforms import GaussianKernel

from retina import train


class OSRQuery:

    def __init__(self, root, pred_offset=128, consume_pad=29, start_ms=200, duration_ms=1400, end_ms=200, flash_duration=40, n_trials=8):
        self._model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
        self._consume_pad = consume_pad
        self._start_ms = start_ms
        self._duration_ms = duration_ms
        self._end_ms = end_ms
        self._flash_duration = flash_duration
        self._n_trials = n_trials

        self._flash_clip = self._generate_flash_sequence()
        self._spikes = self._get_raster_responses()

    @property
    def flash_clip(self):
        return self._flash_clip[0, 0, self._consume_pad:].mean((1, 2)).cpu()

    @property
    def spikes(self):
        return self._spikes[..., 0, 0]

    @property
    def firing_rate(self):
        spike_prob = self._spikes.mean(0)[:, :, 0, 0]   # Mean over trials
        gk = GaussianKernel(3, 101)  # 3 bins is approx 12ms

        firing_rate = gk(spike_prob.permute(1, 0).unsqueeze(0))
        firing_rate = firing_rate[0].permute(1, 0)

        return firing_rate

    @property
    def responsive_idx(self):
        return torch.arange(400)[self.spikes.mean(0).mean(1) > 0.05 * self.spikes.mean(0).mean(1).mean()]

    @property
    def OSR_idx(self):
        return [idx for idx in self.responsive_idx if self._is_OSR(self.firing_rate[idx])]

    def _get_raster_responses(self):
        with torch.no_grad():
            torch.manual_seed(42)
            spikes = self._model(self._flash_clip, "just_spikes").cpu()

        return spikes

    def _generate_flash_sequence(self):
        dt = 1000 / 240
        n_frames = int(self._duration_ms / dt)
        clip = torch.ones(1, 1, n_frames, 20, 20).cuda()

        # Add off pulses - 12Hz
        t = 0
        n_flash_frames = int(self._flash_duration / dt)

        while t < n_frames:
            clip[:, :, t: t+n_flash_frames] = 0  # 40ms
            t += 20  # 80ms increments for 12Hz

        clip = F.pad(clip, (0, 0, 0, 0, self._consume_pad + int(self._start_ms / dt), int(self._end_ms / dt)), value=1)
        clip = clip.repeat(self._n_trials, 1, 1, 1, 1)

        return clip

    def _is_OSR(self, signal):
        dt = 1000 / 240
        start_len = int(self._start_ms / dt)
        duration_len = int(self._duration_ms / dt)

        default_fr = signal[:start_len].max()
        initial_fr = signal[start_len:start_len+duration_len].max()
        post_fr = signal[start_len+duration_len:start_len+duration_len+20].max()

        return ((post_fr / initial_fr) > 1.05) and ((post_fr / default_fr) > 1.2) and (post_fr > 0.2)