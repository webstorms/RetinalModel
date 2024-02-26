import torch
import torch.nn.functional as F
import numpy as np

from retina import train


class PrecisionQuery:

    def __init__(self, root, pred_offset=128, consume_pad=29, duration_ms=1400, n_trials=8, lum=1):
        self._model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
        self._consume_pad = consume_pad
        self._duration_ms = duration_ms
        self._n_trials = n_trials

        self._flicker_clip = self._generate_flicker_clip().cuda() * lum
        self._spikes = self._get_raster_responses()

    @property
    def flicker_clip(self):
        return self._flicker_clip[0, 0, self._consume_pad:].mean((1, 2)).cpu()

    @property
    def spikes(self):
        return self._spikes[..., 0, 0]

    def get_spike_coo(self, unit_idx):
        spike_tensor = self.spikes[:, unit_idx]
        x = np.array([p[1].item() for p in torch.nonzero(spike_tensor)])
        y = np.array([p[0].item() for p in torch.nonzero(spike_tensor)])

        return x, y

    def _get_raster_responses(self):
        with torch.no_grad():
            spikes = self._model(self._flicker_clip, "just_spikes").cpu()

        return spikes

    def _generate_flicker_clip(self):
        dt = 1000 / 240
        n_frames = int(self._duration_ms / dt)
        clip = torch.ones(1, 1, n_frames, 20, 20)
        torch.manual_seed(42)
        intensity_values = torch.normal(torch.zeros(n_frames), torch.ones(n_frames))
        # Repeat same intensity for approx 30ms, as done in Berry et al., 1997
        for i in range(int(len(intensity_values) / 4)):
            for j in range(3):
                intensity_values[i*4 + j + 1] = intensity_values[i*4]
        intensity_values = intensity_values.view(n_frames, 1, 1)
        clip = clip * intensity_values
        clip = clip.repeat(self._n_trials, 1, 1, 1, 1)
        clip = F.pad(clip, (0, 0, 0, 0, self._consume_pad, 0), value=0)

        return clip
