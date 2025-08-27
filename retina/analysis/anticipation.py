import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np


def generate_bar_frame(bar_w, x, kernel, min_l=-1.5, max_l=0):
    kernel = max_l * torch.ones(kernel, kernel)
    s = max(0, x)
    e = max(0, x+bar_w)
    #kernel[2:-2, s:e] = min_l
    kernel[:, s:e] = min_l
    return kernel

def generate_smooth_bar_frame(bar_w, x, kernel, min_l=-1.5, max_l=0):
    kernel = max_l * torch.ones(kernel, kernel)
    s = int(max(0, np.floor(x)+1))
    e = int(max(0, np.floor(x+bar_w)))
    kernel[:, s:e] = min_l
    if s>0 and e<20:
        kernel[:, s-1] = max_l+(1-np.remainder(x,1))*min_l
        kernel[:, e] = max_l+np.remainder(x,1)*min_l
    return kernel

def generate_flash_bar_sequence(begin_pad, end_pad, n_frames, x=17, bar_w=5, kernel=20, min_l=-1.5, max_l=0):
    frames_list = [generate_bar_frame(bar_w, x, kernel, min_l, max_l) for _ in range(n_frames)]
    return gen_clip(begin_pad, end_pad, frames_list, max_l)


def generate_moving_bar_sequence(begin_pad, end_pad, start_x, end_x, dx=1, bar_w=5, kernel=20, min_l=-1.5, max_l=0, smooth=False):
    frames_list = []

    dpx = 1 if end_x > start_x else -1
    for x in range(start_x, end_x, dpx):
        if smooth:
            for xstep in range(dx):
                frames_list.append(generate_smooth_bar_frame(bar_w, x+xstep/dx, kernel, min_l, max_l))
        else:
            for _ in range(dx):
                frames_list.append(generate_bar_frame(bar_w, x, kernel, min_l, max_l))

    return gen_clip(begin_pad, end_pad, frames_list, max_l)


def gen_clip(begin_pad, end_pad, frames_list, lum):
    frames_tensor = torch.stack(frames_list)
    frames_tensor = F.pad(frames_tensor, (0, 0, 0, 0, begin_pad, end_pad), value=lum)
    t, h, w = frames_tensor.shape
    frames_tensor = frames_tensor.view(1, 1, t, h, w)

    return frames_tensor


class AnticipationAnalysis:

    def __init__(self, model, bar_width, bar_x0, luminance=0, n_repeats=20, ablate_recurrence=False, bar_speed=37, smooth=False):
        # bar_x0 is left edge starting position

        self.consume_length = 29  # number of frames consumed by model to produce single output frame
        self.pad = 100  # pad extra frames to beginning of clips
        begin_pad = self.consume_length + self.pad
        right_moving_start_x = -bar_width + 1  # starting x value for left bar edge
        left_moving_start_x = 20  # starting x value for left bar edge
        max_l = luminance
        min_l = -1.5

        dx = bar_speed  # 8*4  # flash of bar in movie consists of dx frames (check maths in method section of the paper)
        self._flash_bar_clip = generate_flash_bar_sequence(begin_pad, end_pad=begin_pad, n_frames=4, x=bar_x0, bar_w=bar_width,
                                                           min_l=min_l, max_l=max_l).float()
        self._right_moving_bar_clip = generate_moving_bar_sequence(begin_pad, end_pad=begin_pad, start_x=right_moving_start_x,
                                                                   end_x=left_moving_start_x + 1, dx=dx, bar_w=bar_width,
                                                                   min_l=min_l, max_l=max_l, smooth=smooth).float()
        self._left_moving_bar_clip = torch.flip(self._right_moving_bar_clip,dims=[4])
        self._left_flash_bar_clip = torch.flip(self._flash_bar_clip,dims=[4])

        # Check that the moving and flash bar align correctly in time
        # (Another check is animating frame by frame as seeing that the clips evolve as expected)
        self._right_moving_offset = dx * (bar_x0 - right_moving_start_x)  # number of frames offset between moving and flash clip
        self._left_moving_offset = self._right_moving_offset
        assert torch.equal(self._flash_bar_clip[0, 0, begin_pad], self._right_moving_bar_clip[0, 0, begin_pad + self._right_moving_offset])  # should match
        assert not torch.equal(self._flash_bar_clip[0, 0, begin_pad - 1], self._right_moving_bar_clip[0, 0, begin_pad + self._right_moving_offset])  # should not match

        self._flash_bar_firing = self._get_firing_rate(model, self._flash_bar_clip.cuda(), n_repeats, ablate_recurrence=ablate_recurrence)
        self._left_flash_bar_firing = self._get_firing_rate(model, self._left_flash_bar_clip.cuda(), n_repeats, ablate_recurrence=ablate_recurrence)
        self._right_moving_bar_firing = self._get_firing_rate(model, self._right_moving_bar_clip.cuda(), n_repeats, ablate_recurrence=ablate_recurrence)
        self._left_moving_bar_firing = self._get_firing_rate(model, self._left_moving_bar_clip.cuda(), n_repeats, ablate_recurrence=ablate_recurrence)

    @property
    def mean_flash_bar_firing(self):
        return self._flash_bar_firing.mean(0)

    @property
    def mean_left_flash_bar_firing(self):
        return self._left_flash_bar_firing.mean(0)

    @property
    def mean_right_moving_bar_firing(self):
        return self._right_moving_bar_firing.mean(0)[:, self._right_moving_offset:]

    @property
    def mean_left_moving_bar_firing(self):
        return self._left_moving_bar_firing.mean(0)[:, self._left_moving_offset:]

    @property
    def flash_clip(self):
        return self._flash_bar_clip[0, 0, self.consume_length:]

    @property
    def left_flash_clip(self):
        return self._left_flash_bar_clip[0, 0, self.consume_length:]

    @property
    def right_moving_clip(self):
        return self._right_moving_bar_clip[0, 0, self.consume_length + self._right_moving_offset:]

    @property
    def left_moving_clip(self):
        return self._left_moving_bar_clip[0, 0, self.consume_length + self._left_moving_offset:]

    def flash_bar_spikes_df(self, unit_idx, rel_start=-400, rel_end=400):
        return self._build_spike_times_df(self._flash_bar_firing[:, unit_idx], rel_start=rel_start, rel_end=rel_end, dt_step=1)

    def left_flash_bar_spikes_df(self, unit_idx, rel_start=-400, rel_end=400):
        return self._build_spike_times_df(self._left_flash_bar_firing[:, unit_idx], rel_start=rel_start, rel_end=rel_end, dt_step=1)

    def right_moving_bar_spikes_df(self, unit_idx, rel_start=-400, rel_end=400):
        return self._build_spike_times_df(self._right_moving_bar_firing[:, unit_idx, self._right_moving_offset:], rel_start=rel_start, rel_end=rel_end)

    def left_moving_bar_spikes_df(self, unit_idx, rel_start=-400, rel_end=400):
        return self._build_spike_times_df(self._left_moving_bar_firing[:, unit_idx, self._left_moving_offset:], rel_start=rel_start, rel_end=rel_end)

    def _get_firing_rate(self, model, clip, n_repeats=40, ablate_recurrence=False):
        with torch.no_grad():
            spikes_list = []
            for _ in range(n_repeats):
                out = model(clip, "just_spikes", ablate_recurrence=ablate_recurrence)
                spikes = out.cpu().detach()
                spikes_list.append(spikes.cpu())

        spike_tensor = torch.stack(spikes_list)[:, 0, :, :, 0, 0]

        return spike_tensor

    def _build_spike_times_df(self, firing, rel_start=-400, rel_end=400, dt_step=10):
        dt = 1000 / 240

        data = []
        for trial in range(20):
            for t in range(rel_start, rel_end, dt_step):
                t_idx = int(t/dt) + self.pad - 1
                if t_idx < 0 or t_idx >= firing.shape[1]:
                    # Out of bounds
                    data.append({"trial": trial, "t": t, "r": 0})
                else:
                    data.append({"trial": trial, "t": t, "r": firing[trial, t_idx].item()})

        return pd.DataFrame(data)