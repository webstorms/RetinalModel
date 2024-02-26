import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode


class YTransformer:

    def __init__(self, y_hz):
        self._y_hz = y_hz

    def __call__(self, data):
        return self._rescale_temporally(data, NeuralDataset.MODEL_HZ, self._y_hz, tile=False)

    def _rescale_temporally(self, data, model_hz, data_hz, tile=False):
        # tile=True is used for movie frames
        # tile=False is used for spike trains
        rescale_factor = model_hz / data_hz
        if rescale_factor == 1:
            return data

        target_len = round(rescale_factor * data.shape[1])
        n_stim, _, a, b = data.shape
        rescaled_data = torch.zeros(n_stim, target_len, a, b)
        data_dt = 1000 / data_hz
        model_dt = 1000 / model_hz

        if model_hz > data_hz:
            last_frame_idx = None

            for t in range(rescaled_data.shape[1]):
                frame_idx = int((t * model_dt) / data_dt)

                if not tile and last_frame_idx == frame_idx:
                    continue

                frame = data[:, frame_idx]
                rescaled_data[:, t] += frame
                last_frame_idx = frame_idx

        elif model_hz < data_hz:
            for t in range(data.shape[1]):
                frame = data[:, t]
                if not tile:
                    rescaled_data[:, int((t * data_dt) / model_dt)] += frame
                else:
                    rescaled_data[:, int((t * data_dt) / model_dt)] = frame

        return rescaled_data


class XTransformer(YTransformer):

    def __init__(self, spatial_scale, max_dim, x_hz, mean, std, luminance, pad):
        self._spatial_scale = spatial_scale
        self._max_dim = max_dim
        self._x_hz = x_hz
        self._mean = mean
        self._std = std
        self._luminance = luminance
        self._pad = pad

    def __call__(self, data):
        data = self._rescale_spatial_dims(data, self._spatial_scale)
        data = self._limit_spatial_dims(data, self._max_dim)
        data = self._rescale_temporally(data, NeuralDataset.MODEL_HZ, self._x_hz, tile=True)
        data = self._normalize(data, self._mean, self._std)
        data = self._luminance * data
        data = F.pad(data, (self._pad, self._pad, self._pad, self._pad))

        return data

    def _normalize(self, data, mean, std):
        return (data - mean) / std

    def _rescale_spatial_dims(self, data, spatial_scale):
        if spatial_scale == 1:
            return data
        h, w = data.shape[2], data.shape[3]
        scaled_h, scaled_w = int(spatial_scale * h), int(spatial_scale * w)
        resize_transform = Resize((scaled_h, scaled_w), interpolation=InterpolationMode.BICUBIC, antialias=None)

        return resize_transform(data)

    def _limit_spatial_dims(self, data, max_dim):
        spatial_diff = max_dim - data.shape[-1]
        assert spatial_diff % 2 == 0, "Need equal padding"

        if spatial_diff > 0:  # Need to zero pad
            half_diff = int(spatial_diff / 2)
            return F.pad(data, (half_diff, half_diff, half_diff, half_diff))
        elif spatial_diff < 0:  # Need to trim
            half_diff = -int(spatial_diff / 2)
            return data[..., half_diff: -half_diff, half_diff: -half_diff]
        else:
            return data


class NeuralDataset(torch.utils.data.Dataset):

    MODEL_HZ = 240

    def __init__(self, root, train, spatial_scale, max_dim, subclip_ms, pad, luminance, cell_idx, x_hz, y_hz):
        self._root = root
        self._train = train
        self._spatial_scale = spatial_scale
        self._max_dim = max_dim
        self._subclip_ms = subclip_ms
        self._pad = pad
        self._luminance = luminance
        self._cell_idx = cell_idx
        self._x_hz = x_hz
        self._y_hz = y_hz

        self._original_x = self._load_x()  # clips x time x h x w
        self._original_y = self._load_y()  # clips x time x repeat x neuron

        mean, std = self._get_train_mean_and_std()
        self._transformed_x = XTransformer(spatial_scale, max_dim, x_hz, mean, std, luminance, pad)(self._original_x)
        self._transformed_y = YTransformer(y_hz)(self._original_y)
        self._run_checks()
        self._slice_out_fold()
        self._run_checks()

        # Post normalise x transforms
        self._idxs = NeuralDataset.build_idxs(subclip_ms, self._transformed_x.shape[0], self._transformed_x.shape[1])

    def __getitem__(self, i):
        clip_idx, t_start_idx, t_end_idx = self._idxs[i]

        x = self._transformed_x[clip_idx, t_start_idx: t_end_idx]
        y = self._transformed_y[clip_idx, t_start_idx: t_end_idx]

        x = x.unsqueeze(0)  # Add channel dimension
        y = y.mean(1)  # Average over trials to get spike probability

        if self._cell_idx is not None:
            y = y[:, self._cell_idx].unsqueeze(1)

        return x, y

    @property
    def hyperparams(self):
        return {"name": self.__class__.__name__, "train": self._train, "spatial_scale": self._spatial_scale, "max_dim": self._max_dim, "subclip_ms": self._subclip_ms, "pad": self._pad, "luminance": self._luminance, "cell_idx": self._cell_idx}

    def __len__(self):
        return len(self._idxs)

    def _get_train_mean_and_std(self):
        raise NotImplementedError

    def _slice_out_fold(self):
        pass

    def _load_x(self):
        raise NotImplementedError

    def _load_y(self):
        raise NotImplementedError

    def _run_checks(self):
        assert self._transformed_x.shape[0] == self._transformed_y.shape[0],  f"Same number of clips {self._transformed_x.shape[0]} {self._transformed_y.shape[0]}"
        assert self._transformed_x.shape[1] == self._transformed_y.shape[1], f"Not Same number of time steps {self._transformed_x.shape[1]} {self._transformed_y.shape[1]}"
        assert self._transformed_x.shape[2] == self._transformed_x.shape[3], f"Same spatial dims {self._transformed_x.shape[2]} {self._transformed_x.shape[3]}"
        assert self._transformed_x.shape[2] % 2 == 0, f"Ensure spatial dims are even {self._transformed_x.shape[2] % 2}"

    @staticmethod
    def build_idxs(subclip_ms, n_clips, dataset_t_len):
        frame_dt = 1000 / NeuralDataset.MODEL_HZ
        if subclip_ms is None:
            subclip_len = dataset_t_len
        else:
            subclip_len = round(subclip_ms / frame_dt)

        idxs = []
        n_steps = max(int(dataset_t_len / subclip_len), 1)

        for clip_idx in range(n_clips):
            for time_idx in range(n_steps):
                t_start_idx = time_idx * subclip_len
                t_end_idx = (time_idx + 1) * subclip_len
                idxs.append((clip_idx, t_start_idx, t_end_idx))

        return idxs
