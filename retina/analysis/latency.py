import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brainbox.neural
import devtorch

from retina import train, analysis
import retina.neural.dataset as neural


def get_larva_sequence(root, begin_pad=29, end_pad=10, n_frames=36, lum=0.5):
    def img_to_clip(begin_pad, end_pad, n_frames, img, lum=0.5):
        h, w = img.shape
        t = begin_pad + end_pad + n_frames
        frames_tensor = lum * torch.ones(1, 1, t, h, w)
        frames_tensor[0, 0, begin_pad: begin_pad+n_frames] = img

        return frames_tensor

    def get_larva_image(root):
        img = plt.imread(f"{root}/data/latency/proc_larva2.png")
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img *= 255

        img = (img - 115.97) / 63.60
        img = torch.from_numpy(img)
        img = img[::6, ::6]

        return img * 1

    larva_image = get_larva_image(root)

    return img_to_clip(begin_pad, end_pad, n_frames, larva_image, lum)


class LatencyQuery:

    def __init__(self, root, clip, pred_offset=128):
        model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")

        with torch.no_grad():
            spikes = model(clip.cuda(), mode="just_spikes", stride=2)
            self.spikes = spikes

    def to_latency_code(self):
        latencies = self.spikes.argmax(2)

        return latencies

    def to_relative_latency_code(self, max_v):
        non_spike_mask = self.spikes.sum(2) == 0
        latencies = self.spikes.argmax(2) + non_spike_mask.float() * max_v
        mean_latency = latencies.mean()

        return latencies - mean_latency

    def to_rate_code(self):
        return self.spikes.sum(2)


class NoiseReconstructionDatasetBuilder:

    def __init__(self, root, pred_offset=128, istrain=True, noise=0, length=36):
        self._length = length
        self._model = train.Trainer.load_model(f"{root}/results", f"0.0031622776601683794_*_0.01_0.6_{pred_offset}_8")
        dataset = neural.SalamanderImage(root, train=istrain, spatial_scale=0.25, max_dim=64)

        self.x_clip = dataset._transformed_x[:, :length]
        self.x_frame = self.x_clip[:, 0]
        self.noise_frame = noise * self._get_noise(self.x_clip)
        self.x_frame_noise = self.x_frame + self.noise_frame

        self.clip = self.x_clip + self.noise_frame.unsqueeze(1).repeat(1, length, 1, 1)  # Use same temporal noise profile

    def build(self):
        self.spikes = self._get_and_cache_neural_responses(self.clip)
        self.rate_code = self._get_rate_code(self.spikes)
        self.latency_code = self._get_latency_code(self.spikes)

    def _get_rate_code(self, spikes):
        return spikes.sum(2).float()

    def _get_latency_code(self, spikes):
        non_spike_mask = spikes.sum(2) == 0
        latencies = spikes.argmax(2) + non_spike_mask.float() * self._length
        return latencies - latencies.mean()

    def _get_noise(self, train_x):
        means = torch.zeros_like(train_x)
        std = torch.ones_like(train_x)
        torch.manual_seed(42)
        noise_tensor = torch.normal(means, std)

        return noise_tensor[:, 0]

    def _get_and_cache_neural_responses(self, x_clip_noise):
        stim_response_list = []

        for i in range(x_clip_noise.shape[0]):
            with torch.no_grad():
                clip = x_clip_noise[i].unsqueeze(0).unsqueeze(0).cuda()
                clip = F.pad(clip, (0, 0, 0, 0, 29, 0))
                spikes = self._model(clip, mode="just_spikes", stride=4)
                stim_response_list.append(spikes.cpu())

        return torch.cat(stim_response_list)


class NoiseReconstructionDataset(torch.utils.data.Dataset):

    def __init__(self, root, pred_offset=128, istrain=True, noise=0, length=36, noise_target=False, code="latency"):
        self._pred_offset = pred_offset
        self._noise = noise
        self._length = length
        self._noise_target = noise_target
        self._code = code
        assert code in ["latency", "rate"]
        name = f"{pred_offset}_{length}_{noise}"
        output_path = f"{root}/data/recon/{name}"
        target_x = "latency_code" if code == "latency" else "rate_code"
        target_y = "x_frame_noise" if noise_target else "x_frame"

        self.x = torch.load(f"{output_path}/{istrain}_{target_x}.pt")
        self.y = torch.load(f"{output_path}/{istrain}_{target_y}.pt")
        self.frame_with_noise = torch.load(f"{output_path}/{istrain}_x_frame_noise.pt")
        self.noise_frame = torch.load(f"{output_path}/{istrain}_noise_frame.pt")

        self.x /= length  # Scale data to avoid large gradients in fitting

    def __getitem__(self, i):
        return self.x[i].flatten(0, 2), self.y[i]

    def __len__(self):
        return self.x.shape[0]

    @property
    def hyperparams(self):
        return {"pred_offset": self._pred_offset, "noise": self._noise, "length": self._length, "noise_target": self._noise_target, "code": self._code}


class LN(devtorch.DevModel):

    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(57600, 64 * 64)
        self.init_weight(self._linear.weight, "uniform", a=-1 / np.sqrt(57600), b=1 / np.sqrt(57600))

    def get_params(self):
        return [self._linear.weight]

    def forward(self, x):
        out = self._linear(x).view(x.shape[0], 64, 64)

        return out


class LNTrainer(devtorch.Trainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, lam, shuffle=True, device="cuda", id=None):
        super().__init__(model, train_dataset, root, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, scheduler_func=None, scheduler_kwargs={}, loader_kwargs={"shuffle": shuffle}, device=device, grad_clip_type=None, grad_clip_value=0, id=id)
        self._lam = lam

    @staticmethod
    def load_model(root, model_id, override_kwargs={}):

        def model_loader(hyperparams):
            model_params = hyperparams["model"]

            return analysis.LN()

        return devtorch.load_model(root, model_id, model_loader)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "lam": self._lam}

    def on_epoch_complete(self, save, epoch):
        if save:
            self.save_model()
            self.save_log()
            self.save_hyperparams()

    def loss(self, output, target, model):
        pred_loss = F.mse_loss(output, target)

        # Compute reg loss
        reg_loss = 0
        for param in model.get_params():
            reg_loss = reg_loss + self._lam * torch.norm(param, p=1)
        total_loss = pred_loss + reg_loss

        return total_loss

    def train(self, save=False):
        super().train(save)


class CrossValidationLNTrainer(devtorch.KFoldValidationTrainer):

    LAMBDAS = [10**-4.5, 10**-5, 10**-5.5, 10**-6, 10**-6.5]

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, k, final_epochs=None):
        val_batch_size = batch_size
        val_loss = lambda output, target: F.mse_loss(output, target)
        trainer_kwargs = {"n_epochs": n_epochs, "batch_size": batch_size, "lr": lr}
        super().__init__(root, model, train_dataset, LNTrainer, trainer_kwargs, CrossValidationLNTrainer.LAMBDAS, k, minimise_score=True, final_repeat=1, val_loss=val_loss, val_batch_size=val_batch_size, final_epochs=final_epochs)


class LNFitAccBuilder:

    def __init__(self, root, pred_offset=128, noise=0.0, length=72, noise_target=False, code="latency"):
        self._root = root
        self.test_dataset = analysis.NoiseReconstructionDataset(root, pred_offset=pred_offset, istrain=False, noise=noise, length=length, noise_target=noise_target, code=code)
        model_root = f"{root}/recon/{noise}_{code}/models"
        self.readout = analysis.LNTrainer.load_model(model_root, "0")
        self.cc = self._get_cc()

    @staticmethod
    def get_list_images(root, noise, image_idx):
        ln_fit_acc_builder = analysis.LNFitAccBuilder(root, noise=noise, code="latency")
        with torch.no_grad():
            x = ln_fit_acc_builder.test_dataset.x.flatten(1, 3)
            prediction = ln_fit_acc_builder.readout(x.cuda()).cpu()
        latency_image = prediction[image_idx]

        ln_fit_acc_builder = analysis.LNFitAccBuilder(root, noise=noise, code="rate")
        with torch.no_grad():
            x = ln_fit_acc_builder.test_dataset.x.flatten(1, 3)
            prediction = ln_fit_acc_builder.readout(x.cuda()).cpu()
        rate_image = prediction[image_idx]
        input_image = ln_fit_acc_builder.test_dataset.frame_with_noise[image_idx]
        target_image = ln_fit_acc_builder.test_dataset.y[image_idx]

        return input_image, rate_image, latency_image, target_image

    def build_all_fit_df(self):
        fit_list = []

        for noise in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            for code in ["latency", "rate"]:
                ln_fit_acc_builder = LNFitAccBuilder(self._root, noise=noise, code=code)
                for cc_v in ln_fit_acc_builder.cc:
                    fit_list.append({"code": code, "noise": noise, "cc": cc_v.item()})

        return pd.DataFrame(fit_list)

    def _get_cc(self):
        with torch.no_grad():
            x = self.test_dataset.x.flatten(1, 3)
            prediction = self.readout(x.cuda()).cpu()

            return brainbox.neural.cc(prediction.flatten(1, 2).cpu(), self.test_dataset.y.flatten(1, 2))
