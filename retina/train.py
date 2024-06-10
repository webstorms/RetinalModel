import torch
import torch.nn.functional as F
import numpy as np
import devtorch

from retina import models


class Trainer(devtorch.Trainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, lam, gamma, warmup, crop, prediction_offset=0, predictive_coding=False, device="cuda", id=None):
        super().__init__(model, train_dataset, root, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, scheduler_func=torch.optim.lr_scheduler.MultiStepLR, scheduler_kwargs={"milestones": [50, 130], "gamma": 0.1}, loader_kwargs={"shuffle": True, "pin_memory": True,  "num_workers": 1}, device=device, grad_clip_type=None, grad_clip_value=0, id=id)
        self._lam = lam
        self._gamma = gamma
        self._warmup = warmup
        self._crop = crop

        self._normative_loss_function = NormativeLoss(warmup, crop, prediction_offset, predictive_coding)

        self._normative_loss = 0
        self._graded_loss = 0
        self._spiking_loss = 0
        self._min_loss = np.inf
        self.log = {**self.log, "normative_loss": [], "graded_loss": [], "spiking_loss": []}

    @staticmethod
    def load_model(root, model_id, override_kwargs={}):

        def model_loader(hyperparams):
            model_params = hyperparams["model"]["params"]

            for key, value in override_kwargs.items():
                model_params[key] = value

            return models.RetinaModel(models.RetinalParameters(**model_params))

        return devtorch.load_model(root, model_id, model_loader)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "lam": self._lam, "gamma": self._gamma, "warmup": self._warmup, "crop": self._crop}

    def on_epoch_complete(self, save):

        def _write_and_reset_loss_variables():
            train_loss = self.log["train_loss"][-1]
            self.log["normative_loss"].append(self._normative_loss)
            self.log["graded_loss"].append(self._graded_loss)
            self.log["spiking_loss"].append(self._spiking_loss)
            print(f"Epoch train_loss={train_loss:.4f} (normative_loss={self._normative_loss:.4f} graded_loss={self._graded_loss:.4f} spiking_loss={self._spiking_loss:.4f})")

            self._normative_loss = 0
            self._graded_loss = 0
            self._spiking_loss = 0

        def _save_model_if_best():
            train_loss = self.log["train_loss"][-1]

            if train_loss < self._min_loss:
                print(f"Saving model train_loss={train_loss:.4f} < min_loss={self._min_loss:.4f}")
                self._min_loss = train_loss
                self.save_model()

        _write_and_reset_loss_variables()
        if save:
            _save_model_if_best()
            self.save_model_log()  # Incase we do not complete all epochs

    def loss(self, output, target, model):
        abs_graded_current = output[1]
        abs_spiking_current = output[2]
        output = output[0]

        normative_loss = self._normative_loss_function(output, target)
        reg_loss = self._lam * (self._gamma * abs_graded_current + (1 - self._gamma) * abs_spiking_current)
        total_loss = normative_loss + reg_loss

        # Track loss variables
        with torch.no_grad():
            self._normative_loss += normative_loss.item()
            self._graded_loss += abs_graded_current.mean().item()
            self._spiking_loss += abs_spiking_current.mean().item()

        return total_loss


class NormativeLoss:

    def __init__(self, warmup, crop, prediction_offset, predictive_coding):
        self._warmup = warmup
        self._crop = crop
        self._prediction_offset = prediction_offset
        self._predictive_coding = predictive_coding

    def __call__(self, output, target):
        # Remove warmup frames and align output with target (due to encoder span creating misalignment)
        encoder_span_offset = target.shape[2] - output.shape[2]
        output = output[:, :, self._warmup:]
        target = target[:, :, self._warmup + encoder_span_offset:]
        assert output.shape[2] == target.shape[2]

        # Remove boarders
        if self._crop > 0:
            output = output[:, :, :, self._crop:-self._crop, self._crop:-self._crop]
            target = target[:, :, :, self._crop:-self._crop, self._crop:-self._crop]

        # Normative loss
        if self._prediction_offset == 0:  # Compression
            return F.mse_loss(output, target, reduction="mean")
        elif self._prediction_offset > 0:  # Predict the future
            if not self._predictive_coding:
                return F.mse_loss(output[:, :, :-self._prediction_offset], target[:, :, self._prediction_offset:], reduction="mean")
            else:
                prediction_errors = target[:, :, self._prediction_offset:] - target[:, :, :-self._prediction_offset]
                return F.mse_loss(output[:, :, :-self._prediction_offset], prediction_errors, reduction="mean")

        elif self._prediction_offset < 0:  # Encode the past
            return F.mse_loss(output[:, :, self._prediction_offset:], target[:, :, :-self._prediction_offset], reduction="mean")
