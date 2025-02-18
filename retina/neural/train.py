from pathlib import Path

import torch
import torch.nn.functional as F
import devtorch

from retina.neural.glm import GLM


class Trainer(devtorch.Trainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, lam, shuffle=True, device="cuda", id=None):
        super().__init__(root, model, train_dataset, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, scheduler_func=None, scheduler_kwargs={}, loader_kwargs={"shuffle": shuffle}, device=device, grad_clip_type=None, grad_clip_value=0, id=id)
        self._lam = lam

    @staticmethod
    def load_model(root, model_id, override_kwargs={}):

        def model_loader(hyperparams):
            model_params = hyperparams["model"]

            return GLM(model_params["n_in"], model_params["n_out"], model_params["h"], model_params["w"], model_params["rf_len_ms"])

        return devtorch.load_model(root, model_id, model_loader)

    @staticmethod
    def prediction_loss(output, target):
        t_offset = target.shape[1] - output.shape[1]
        target = target[:, t_offset:, :]

        assert output.shape == target.shape, f"Not same dims {output.shape} {target.shape}"

        return F.poisson_nll_loss(output, target, log_input=False, full=False, eps=1e-8, reduction="mean")

    @property
    def hyperparams(self):
        return {**super().hyperparams, "lam": self._lam}

    def on_epoch_complete(self, save, epoch):
        # Save logs and hyperparams
        if save:
            self.save_model()
            self.save_log()
            self.save_hyperparams()

    def loss(self, output, target, model):
        pred_loss = Trainer.prediction_loss(output, target)

        # # Compute reg loss
        reg_loss = 0
        for param in model.get_params():
            reg_loss = reg_loss + self._lam * torch.norm(param, p=1)
        total_loss = pred_loss + reg_loss

        return total_loss

    def train(self, save=False):
        super().train(save)


class CrossValidationTrainer(devtorch.KFoldValidationTrainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, k, lambdas, final_epochs=None):
        Path(root).mkdir(parents=True, exist_ok=True)
        val_batch_size = batch_size  # None  # len(train_dataset)
        val_loss = lambda output, target: Trainer.prediction_loss(output, target)
        trainer_kwargs = {"n_epochs": n_epochs, "batch_size": batch_size, "lr": lr}
        super().__init__(root, model, train_dataset, Trainer, trainer_kwargs, lambdas, k, minimise_score=True, final_repeat=1, val_loss=val_loss, val_batch_size=val_batch_size, final_epochs=final_epochs)
