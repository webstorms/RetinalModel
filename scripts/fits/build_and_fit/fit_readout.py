import os
import logging
logging.getLogger("trainer").setLevel(logging.WARNING)

from retina.neural import train
import retina.neural.glm as glm
import retina.neural.pca as pca

root = os.path.expanduser("~/PycharmProjects/RetinalModel")


def fit_readouts(n_epochs, batch_size, lambdas, dataset_name, spatial_args, luminance=1, rf_len_ms=21, clamp=False):
    for pred_ms in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 136, 144, 152]:
        for spatial_scale, max_dim in spatial_args:
            print(f"Building {dataset_name}: {pred_ms}ms spatial_scale={spatial_scale} max_dim={max_dim}")
            try:
                fit_readout(n_epochs, batch_size, lambdas, dataset_name, pred_ms, spatial_scale, max_dim, luminance, rf_len_ms, clamp)
            except Exception as e:
                print(f"Probably already build for... {e}")


def load_dataset(dataset_name, spatial_scale=0.1796875, max_dim=46, luminance=0.5, pred_ms=60):
    dataset = pca.PCDataset(root, train=True, n_pca=500, cell_idx=None, subclip_ms=600)
    dataset.load_responses(dataset=dataset_name, spatial_scale=spatial_scale, max_dim=max_dim, luminance=luminance)
    dataset.load_model(lam=10**-2.5, noise_type="*", photo_noise=0.01, ganglion_noise=0.6, pred_ms=pred_ms, decoder_span=2)

    return dataset


def fit_readout(n_epochs, batch_size, lambdas, dataset_name, pred_ms, spatial_scale, max_dim, luminance=1, rf_len_ms=13, clamp=False):
    fit_id = f"{dataset_name}_{pred_ms}_{spatial_scale}_{max_dim}_{luminance}_{rf_len_ms}"
    train_dataset = load_dataset(dataset_name=dataset_name, spatial_scale=spatial_scale, max_dim=max_dim, luminance=luminance, pred_ms=pred_ms)
    readout = glm.GLM(n_in=1, n_out=train_dataset.n_neurons, h=500, w=1, rf_len_ms=rf_len_ms, clamp=clamp)
    cross_val = train.CrossValidationTrainer(f"{root}/fits/{fit_id}", readout, train_dataset, n_epochs=n_epochs, batch_size=batch_size, lr=10**-4, k=5, lambdas=lambdas)
    cross_val.train()
