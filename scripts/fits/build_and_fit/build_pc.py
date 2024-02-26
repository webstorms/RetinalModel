import os

from retina import train
import retina.neural.pca as pca
import retina.neural.dataset as neural_dataset

root = os.path.expanduser("~/PycharmProjects/RetinalModel")


def build_pcs(dataset_name, spatial_args, luminance=1):
    for pred_ms in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 136, 144, 152]:
        for spatial_scale, max_dim in spatial_args:
            print(f"Building {dataset_name}: {pred_ms}ms spatial_scale={spatial_scale} max_dim={max_dim}")
            build_output(dataset_name, pred_ms, spatial_scale, max_dim, pad=5, luminance=luminance)


def build_output(dataset_name, pred_ms, spatial_scale, max_dim, pad=5, luminance=1):
    model_name = pca.PCBuilder.model_to_str(lam=10**-2.5, noise_type="*", photo_noise=0.01, ganglion_noise=0.6, pred_ms=pred_ms, decoder_span=8)
    model = train.Trainer.load_model(f"{root}/results", model_name)

    dataset = getattr(neural_dataset, dataset_name)
    train_dataset = dataset(root, True, spatial_scale=spatial_scale, max_dim=max_dim, pad=pad, luminance=luminance)
    test_dataset = dataset(root, False, spatial_scale=spatial_scale, max_dim=max_dim, pad=pad, luminance=luminance)

    pc_builder = pca.PCBuilder(root, lam=10**-2.5, pred_ms=pred_ms, model=model, train_dataset=train_dataset, test_dataset=test_dataset, n_pca=500, pad_t_len=29, stride=4, repeats=8)
    pc_builder.fit_and_transform()
