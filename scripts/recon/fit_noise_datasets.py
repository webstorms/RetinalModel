from pathlib import Path

from retina import analysis

root = str(Path(__file__).resolve().parents[2])


def fit_model(noise, code):
    train_dataset = analysis.NoiseReconstructionDataset(root, pred_offset=128, istrain=True, noise=noise, length=72, noise_target=False, code=code)
    ln = analysis.LN()
    name = f"{noise}_{code}"
    cross_root = f"{root}/data/recon/{name}"
    cross_val = analysis.CrossValidationLNTrainer(cross_root, ln, train_dataset, n_epochs=200, batch_size=32, lr=10**-5, k=5)
    cross_val.train()


for noise in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
    for code in ["latency", "rate"]:
        fit_model(noise, code)
