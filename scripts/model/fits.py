import os

root = os.path.expanduser("~/PycharmProjects/RetinalModel")
dataset_path = "/home/datasets/natural"


def train(lam, noise_type, photo_noise, ganglion_noise, pred_ms, decoder_span=16):
    name = f"{lam}_{noise_type}_{photo_noise}_{ganglion_noise}_{pred_ms}_{decoder_span}"
    os.system(f"python {root}/scripts/model/train.py --root={root} --photo_noise={photo_noise} --ganglion_noise={ganglion_noise} --noise_type={noise_type} --dataset_path={dataset_path} --lam={lam} --prediction_offset_ms={pred_ms} --decoder_span={decoder_span} --id={name}")


for pred_ms in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 136, 144, 152]:
    train(10**-2.5, "*", 0.01, 0.6, pred_ms, 8)

# Train prediction model without any metabolic regularization
train(0, "*", 0.01, 0.6, 128, 8)
