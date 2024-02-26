import sys
import ast
import logging
import argparse

from retina import dataset, models, train

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def eval(v):
    return ast.literal_eval(v)


def get_dataset(args):
    temp_len = int(args.temp_len_ms / 8)
    n_frame_ext = int(8 / args.dt)
    return dataset.PatchNaturalDataset(root=args.dataset_path, train=True, temp_len=temp_len, kernel=args.rf_size, flip=eval(args.flip), n_frame_ext=n_frame_ext)


def get_model(args):
    if args.model_id != "":
        print(f"Loading {args.model_id}")
        return train.Trainer.load_model(f"{args.root}/results", args.model_id)

    encoder_span = int(args.encoder_span_ms / args.dt)
    decoder_span = int(args.decoder_span_ms / args.dt)
    params = models.RetinalParameters(args.n_in, args.rf_size, encoder_span, decoder_span, args.photo_noise, args.ganglion_noise, args.noise_type, args.mem_tc, args.dt)
    return models.RetinaModel(params)


def get_trainer(args, model, train_dataset):
    warmup = int(args.warmup_ms / args.dt)
    prediction_offset = int(args.prediction_offset_ms / args.dt)
    return train.Trainer(f"{args.root}/results", model, train_dataset, args.n_epochs, args.batch_size, args.lr, args.lam, args.gamma, warmup, args.crop, prediction_offset, device="cuda", id=args.id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")

    # Model
    parser.add_argument("--n_in", type=int, default=400)
    parser.add_argument("--rf_size", type=int, default=20)
    parser.add_argument("--encoder_span_ms", type=int, default=120)
    parser.add_argument("--decoder_span_ms", type=int, default=12)
    parser.add_argument("--photo_noise", type=float, default=0.0)
    parser.add_argument("--ganglion_noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="*")
    parser.add_argument("--mem_tc", type=float, default=20)
    parser.add_argument("--dt", type=int, default=4)
    parser.add_argument("--model_id", type=str, default="")

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--temp_len_ms", type=int, default=320)
    parser.add_argument("--flip", type=str, default="True")

    # Trainer
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lam", type=float, default=10**-3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--warmup_ms", type=int, default=40)
    parser.add_argument("--crop", type=int, default=3)
    parser.add_argument("--prediction_offset_ms", type=int, default=60)
    parser.add_argument('--id', type=str, default="")

    args = parser.parse_args()

    train_dataset = get_dataset(args)
    model = get_model(args)
    model_trainer = get_trainer(args, model, train_dataset)
    model_trainer.train(save=True)


if __name__ == "__main__":
    main()
