import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torchvision.transforms import Compose, ToTensor, Normalize
from datamodule import DATA_MODULE_TABLE
from model import MODEL_TABLE
from trainer import Model


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # project
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--root_dir", type=str, default="DATASET")

    # data
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATA_MODULE_TABLE.keys()),
    )

    # model
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument(
        "--backbone",
        type=str,
        default="ResNet18",
        choices=list(MODEL_TABLE.keys()),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ArcFace",
    )
    parser.add_argument("--margin", type=float, default=28.6)
    parser.add_argument("--scale", type=int, default=64)

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    args = pl.Trainer.parse_argparser(parser.parse_args())

    transform = Compose(
        [
            ToTensor(),
            Normalize([0.5], 0.5),
        ]
    )

    dm = DATA_MODULE_TABLE[args.dataset](
        root_dir=args.root_dir,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    callbacks = [TQDMProgressBar(10)]

    model = Model(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
