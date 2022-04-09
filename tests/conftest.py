import torch
import torch.nn as nn
import numpy as np
import tempfile
import pytest
from easydict import EasyDict


@pytest.fixture(scope="session")
def args():
    return EasyDict(
        {
            "pretrained": True,
            "embedding_size": 1024,
            "backbone": "ResNet",
            "batch_size": 10,
            "num_classes": 10,
            "margin": 10,
            "scale": 10,
            "lr": 0.0001,
            "momentum": 0.0000,
            "weight_decay": 0.0000,
        }
    )


@pytest.fixture(scope="session")
def save_dir():
    return tempfile.TemporaryDirectory()


def tensor(b, size, c):
    return torch.rand([b, c, size, size])


@pytest.fixture(scope="session")
def batch(args):
    return (
        tensor(args.batch_size, 28, 1),
        torch.randint(low=0, high=9, size=[args.batch_size]),
    )
