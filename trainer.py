from typing import List
import os
import pytorch_lightning as pl
from pytorch_metric_learning import miners, losses
import torch.nn as nn
import torch
from model import VGG11, ResNet18
import torch.optim as optim

import umap
import matplotlib.pyplot as plt
import numpy as np


class Model(pl.LightningModule):
    def __init__(
        self,
        seed: int,
        embedding_size: int,
        pretrained: bool,
        backbone: str,
        margin: float,
        scale: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        num_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.backbone = ResNet18(embedding_size, pretrained)
        self.backbone = VGG11(embedding_size, pretrained)
        self.loss = losses.ArcFaceLoss(
            num_classes,
            embedding_size,
            margin=margin,
            scale=scale,
        )
        self.reducer = umap.UMAP(random_state=seed)

    def configure_optimizers(self) -> optim.Optimizer:
        backbone_params = self.backbone.parameters()
        metric_params = self.loss.parameters()
        params = [{"params": backbone_params}, {"params": metric_params}]

        optimizer = optim.SGD(
            params=params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y = y.long()
        embedding = self(x)
        loss = self.loss(embedding, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        embedding = self(x)
        return (embedding, y)

    def validation_epoch_end(self, outputs) -> None:
        torch.set_grad_enabled(False)
        embedding = torch.cat([output[0] for output in outputs], 0)
        y = torch.cat([output[1] for output in outputs], 0)
        embedding = embedding.cpu().numpy()
        y = y.cpu().numpy()

        embedding = self.reducer.fit_transform(embedding)
        fig, ax = plt.subplots(figsize=(11, 10))
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y,
            cmap="Spectral",
            s=0.1,
            alpha=0.8,
        )
        cbar = plt.colorbar(boundaries=np.arange(10))
        cbar.set_ticks(np.arange(10))
        plt.title(f"Epoch:{self.current_epoch}", fontsize=18)
        plt.savefig(
            os.path.join(
                self.logger.experiment.log_dir,
                f"epoch_{self.current_epoch}.png",
            )
        )
        torch.set_grad_enabled(True)
