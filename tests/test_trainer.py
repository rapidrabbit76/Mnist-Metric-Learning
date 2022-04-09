import torch
import pytest
from trainer import Model


class TestModel:
    @pytest.fixture(scope="class")
    def model(self, args):
        pl_model = Model(
            args.embedding_size,
            pretrained=args.pretrained,
            backbone=args.backbone,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            num_classes=args.num_classes,
        )
        return pl_model

    def test_training_step(self, batch, model: Model):
        loss = model.training_step(batch, 0)
        assert list(loss.shape) == []

    def test_validation_step(self, args, batch, model: Model):
        embedding, y = model.validation_step(batch, 0)

        assert list(embedding.shape) == [args.batch_size, args.embedding_size]
        assert list(y.shape) == [args.batch_size]
        model.validation_epoch_end([(embedding, y)] * 100)
