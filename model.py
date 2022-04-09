from torchvision.models.vgg import vgg11
from torchvision.models.resnet import resnet18
import torch.nn as nn


def VGG11(embedding_size=2024, pretrained: bool = True):
    model = vgg11(pretrained=pretrained)
    layer = list(model.children())[:2]
    layer[0] = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
    layer += [nn.Flatten()]
    layer += [nn.Linear(3136, embedding_size)]
    return nn.Sequential(*layer)


def ResNet18(embedding_size=512, pretrained: bool = True):
    model = resnet18(pretrained=pretrained)
    layer = list(model.children())[:-1]
    layer[0] = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    layer += [nn.Flatten()]
    layer += [nn.Linear(512, embedding_size)]
    return nn.Sequential(*layer)
