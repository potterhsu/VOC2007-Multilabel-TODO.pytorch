import os
import time

import torch
import torch.nn.functional
from torch import nn, Tensor
from torchvision import models


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        classifier = list(vgg16.classifier.children())
        classifier.pop()
        classifier.append(nn.Linear(4096, 20))
        new_classifier = nn.Sequential(*classifier)
        vgg16.classifier = new_classifier
        self._net = vgg16

    def forward(self, images: Tensor) -> Tensor:
        logits = self._net.forward(images)
        return logits

    def loss(self, logits: Tensor, multilabels: Tensor) -> Tensor:
        cross_entropy = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), multilabels)
        return cross_entropy

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
