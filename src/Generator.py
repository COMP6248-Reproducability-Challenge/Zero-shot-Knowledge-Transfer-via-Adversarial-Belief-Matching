# Background from https://blog.usejournal.com/train-your-first-gan-model-from-scratch-using-pytorch-9b72987fd2c0
# based on documentation of page https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# "We use a generic generator with only three convolutional layers"
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.Sequential(
            nn.Linear(x, 128*8**2),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3, affine=True)
        )

    def forward(self, x):
        """
            - generate gaussian noise
            - feed the noise to layers
            - that returns noised based x
        """
        z = nn.Linear(100, 128*64).view(-1, 128,8,8) # not sure this is how you introduce noise TODO: check noise function
        z = self.layers(x)

        return z

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)