# Background from https://blog.usejournal.com/train-your-first-gan-model-from-scratch-using-pytorch-9b72987fd2c0
# based on documentation of page https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.linear = nn.Linear(input_dim, 128*8**2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.BatchNorm2d(128)
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.layers = nn.Sequential(
            self.norm,
            self.upsample,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.norm,
            self.activation,
            self.upsample,
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3, affine=True)
        )

    def forward(self, x):
        """
            - generate gaussian noise
            - feed the noise to layers
            - that returns noised based x
        """
        out = self.linear(x)
        out = out.view((-1,128,8,8))
        return self.layers(out)
