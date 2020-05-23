"""
Implementation of https://arxiv.org/pdf/1905.11946.pdf as a replacement for WRN
"""

from torch.nn import functional as F
from collections import OrderedDict
from torch import nn
import torch
import math


# Using "Swish" activation function
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


# The CNN part
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, dilate=1):
        super(ConvBlock, self).__init__()

        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate

        self.convolutional_sequence = nn.Sequential(OrderedDict([
           ("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("activate", Swish(inplace=True))
        ]))

    # TODO: forward pass


"""
EfficientNet-B0's' main building block is mobile inverted bottleneck MBConv
(Sandler et al., 2018; Tan et al., 2019), to which we also add
squeeze-and-excitation optimization (Hu et al., 2018)
"""

class Squeeze(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(Squeeze, self).__init__()
        self.squeeze_sequence = nn.Sequential(OrderedDict([
            ("L!", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("activate", Swish(inplace=True)),
            ("L2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    # TODO: forward pass


# MBConv from above
class MBConv(nn.Module):
    # set dropout as 0.2 from paper
    """ Code stolen form MBConv paper """
    def __init__(self, in_planes, out_planes, expand_ratio,  kernel_size,
        stride, dilate, reduction_ratio=4, dropout_rate=0.2):
        super(MBConv, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = ConvBlock(in_planes, hidden_dim, 1)

        # step 2. Depth-wise convolution phase
        self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
                                    stride=stride, groups=hidden_dim, dilate=dilate)
        # step 3. Squeeze and Excitation
        if self.use_se:
            self.se_block = Squeeze(hidden_dim, reduced_dim)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_planes, kernel_size=1,
                               stride=1, padding=0, dilation=1, groups=1, bias=False)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))

    def forward(self, x):
        res = x

        # step 1. Expansion phase/Point-wise convolution
        if self.expand_ratio != 1:
            x = self.expansion(x)

        # step 2. Depth-wise convolution phase
        x = self.depth_wise(x)

        # step 3. Squeeze and Excitation
        if self.use_se:
            x = self.se_block(x)

        # step 4. Point-wise convolution phase
        x = self.point_wise(x)

        # step 5. Skip connection and drop connect
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)
            x = x + res

        return x


# TODO: Efficient net implementation