"""Rectangular CNO and U-Net backbones for UrbanFlow-3K mean velocity."""
import torch
from torch import nn
from torch.nn import functional as F

from _cno_upstream import CNO2d, CNO_LReLu


class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.GroupNorm(1, channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.GroupNorm(1, channels_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Four encoder widths of 24, 48, 96 and 192 channels."""

    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([
            DoubleConv(5, 24), DoubleConv(24, 48),
            DoubleConv(48, 96), DoubleConv(96, 192),
        ])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(192, 96, 2, stride=2),
            nn.ConvTranspose2d(96, 48, 2, stride=2),
            nn.ConvTranspose2d(48, 24, 2, stride=2),
        ])
        self.dec = nn.ModuleList([
            DoubleConv(192, 96), DoubleConv(96, 48), DoubleConv(48, 24),
        ])
        self.out = nn.Conv2d(24, 2, 1)

    def forward(self, x):
        skips = []
        for level, block in enumerate(self.enc):
            if level:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for upsample, block, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = block(torch.cat((upsample(x), skip), dim=1))
        return self.out(x)


class RectangularActivation(nn.Module):
    """Adapt the CNO activation to the rectangular resolution at each level."""

    def __init__(self, original, sizes):
        super().__init__()
        self.in_size = sizes[original.in_size]
        self.out_size = sizes[original.out_size]
        self.act = original.act

    def forward(self, x):
        doubled = tuple(2 * side for side in self.in_size)
        x = F.interpolate(x, size=doubled, mode="bicubic", antialias=True)
        return F.interpolate(self.act(x), size=self.out_size,
                             mode="bicubic", antialias=True)


def rectangular_cno():
    core = CNO2d(5, 2, 160, 3, N_res=1, N_res_neck=2, channel_multiplier=30)
    sizes = {160 // 2**i: (160 // 2**i, 200 // 2**i) for i in range(4)}

    def adapt(module):
        for name, child in list(module.named_children()):
            if isinstance(child, CNO_LReLu):
                setattr(module, name, RectangularActivation(child, sizes))
            else:
                adapt(child)

    adapt(core)
    return core


class MeanVelocityModel(nn.Module):
    """Map (N, 5, 140, 180) inputs to (N, 2, 140, 180) normalized velocities."""

    def __init__(self, backbone):
        super().__init__()
        self.core = backbone

    def forward(self, features):
        if features.ndim != 4 or features.shape[1:] != (5, 140, 180):
            raise ValueError("Expected an (N, 5, 140, 180) feature tensor.")
        padded = F.pad(features, (10, 10, 10, 10))
        return self.core(padded)[..., 10:-10, 10:-10]


def build_model(name):
    """Construct the model used in the comparison with randomly initialized weights."""
    if name == "cno_rect":
        return MeanVelocityModel(rectangular_cno())
    if name == "unet":
        return MeanVelocityModel(UNet())
    raise ValueError(f"Unknown model {name!r}. Choose 'cno_rect' or 'unet'.")
