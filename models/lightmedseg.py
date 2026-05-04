from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Small convolution block used throughout the lightweight U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class LightweightHybridAttention(nn.Module):
    """Channel-spatial attention for emphasizing small foreground regions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.channel_gate(x)
        spatial_context = x.mean(dim=1, keepdim=True)
        return x * self.spatial_gate(spatial_context)


class DomainFeatureCalibration(nn.Module):
    """Per-sample feature distribution calibration at the bottleneck."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.gamma + self.beta


class EdgeGuidedFeatureFusion(nn.Module):
    """Predicts an edge prior and fuses it back into the decoder feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_features = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        edge_logits = self.edge_features(x)
        fused = self.fuse(torch.cat([x, torch.sigmoid(edge_logits)], dim=1))
        return fused, edge_logits


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.attention = LightweightHybridAttention(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.attention(self.conv(x))


@dataclass(frozen=True)
class LightMedSegOutput:
    mask_logits: Tensor
    edge_logits: Tensor


class LightMedSeg2D(nn.Module):
    """LightMedSeg-2D network for binary or multi-class 2D medical segmentation.

    By default the module returns only segmentation logits with shape
    ``[B, num_classes, H, W]``. Set ``return_aux=True`` to also get edge logits
    for auxiliary edge supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.domain_calibration = DomainFeatureCalibration(c4)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.edge_fusion = EdgeGuidedFeatureFusion(c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.segmentation_head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: Tensor, return_aux: bool = False) -> Tensor | LightMedSegOutput:
        input_size = x.shape[-2:]

        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        bottleneck = self.domain_calibration(self.down3(skip3))

        x = self.up3(bottleneck, skip3)
        x = self.up2(x, skip2)
        x, edge_logits = self.edge_fusion(x)
        x = self.up1(x, skip1)

        mask_logits = self.segmentation_head(x)
        if mask_logits.shape[-2:] != input_size:
            mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits, size=input_size, mode="bilinear", align_corners=False)

        if return_aux:
            return LightMedSegOutput(mask_logits=mask_logits, edge_logits=edge_logits)
        return mask_logits


def lightmedseg_tiny(in_channels: int = 3, num_classes: int = 1) -> LightMedSeg2D:
    return LightMedSeg2D(in_channels=in_channels, num_classes=num_classes, base_channels=16)


def lightmedseg_small(in_channels: int = 3, num_classes: int = 1) -> LightMedSeg2D:
    return LightMedSeg2D(in_channels=in_channels, num_classes=num_classes, base_channels=32)
