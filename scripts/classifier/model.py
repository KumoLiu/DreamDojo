"""ResNet-18 with a stacked-frame input and three ordinal heads."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from scripts.classifier.dataset import NUM_STACK
from scripts.classifier.labels import NUM_HEADS


class MilestoneNet(nn.Module):
    """Predicts, for each milestone, whether it has already happened.

    The three outputs are cumulative rather than mutually exclusive, so they
    are independent sigmoids. Their sum is a progress score in [0, 3].
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = self._inflate(backbone.conv1, NUM_STACK)
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(backbone.fc.in_features, NUM_HEADS)
        )
        self.backbone = backbone

    @staticmethod
    def _inflate(conv: nn.Conv2d, repeats: int) -> nn.Conv2d:
        """Widen the stem to accept a frame stack, preserving pretrained filters.

        Copying the RGB filters across every frame and dividing by the number
        of frames leaves the response to a static scene exactly as ImageNet
        pretraining left it, so the stack starts out as a no-op average and the
        network only has to learn what the differences between frames mean.
        """
        inflated = nn.Conv2d(
            conv.in_channels * repeats,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            inflated.weight.copy_(conv.weight.repeat(1, repeats, 1, 1) / repeats)
            if conv.bias is not None:
                inflated.bias.copy_(conv.bias)
        return inflated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def progress(logits: torch.Tensor) -> torch.Tensor:
    """Expected number of milestones reached, in [0, NUM_HEADS]."""
    return torch.sigmoid(logits).sum(-1)
