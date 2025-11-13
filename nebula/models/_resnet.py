from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tv_models


@dataclass
class ResNetConfig:
    num_classes: int = 3
    arch: str = "resnet18"
    pretrained: bool = True
    trainable_layers: int = 2
    dropout: float = 0.3


class ResNetBackbone(nn.Module):

    def __init__(
        self,
        config: ResNetConfig,
    ) -> None:
        super().__init__()

        resnet_ctor = getattr(tv_models, config.arch)
        try:
            backbone = resnet_ctor(pretrained=config.pretrained)
        except TypeError:
            # for torchvision>=0.13, use weights arg
            weights = "IMAGENET1K_V1" if config.pretrained else None
            backbone = resnet_ctor(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # Freeze all, then unfreeze top blocks per trainable_layers count
        for param in backbone.parameters():
            param.requires_grad = False
        if config.trainable_layers >= 1 and hasattr(backbone, "layer4"):
            for param in backbone.layer4.parameters():
                param.requires_grad = True
        if config.trainable_layers >= 2 and hasattr(backbone, "layer3"):
            for param in backbone.layer3.parameters():
                param.requires_grad = True

        self.backbone = backbone
        self.feature_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )

        self.class_scales = nn.Parameter(torch.ones(config.num_classes))
        self.class_embed = nn.Embedding(config.num_classes, 256)

    def forward(self, x):
        z = self.backbone(x)
        out = self.classifier(z)
        out = out * self.class_scales.unsqueeze(0)
        return out, z
