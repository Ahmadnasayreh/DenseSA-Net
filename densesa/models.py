"""Model definitions.

DenseSANet is the proposed model. AblationVariant covers all eight
combinations of the 2^3 factorial ablation (dilated block, channel
attention, spatial attention); its spatial-only variant (F4) has the same
architecture and state_dict keys as DenseSANet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

FEATURES = 1024   # channels at the output of DenseNet-121


def densenet121_features(pretrained=True):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    return models.densenet121(weights=weights).features


class SpatialAttention(nn.Module):
    """Channel-mean descriptor, 7x7 convolution, sigmoid gate (50 parameters)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2)

    def attention_map(self, x):
        return torch.sigmoid(self.conv(x.mean(dim=1, keepdim=True)))

    def forward(self, x):
        return x * self.attention_map(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation block (reduction 16). Used in the ablation only."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        w = self.gap(x).flatten(1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w[:, :, None, None]


class DilatedBlock(nn.Module):
    """Four dilated convolutions (d=2) with SiLU. Used in the ablation only."""

    def __init__(self, channels):
        super().__init__()
        h = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, h, 3, dilation=2, padding=2), nn.SiLU(),
            nn.Conv2d(h, h, 3, dilation=2, padding=2), nn.SiLU(),
            nn.Conv2d(h, h, 3, dilation=2, padding=2), nn.SiLU(),
            nn.Conv2d(h, channels, 1), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DenseSANet(nn.Module):
    """DenseNet-121 + spatial attention, fused by concatenation.

        F_base    = DenseNet121(x)              (B, 1024, 7, 7)
        A         = sigmoid(conv7x7(mean_c F))  (B, 1, 7, 7)
        F_concat  = [F_base, F_base * A]        (B, 2048, 7, 7)
        logits    = FC(GAP(F_concat))
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.features = densenet121_features(pretrained)
        self.sp_attn = SpatialAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2 * FEATURES, num_classes)
        for p in self.features[:7].parameters():
            p.requires_grad_(False)
        self.last_attention = None   # kept for visualisation

    def forward(self, x):
        f_base = self.features(x)
        a = self.sp_attn.attention_map(f_base)
        self.last_attention = a.detach()
        f = torch.cat([f_base, f_base * a], dim=1)
        return self.head(self.gap(f).flatten(1))

    def freeze_backbone(self, frozen=True):
        for p in self.features.parameters():
            p.requires_grad_(not frozen)


class AblationVariant(nn.Module):
    """One cell of the 2^3 factorial design.

    With every component switched off the head sees F_base only, which is
    a plain DenseNet-121 classifier (variant F1).
    """

    def __init__(self, num_classes, dilated=False, channel=False, spatial=False,
                 pretrained=True):
        super().__init__()
        self.features = densenet121_features(pretrained)
        self.use = (dilated, channel, spatial)
        if dilated:
            self.dilated = DilatedBlock(FEATURES)
        if channel:
            self.ch_attn = ChannelAttention(FEATURES)
        if spatial:
            self.sp_attn = SpatialAttention()
        self.any = dilated or channel or spatial
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2 * FEATURES if self.any else FEATURES, num_classes)
        for p in self.features[:7].parameters():
            p.requires_grad_(False)

    def forward(self, x):
        f_base = self.features(x)
        if not self.any:
            return self.head(self.gap(f_base).flatten(1))
        f = f_base
        if self.use[0]:
            f = self.dilated(f)
        if self.use[1]:
            f = self.ch_attn(f)
        if self.use[2]:
            f = self.sp_attn(f)
        f = torch.cat([f_base, f], dim=1)
        return self.head(self.gap(f).flatten(1))

    def freeze_backbone(self, frozen=True):
        for p in self.features.parameters():
            p.requires_grad_(not frozen)


# (name, label, dilated, channel, spatial)
VARIANTS = [
    ("F1", "Backbone only",           False, False, False),
    ("F2", "+Dilated",                True,  False, False),
    ("F3", "+ChannelAttn",            False, True,  False),
    ("F4", "+SpatialAttn",            False, False, True),
    ("F5", "+Dilated +ChannelAttn",   True,  True,  False),
    ("F6", "+Dilated +SpatialAttn",   True,  False, True),
    ("F7", "+ChannelAttn +SpatialAttn", False, True, True),
    ("F8", "Full (D+C+S)",            True,  True,  True),
]


class VGG16Net(nn.Module):
    """ImageNet VGG16 with a new final layer; trained end to end."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vgg16(weights=weights)
        m.classifier[6] = nn.Linear(4096, num_classes)
        self.m = m

    def forward(self, x):
        return self.m(x)


class CustomCNN(nn.Module):
    """Small task-specific baseline: three conv blocks and two FC layers."""

    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes, pretrained=True):
    if name == "DenseSANet":
        return DenseSANet(num_classes, pretrained)
    if name == "VGG16":
        return VGG16Net(num_classes, pretrained)
    if name == "CustomCNN":
        return CustomCNN(num_classes)
    for tag, _, d, c, s in VARIANTS:
        if name == tag:
            return AblationVariant(num_classes, d, c, s, pretrained)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model):
    """Total parameters and those updated in Stage 1 (backbone frozen)."""
    total = sum(p.numel() for p in model.parameters())
    if hasattr(model, "freeze_backbone"):
        state = [p.requires_grad for p in model.parameters()]
        model.freeze_backbone(True)
        stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for p, s in zip(model.parameters(), state):
            p.requires_grad_(s)
    else:
        stage1 = total
    return total, stage1
