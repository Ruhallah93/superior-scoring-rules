"""Backbones matching experiments/config.yaml's `backbones` section.

    smallcnn:
      conv: [[32, 3], [64, 3]]
      dropout: 0.3
      hidden: 128
      note: "Deliberately under-capacity, to keep predictions diffuse."

    resnet18:
      pretrained: false
      dropout: 0.0

Both backbones must accept either MNIST (1x28x28) or CIFAR-10 (3x32x32)
input, and output `classes` logits (10 for both datasets in this design).
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Two conv blocks, deliberately under-capacity (config's own note)."""

    def __init__(self, in_channels, classes, conv=((32, 3), (64, 3)),
                 dropout=0.3, hidden=128, input_hw=28):
        super().__init__()
        layers = []
        c_in = in_channels
        hw = input_hw
        for c_out, k in conv:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
            hw = hw // 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in * hw * hw, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet18(nn.Module):
    """CIFAR/MNIST-scale ResNet-18: 3x3 stride-1 stem, no initial maxpool
    (the standard adaptation for 28x28/32x32 input; the torchvision
    ImageNet-scale stem would collapse a 28x28 image before the first
    residual block). `pretrained: false` per config -- trained from scratch.
    """

    def __init__(self, in_channels, classes, dropout=0.0):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, 1, dropout)
        self.layer2 = self._make_layer(128, 2, 2, dropout)
        self.layer3 = self._make_layer(256, 2, 2, dropout)
        self.layer4 = self._make_layer(512, 2, 2, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, classes)

    def _make_layer(self, planes, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_backbone(name, in_channels, classes, input_hw, cfg_backbones):
    spec = cfg_backbones[name]
    if name == "smallcnn":
        return SmallCNN(in_channels, classes, conv=[tuple(c) for c in spec["conv"]],
                         dropout=spec["dropout"], hidden=spec["hidden"], input_hw=input_hw)
    elif name == "resnet18":
        return ResNet18(in_channels, classes, dropout=spec.get("dropout", 0.0))
    raise ValueError(f"unknown backbone {name!r}")
