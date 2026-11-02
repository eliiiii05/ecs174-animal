"""ResNet-18 classifier: torchvision ResNet-18 with a new 512 -> 10 head."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.common import NUM_CLASSES


def build_classifier(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    """pretrained=True loads ImageNet weights (C1-C3); False = random init (C4).
    freeze_backbone=True trains only the new fc layer (C3, linear probe)."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model


def freeze_batchnorm(model):
    """Keep BatchNorm running stats fixed (used for the linear probe)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
