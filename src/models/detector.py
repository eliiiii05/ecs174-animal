"""Faster R-CNN with an ImageNet-pretrained ResNet-18 + FPN backbone.

"""
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_detector(num_classes=2, min_size=640, max_size=1066, trainable_layers=3, pretrained_backbone=True):
    """num_classes includes background: 2 = class-agnostic (D1), 11 = 10 animals (D2)."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet18",
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
        trainable_layers=trainable_layers,
    )
    return FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)
