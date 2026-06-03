from __future__ import annotations

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(num_classes: int = 2, weights: str | None = "DEFAULT"):
    weights_arg = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    weights_backbone = None
    if weights is None or weights.lower() in {"none", "false", "0"}:
        weights_arg = None
        weights_backbone = None

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights_arg,
        weights_backbone=weights_backbone,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)
    return model
