import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_tiny",
    "vit_b16",
]


def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Create a classification model with a binary output layer.

    Supported:
    - resnet50
    - efficientnet_b0
    - convnext_tiny
    - swin_tiny
    - vit_b16
    """

    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer_name = "layer4"

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer_name = "features"

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        target_layer_name = "features"

    elif model_name == "swin_tiny":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        target_layer_name = "features"

    elif model_name == "vit_b16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        target_layer_name = "encoder"

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {SUPPORTED_MODELS}"
        )

    return model, target_layer_name


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable