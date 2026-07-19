from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    EigenCAM,
    ScoreCAM,
    HiResCAM,
    XGradCAM,
    AblationCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit


SUPPORTED_CAM_METHODS = [
    "gradcam",
    "gradcamplusplus",
    "layercam",
    "eigencam",
    "scorecam",
    "hirescam",
    "xgradcam",
    "ablationcam",
]


def vit_reshape_transform(tensor):
    """
    Torchvision ViT output:
    [batch, 197 tokens, channels]

    Remove CLS token and reshape 196 patch tokens to 14x14.
    """
    result = tensor[:, 1:, :]

    height = width = int(result.shape[1] ** 0.5)

    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2),
    )

    return result.permute(0, 3, 1, 2)


def swin_reshape_transform(tensor):
    """
    Torchvision Swin target-layer output is usually:
    [batch, height, width, channels]

    Convert to:
    [batch, channels, height, width]
    """
    if tensor.dim() != 4:
        raise ValueError(
            f"Expected 4D Swin activation tensor, got shape {tuple(tensor.shape)}"
        )

    return tensor.permute(0, 3, 1, 2)


def get_target_layers(model, model_name):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return [model.layer4[-1]], None

    if model_name == "efficientnet_b0":
        return [model.features[-1]], None

    if model_name == "convnext_tiny":
        return [model.features[-1]], None

    if model_name == "swin_tiny":
        # Last transformer block, before the final attention computation.
        target_layer = model.features[-1][-1].norm1
        return [target_layer], swin_reshape_transform

    if model_name == "vit_b16":
        # Layer before final attention block.
        target_layer = model.encoder.layers[-1].ln_1
        return [target_layer], vit_reshape_transform

    raise ValueError(f"Unsupported model for CAM: {model_name}")


def create_cam_method(
    method_name,
    model,
    model_name,
    target_layers,
    reshape_transform=None,
):
    method_name = method_name.lower()
    model_name = model_name.lower()

    common_kwargs = {
        "model": model,
        "target_layers": target_layers,
    }

    if reshape_transform is not None:
        common_kwargs["reshape_transform"] = reshape_transform

    if method_name == "gradcam":
        cam = GradCAM(**common_kwargs)

    elif method_name == "gradcamplusplus":
        cam = GradCAMPlusPlus(**common_kwargs)

    elif method_name == "layercam":
        cam = LayerCAM(**common_kwargs)

    elif method_name == "eigencam":
        cam = EigenCAM(**common_kwargs)

    elif method_name == "scorecam":
        cam = ScoreCAM(**common_kwargs)

    elif method_name == "hirescam":
        cam = HiResCAM(**common_kwargs)

    elif method_name == "xgradcam":
        cam = XGradCAM(**common_kwargs)

    elif method_name == "ablationcam":
        # Transformer models require token-aware ablation.
        if model_name in {"swin_tiny", "vit_b16"}:
            cam = AblationCAM(
                **common_kwargs,
                ablation_layer=AblationLayerVit(),
            )
        else:
            cam = AblationCAM(**common_kwargs)

    else:
        raise ValueError(f"Unsupported CAM method: {method_name}")

    # ScoreCAM and AblationCAM internally process channels in batches.
    if method_name in {"scorecam", "ablationcam"}:
        cam.batch_size = 16

    return cam