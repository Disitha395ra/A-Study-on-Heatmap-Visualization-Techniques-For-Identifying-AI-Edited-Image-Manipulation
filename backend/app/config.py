from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["edited", "real"]

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_UPLOAD_SIZE_MB = 15
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


MODEL_CONFIGS = {
    "resnet50": {
        "display_name": "ResNet50",
        "experiment_name": "resnet50_autosplice",
        "reference_accuracy": 0.9114,
        "reference_f1": 0.9132,
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "experiment_name": "efficientnet_b0_autosplice",
        "reference_accuracy": 0.9127,
        "reference_f1": 0.9142,
    },
    "convnext_tiny": {
        "display_name": "ConvNeXt Tiny",
        "experiment_name": "convnext_tiny_autosplice",
        "reference_accuracy": 0.9658,
        "reference_f1": 0.9660,
    },
    "swin_tiny": {
        "display_name": "Swin Tiny",
        "experiment_name": "swin_tiny_autosplice",
        "reference_accuracy": 0.9329,
        "reference_f1": 0.9340,
    },
    "vit_b16": {
        "display_name": "ViT-B/16",
        "experiment_name": "vit_b16_autosplice",
        "reference_accuracy": 0.8709,
        "reference_f1": 0.8713,
    },
}


def get_checkpoint_path(model_name: str) -> Path:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    experiment_name = MODEL_CONFIGS[model_name]["experiment_name"]

    return (
        PROJECT_ROOT
        / "outputs"
        / "training"
        / experiment_name
        / "checkpoints"
        / "best_model.pth"
    )