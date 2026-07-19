from typing import Dict

import torch
import torch.nn as nn
import gc

from backend.app.config import (
    MODEL_CONFIGS,
    NUM_CLASSES,
    get_checkpoint_path,
)
from models.model_factory import create_model


class ModelService:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.models: Dict[str, nn.Module] = {}
        self.target_layer_names: Dict[str, str] = {}
        self.loaded = False

    def load_all_models(self) -> None:
        if self.loaded:
            return

        print("=" * 70)
        print("Initializing model service for web application")
        print("=" * 70)
        print("Device:", self.device)

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        # We no longer preload all models to prevent PyTorch OOM on Render
        self.loaded = True

        print("\nLazy loading enabled. Models will be loaded on demand.")
        print("=" * 70)

    def get_model(self, model_name: str) -> nn.Module:
        if not self.loaded:
            raise RuntimeError(
                "Models have not been loaded."
            )

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model not available: {model_name}"
            )

        if model_name not in self.models:
            # Enforce max 1 model in memory for Render free tier (512MB RAM)
            if len(self.models) >= 1:
                self.models.clear()
                self.target_layer_names.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"Loading {model_name} on demand...")
            model, target_layer_name = create_model(
                model_name=model_name,
                num_classes=NUM_CLASSES,
                pretrained=False,
            )

            checkpoint_path = get_checkpoint_path(model_name)

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model = model.to(self.device)
            model.eval()

            self.models[model_name] = model
            self.target_layer_names[model_name] = (
                target_layer_name
            )

        return self.models[model_name]

    def model_status(self) -> dict:
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "loaded": self.loaded,
            "loaded_models": list(self.models.keys()),
            "model_count": len(self.models),
        }


model_service = ModelService()