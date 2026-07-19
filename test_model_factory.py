import torch
from models.model_factory import create_model, count_parameters, SUPPORTED_MODELS


def main():
    print("=" * 70)
    print("Model Factory Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for model_name in SUPPORTED_MODELS:
        print("\nTesting:", model_name)

        model, target_layer_name = create_model(
            model_name=model_name,
            num_classes=2,
            pretrained=True
        )

        model = model.to(device)
        model.eval()

        total, trainable = count_parameters(model)

        dummy = torch.randn(1, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(dummy)

        print("Output shape:", tuple(output.shape))
        print("Target layer group:", target_layer_name)
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")

    print("\nModel factory test completed successfully.")


if __name__ == "__main__":
    main()