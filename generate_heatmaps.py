import argparse
import csv
import gc
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cam.cam_factory import (
    SUPPORTED_CAM_METHODS,
    create_cam_method,
    get_target_layers,
)
from datasets.dataset_index import build_dataset_index
from models.model_factory import create_model


CLASS_NAMES = ["edited", "real"]
EDITED_CLASS_INDEX = 0


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = transform(image).unsqueeze(0)
    rgb_float = np.asarray(image).astype(np.float32) / 255.0

    return input_tensor, rgb_float


def load_trained_model(
    model_name,
    num_classes,
    checkpoint_path,
    device,
):
    model, _ = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def save_rows(csv_path, rows):
    fieldnames = [
        "experiment_name",
        "model_name",
        "image_name",
        "image_path",
        "mask_path",
        "method",
        "prediction",
        "confidence",
        "prob_edited",
        "prob_real",
        "is_correct",
        "status",
        "error_message",
        "runtime_sec",
        "heatmap_path",
        "overlay_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=SUPPORTED_CAM_METHODS,
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete the existing experiment heatmap folder.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    model_name = config["model_name"]
    image_size = config["image_size"]

    checkpoint_path = (
        root
        / "outputs"
        / "training"
        / experiment_name
        / "checkpoints"
        / "best_model.pth"
    )

    output_dir = (
        root
        / "outputs"
        / "heatmaps"
        / experiment_name
    )

    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "heatmap_generation.csv"
    report_path = output_dir / "heatmap_generation_report.txt"

    methods = [method.lower() for method in args.methods]

    for method in methods:
        if method not in SUPPORTED_CAM_METHODS:
            raise ValueError(
                f"Unsupported CAM method: {method}. "
                f"Available methods: {SUPPORTED_CAM_METHODS}"
            )

    print("=" * 70)
    print(f"Heatmap Generation V2: {experiment_name}")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Model:", model_name)
    print("Methods:", methods)
    print("Checkpoint:", checkpoint_path)

    samples = build_dataset_index(root, args.split)
    edited_samples = [
        sample for sample in samples
        if sample["is_edited"]
    ]

    if args.max_images is not None:
        edited_samples = edited_samples[:args.max_images]

    print("Edited images selected:", len(edited_samples))

    rows = []

    method_success = {
        method: 0 for method in methods
    }

    method_errors = {
        method: 0 for method in methods
    }

    method_runtime = {
        method: 0.0 for method in methods
    }

    total_start = time.time()

    # Run one method at a time using a fresh model.
    for method_index, method_name in enumerate(methods, start=1):
        print("\n" + "=" * 70)
        print(
            f"Method {method_index}/{len(methods)}: "
            f"{method_name}"
        )
        print("=" * 70)

        model = load_trained_model(
            model_name=model_name,
            num_classes=config["num_classes"],
            checkpoint_path=checkpoint_path,
            device=device,
        )

        target_layers, reshape_transform = get_target_layers(
            model,
            model_name,
        )

        cam = create_cam_method(
            method_name=method_name,
            model=model,
            model_name=model_name,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

        method_dir = output_dir / method_name
        heatmap_dir = method_dir / "heatmaps"
        overlay_dir = method_dir / "overlays"

        heatmap_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        targets = [
            ClassifierOutputTarget(EDITED_CLASS_INDEX)
        ]

        for image_index, sample in enumerate(
            edited_samples,
            start=1,
        ):
            image_path = sample["image_path"]
            mask_path = sample["mask_path"]

            input_tensor, rgb_float = preprocess_image(
                image_path,
                image_size,
            )

            input_tensor = input_tensor.to(device)

            try:
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )[0]

                    prediction_index = int(
                        torch.argmax(probabilities).item()
                    )

                    confidence = float(
                        probabilities[prediction_index].item()
                    )

                prediction = CLASS_NAMES[prediction_index]
                is_correct = (
                    prediction_index == EDITED_CLASS_INDEX
                )

                if args.correct_only and not is_correct:
                    continue

                heatmap_path = (
                    heatmap_dir
                    / f"{image_path.stem}_{method_name}.png"
                )

                overlay_path = (
                    overlay_dir
                    / f"{image_path.stem}_{method_name}_overlay.png"
                )

                cam_start = time.time()

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets,
                )[0]

                runtime = time.time() - cam_start

                grayscale_cam = np.nan_to_num(
                    grayscale_cam,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )

                grayscale_cam = np.clip(
                    grayscale_cam,
                    0.0,
                    1.0,
                )

                overlay = show_cam_on_image(
                    rgb_float,
                    grayscale_cam,
                    use_rgb=True,
                )

                cv2.imwrite(
                    str(heatmap_path),
                    (grayscale_cam * 255).astype(np.uint8),
                )

                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(
                        overlay,
                        cv2.COLOR_RGB2BGR,
                    ),
                )

                status = "success"
                error_message = ""

                method_success[method_name] += 1
                method_runtime[method_name] += runtime

                heatmap_relative = str(
                    heatmap_path.relative_to(root)
                )

                overlay_relative = str(
                    overlay_path.relative_to(root)
                )

            except Exception as error:
                runtime = 0.0
                status = "error"
                error_message = str(error)

                method_errors[method_name] += 1

                heatmap_relative = ""
                overlay_relative = ""

                prediction = ""
                confidence = 0.0
                probabilities = torch.tensor([0.0, 0.0])
                is_correct = False

                print(
                    f"[ERROR] {method_name} failed on "
                    f"{image_path.name}: {error_message}"
                )

            rows.append({
                "experiment_name": experiment_name,
                "model_name": model_name,
                "image_name": image_path.name,
                "image_path": str(
                    image_path.relative_to(root)
                ),
                "mask_path": (
                    str(mask_path.relative_to(root))
                    if mask_path is not None
                    else ""
                ),
                "method": method_name,
                "prediction": prediction,
                "confidence": confidence,
                "prob_edited": float(
                    probabilities[0].item()
                ),
                "prob_real": float(
                    probabilities[1].item()
                ),
                "is_correct": is_correct,
                "status": status,
                "error_message": error_message,
                "runtime_sec": runtime,
                "heatmap_path": heatmap_relative,
                "overlay_path": overlay_relative,
            })

            if (
                image_index % 25 == 0
                or image_index == len(edited_samples)
            ):
                print(
                    f"{method_name}: "
                    f"{image_index}/{len(edited_samples)}"
                )

        # Save progress after every complete method.
        save_rows(csv_path, rows)

        try:
            cam.activations_and_grads.release()
        except Exception:
            pass

        del cam
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(
            f"Heatmap Generation Report V2: "
            f"{experiment_name}\n"
        )

        file.write("=" * 70 + "\n\n")
        file.write(f"Model: {model_name}\n")
        file.write(f"Split: {args.split}\n")
        file.write(
            f"Selected edited images: "
            f"{len(edited_samples)}\n"
        )
        file.write(f"Methods: {methods}\n")
        file.write(
            f"Total elapsed time: "
            f"{total_elapsed:.2f} sec\n\n"
        )

        for method_name in methods:
            success = method_success[method_name]
            errors = method_errors[method_name]
            runtime = method_runtime[method_name]

            average_runtime = (
                runtime / success
                if success > 0
                else 0.0
            )

            file.write(f"[{method_name}]\n")
            file.write(f"Success: {success}\n")
            file.write(f"Errors: {errors}\n")
            file.write(
                f"Average runtime/image: "
                f"{average_runtime:.4f} sec\n\n"
            )

    print("\nHeatmap generation completed.")
    print("=" * 70)
    print(f"Total elapsed time: {total_elapsed:.2f} sec")

    for method_name in methods:
        success = method_success[method_name]
        errors = method_errors[method_name]
        runtime = method_runtime[method_name]

        average_runtime = (
            runtime / success
            if success > 0
            else 0.0
        )

        print(f"\n[{method_name}]")
        print("Success:", success)
        print("Errors :", errors)
        print(
            f"Average runtime/image: "
            f"{average_runtime:.4f} sec"
        )

    print("\nSaved outputs:")
    print("CSV report:", csv_path)
    print("TXT report:", report_path)

    print("\nStep 19 completed successfully.")


if __name__ == "__main__":
    main()