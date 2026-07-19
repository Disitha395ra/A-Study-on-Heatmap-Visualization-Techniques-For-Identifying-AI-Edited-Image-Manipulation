import argparse
import csv
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from datasets.autosplice_dataset import AutoSpliceDataset
from datasets.transforms import get_eval_image_transform, get_mask_transform
from datasets.collate import autosplice_collate_fn
from models.model_factory import create_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    checkpoint_path = (
        root
        / "outputs"
        / "training"
        / experiment_name
        / "checkpoints"
        / "best_model.pth"
    )

    output_dir = root / "outputs" / "evaluation" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "test_predictions.csv"
    report_txt = output_dir / "test_evaluation_report.txt"
    confusion_png = output_dir / "confusion_matrix.png"

    print("=" * 70)
    print(f"Test Evaluation: {experiment_name}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    test_dataset = AutoSpliceDataset(
        project_root=root,
        split="test",
        image_transform=get_eval_image_transform(config["image_size"]),
        mask_transform=get_mask_transform(config["image_size"]),
        image_size=config["image_size"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=autosplice_collate_fn,
    )

    print("\nTest size:", len(test_dataset))

    model, target_layer_name = create_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Loaded model:", checkpoint_path)
    print("Target layer group:", target_layer_name)

    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []
    all_names = []
    all_classes = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            all_paths.extend(batch["image_path"])
            all_names.extend(batch["image_name"])
            all_classes.extend(batch["class_name"])

    class_names = ["edited", "real"]

    acc = accuracy_score(all_labels, all_preds)
    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    edited_probs = [p[0] for p in all_probs]

    try:
        roc_auc = roc_auc_score([1 if y == 0 else 0 for y in all_labels], edited_probs)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(all_labels, all_preds)

    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nTest Results")
    print("=" * 70)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision weighted: {precision_weighted:.4f}")
    print(f"Recall weighted   : {recall_weighted:.4f}")
    print(f"F1 weighted       : {f1_weighted:.4f}")
    print(f"Precision macro   : {precision_macro:.4f}")
    print(f"Recall macro      : {recall_macro:.4f}")
    print(f"F1 macro          : {f1_macro:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC edited    : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print("Rows = true labels, Columns = predicted labels")
    print(cm)

    print("\nClassification Report")
    print(cls_report)

    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "image_name",
            "true_label_index",
            "true_label_name",
            "pred_label_index",
            "pred_label_name",
            "prob_edited",
            "prob_real",
            "is_correct",
        ])

        for path, name, true_label, pred_label, prob in zip(
            all_paths, all_names, all_labels, all_preds, all_probs
        ):
            writer.writerow([
                path,
                name,
                true_label,
                class_names[true_label],
                pred_label,
                class_names[pred_label],
                prob[0],
                prob[1],
                true_label == pred_label,
            ])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {config['model_name']}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(confusion_png, dpi=300)
    plt.close()

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Test Evaluation Report: {experiment_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Target layer group: {target_layer_name}\n")
        f.write(f"Test size: {len(test_dataset)}\n\n")

        f.write("Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"Precision weighted: {precision_weighted:.4f}\n")
        f.write(f"Recall weighted   : {recall_weighted:.4f}\n")
        f.write(f"F1 weighted       : {f1_weighted:.4f}\n")
        f.write(f"Precision macro   : {precision_macro:.4f}\n")
        f.write(f"Recall macro      : {recall_macro:.4f}\n")
        f.write(f"F1 macro          : {f1_macro:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC edited    : {roc_auc:.4f}\n")

        f.write("\nConfusion Matrix\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm))
        f.write("\n\nClassification Report\n")
        f.write("-" * 70 + "\n")
        f.write(cls_report)

    print("\nSaved outputs:")
    print("Predictions CSV :", predictions_csv)
    print("Report TXT      :", report_txt)
    print("Confusion matrix:", confusion_png)

    print("\nStep 15 completed successfully.")


if __name__ == "__main__":
    main()