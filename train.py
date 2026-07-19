import argparse
import csv
import random
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from datasets.autosplice_dataset import AutoSpliceDataset
from datasets.transforms import get_train_image_transform, get_eval_image_transform, get_mask_transform
from datasets.collate import autosplice_collate_fn
from models.model_factory import create_model, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset, device):
    labels = [sample["label"] for sample in dataset.samples]
    counts = [labels.count(0), labels.count(1)]
    total = sum(counts)

    weights = [
        total / (2 * counts[0]),
        total / (2 * counts[1]),
    ]

    return torch.tensor(weights, dtype=torch.float32).to(device), counts


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_training_curves(history, save_path):
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Val Loss")
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train Accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(args.config)

    set_seed(config["seed"])

    experiment_name = config["experiment_name"]

    output_dir = root / "outputs" / "training" / experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_csv = output_dir / "training_log.csv"
    curve_png = output_dir / "training_curves.png"
    report_txt = output_dir / "training_report.txt"
    best_model_path = checkpoint_dir / "best_model.pth"

    print("=" * 70)
    print(f"Training Experiment: {experiment_name}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = AutoSpliceDataset(
        project_root=root,
        split="train",
        image_transform=get_train_image_transform(config["image_size"]),
        mask_transform=get_mask_transform(config["image_size"]),
        image_size=config["image_size"],
    )

    val_dataset = AutoSpliceDataset(
        project_root=root,
        split="val",
        image_transform=get_eval_image_transform(config["image_size"]),
        mask_transform=get_mask_transform(config["image_size"]),
        image_size=config["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=autosplice_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=autosplice_collate_fn,
    )

    print("\nDataset")
    print("Train size:", len(train_dataset))
    print("Val size  :", len(val_dataset))

    model, target_layer_name = create_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
    )

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel")
    print("Model name:", config["model_name"])
    print("Target layer group:", target_layer_name)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    if config["use_class_weights"]:
        class_weights, class_counts = compute_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        class_counts = None
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    print("\nClass counts:", class_counts)
    print("Class weights:", class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    start_time = time.time()

    print("\nStarting training...\n")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        train_true = []
        train_pred = []

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(preds.detach().cpu().numpy())

        train_loss = running_loss / len(train_loader)

        train_metrics = {
            "loss": train_loss,
            "accuracy": accuracy_score(train_true, train_pred),
            "precision": precision_score(train_true, train_pred, average="weighted", zero_division=0),
            "recall": recall_score(train_true, train_pred, average="weighted", zero_division=0),
            "f1": f1_score(train_true, train_pred, average="weighted", zero_division=0),
        }

        val_metrics = evaluate_model(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_time_sec": epoch_time,
        }

        history.append(row)

        print(f"Epoch [{epoch}/{config['epochs']}]")
        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Val   Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )
        print(f"Epoch time: {epoch_time:.2f} sec")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0

            torch.save(
                {
                    "model_name": config["model_name"],
                    "experiment_name": experiment_name,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_layer_name": target_layer_name,
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                    "class_mapping": {"edited": 0, "real": 1},
                },
                best_model_path,
            )

            print("Saved best model:", best_model_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")

        print("-" * 70)

        if patience_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_training_curves(history, curve_png)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Training Report: {experiment_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Target layer group: {target_layer_name}\n")
        f.write(f"Total params: {total_params}\n")
        f.write(f"Trainable params: {trainable_params}\n")
        f.write(f"Train size: {len(train_dataset)}\n")
        f.write(f"Val size: {len(val_dataset)}\n")
        f.write(f"Class counts: {class_counts}\n")
        f.write(f"Class weights: {class_weights}\n")
        f.write(f"Best Val F1: {best_val_f1:.4f}\n")
        f.write(f"Best model path: {best_model_path}\n")
        f.write(f"Total training time: {total_time:.2f} sec\n")

    print("\nTraining completed.")
    print("Best Val F1:", round(best_val_f1, 4))
    print("Saved model :", best_model_path)
    print("Saved log   :", log_csv)
    print("Saved curves:", curve_png)
    print("Saved report:", report_txt)


if __name__ == "__main__":
    main()