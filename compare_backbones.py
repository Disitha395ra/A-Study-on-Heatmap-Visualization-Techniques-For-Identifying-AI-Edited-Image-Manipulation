import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent

MODELS = [
    {
        "model_name": "resnet50",
        "display_name": "ResNet50",
        "experiment_name": "resnet50_autosplice",
    },
    {
        "model_name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "experiment_name": "efficientnet_b0_autosplice",
    },
    {
        "model_name": "convnext_tiny",
        "display_name": "ConvNeXt Tiny",
        "experiment_name": "convnext_tiny_autosplice",
    },
    {
        "model_name": "swin_tiny",
        "display_name": "Swin Tiny",
        "experiment_name": "swin_tiny_autosplice",
    },
    {
        "model_name": "vit_b16",
        "display_name": "ViT-B/16",
        "experiment_name": "vit_b16_autosplice",
    },
]

METHOD_ORDER = [
    "gradcam",
    "gradcamplusplus",
    "layercam",
    "eigencam",
    "scorecam",
    "hirescam",
    "xgradcam",
    "ablationcam",
]

METHOD_DISPLAY_NAMES = {
    "gradcam": "Grad-CAM",
    "gradcamplusplus": "Grad-CAM++",
    "layercam": "LayerCAM",
    "eigencam": "EigenCAM",
    "scorecam": "ScoreCAM",
    "hirescam": "HiResCAM",
    "xgradcam": "XGrad-CAM",
    "ablationcam": "Ablation-CAM",
}

OUTPUT_DIR = ROOT / "outputs" / "cross_backbone_comparison"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ALL_COMBINATIONS_CSV = OUTPUT_DIR / "all_40_combinations.csv"
BEST_BACKBONE_CSV = OUTPUT_DIR / "best_cam_per_backbone.csv"
CLASSIFICATION_CSV = OUTPUT_DIR / "backbone_classification_summary.csv"
REPORT_TXT = OUTPUT_DIR / "cross_backbone_comparison_report.txt"

IOU_HEATMAP = FIGURES_DIR / "backbone_cam_iou_heatmap.png"
DICE_HEATMAP = FIGURES_DIR / "backbone_cam_dice_heatmap.png"
BEST_IOU_BAR = FIGURES_DIR / "best_iou_per_backbone.png"
CLASSIFICATION_BAR = FIGURES_DIR / "classification_accuracy_comparison.png"
RUNTIME_HEATMAP = FIGURES_DIR / "backbone_cam_runtime_heatmap.png"
ACCURACY_VS_IOU = FIGURES_DIR / "classification_accuracy_vs_localization_iou.png"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_classification_metrics(experiment_name):
    predictions_path = (
        ROOT
        / "outputs"
        / "evaluation"
        / experiment_name
        / "test_predictions.csv"
    )

    rows = read_csv(predictions_path)

    true_labels = [int(row["true_label_index"]) for row in rows]
    pred_labels = [int(row["pred_label_index"]) for row in rows]
    edited_probabilities = [float(row["prob_edited"]) for row in rows]

    accuracy = accuracy_score(true_labels, pred_labels)
    weighted_f1 = f1_score(
        true_labels,
        pred_labels,
        average="weighted",
        zero_division=0,
    )
    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        average="macro",
        zero_division=0,
    )

    binary_edited_targets = [
        1 if label == 0 else 0
        for label in true_labels
    ]

    roc_auc = roc_auc_score(
        binary_edited_targets,
        edited_probabilities,
    )

    edited_correct = sum(
        true_label == 0 and pred_label == 0
        for true_label, pred_label in zip(true_labels, pred_labels)
    )

    edited_total = sum(label == 0 for label in true_labels)

    edited_recall = (
        edited_correct / edited_total
        if edited_total > 0
        else 0.0
    )

    return {
        "test_size": len(rows),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "edited_recall": float(edited_recall),
    }


def load_localization_summary(model_info):
    summary_path = (
        ROOT
        / "outputs"
        / "localization"
        / model_info["experiment_name"]
        / "best_method_summary.csv"
    )

    rows = read_csv(summary_path)

    parsed_rows = []

    for row in rows:
        parsed_rows.append({
            "model_name": model_info["model_name"],
            "backbone": model_info["display_name"],
            "experiment_name": model_info["experiment_name"],
            "method": row["method"],
            "method_display": METHOD_DISPLAY_NAMES.get(
                row["method"],
                row["method"],
            ),
            "rank_within_backbone": int(row["rank_by_iou"]),
            "best_threshold": int(float(row["threshold"])),
            "num_images": int(row["num_images"]),
            "mean_iou": float(row["mean_iou"]),
            "std_iou": float(row["std_iou"]),
            "median_iou": float(row["median_iou"]),
            "ci95_iou_low": float(row["ci95_iou_low"]),
            "ci95_iou_high": float(row["ci95_iou_high"]),
            "mean_dice": float(row["mean_dice"]),
            "std_dice": float(row["std_dice"]),
            "median_dice": float(row["median_dice"]),
            "ci95_dice_low": float(row["ci95_dice_low"]),
            "ci95_dice_high": float(row["ci95_dice_high"]),
            "mean_pixel_precision": float(row["mean_pixel_precision"]),
            "mean_pixel_recall": float(row["mean_pixel_recall"]),
            "mean_specificity": float(row["mean_specificity"]),
            "mean_fpr": float(row["mean_fpr"]),
            "mean_pixel_accuracy": float(row["mean_pixel_accuracy"]),
            "mean_runtime_sec": float(row["mean_runtime_sec"]),
        })

    return parsed_rows


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows available for: {path}")

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(rows, metric_name):
    matrix = np.full(
        (len(MODELS), len(METHOD_ORDER)),
        np.nan,
        dtype=np.float64,
    )

    for model_index, model_info in enumerate(MODELS):
        for method_index, method in enumerate(METHOD_ORDER):
            matches = [
                row
                for row in rows
                if row["model_name"] == model_info["model_name"]
                and row["method"] == method
            ]

            if matches:
                matrix[model_index, method_index] = matches[0][metric_name]

    return matrix


def save_matrix_heatmap(matrix, title, value_format, save_path):
    model_labels = [model["display_name"] for model in MODELS]
    method_labels = [
        METHOD_DISPLAY_NAMES[method]
        for method in METHOD_ORDER
    ]

    figure_width = max(12, len(method_labels) * 1.5)

    plt.figure(figsize=(figure_width, 6))
    plt.imshow(matrix, aspect="auto")
    plt.title(title)
    plt.xlabel("CAM method")
    plt.ylabel("Backbone")

    plt.xticks(
        range(len(method_labels)),
        method_labels,
        rotation=35,
        ha="right",
    )
    plt.yticks(
        range(len(model_labels)),
        model_labels,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            if not np.isnan(value):
                plt.text(
                    column_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_bar(labels, values, title, ylabel, save_path):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Backbone")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print("=" * 75)
    print("Cross-Backbone and CAM Comparison")
    print("=" * 75)

    all_localization_rows = []
    classification_rows = []

    for model_info in MODELS:
        print(f"\nLoading: {model_info['display_name']}")

        localization_rows = load_localization_summary(model_info)
        classification_metrics = load_classification_metrics(
            model_info["experiment_name"]
        )

        all_localization_rows.extend(localization_rows)

        classification_rows.append({
            "model_name": model_info["model_name"],
            "backbone": model_info["display_name"],
            "experiment_name": model_info["experiment_name"],
            **classification_metrics,
        })

        print(
            f"Localization combinations: {len(localization_rows)} | "
            f"Classification accuracy: "
            f"{classification_metrics['accuracy']:.4f}"
        )

    if len(all_localization_rows) != 40:
        print(
            f"\n[WARNING] Expected 40 combinations, "
            f"but found {len(all_localization_rows)}."
        )

    all_localization_rows = sorted(
        all_localization_rows,
        key=lambda row: (
            -row["mean_iou"],
            -row["mean_dice"],
        ),
    )

    for overall_rank, row in enumerate(
        all_localization_rows,
        start=1,
    ):
        row["overall_rank_by_iou"] = overall_rank

    best_per_backbone = []

    for model_info in MODELS:
        model_rows = [
            row
            for row in all_localization_rows
            if row["model_name"] == model_info["model_name"]
        ]

        best_row = max(
            model_rows,
            key=lambda row: row["mean_iou"],
        ).copy()

        classification = next(
            row
            for row in classification_rows
            if row["model_name"] == model_info["model_name"]
        )

        best_row.update({
            "classification_accuracy": classification["accuracy"],
            "classification_weighted_f1": classification["weighted_f1"],
            "classification_macro_f1": classification["macro_f1"],
            "classification_roc_auc": classification["roc_auc"],
            "classification_edited_recall": classification["edited_recall"],
        })

        best_per_backbone.append(best_row)

    best_per_backbone = sorted(
        best_per_backbone,
        key=lambda row: row["mean_iou"],
        reverse=True,
    )

    for rank, row in enumerate(best_per_backbone, start=1):
        row["backbone_rank_by_best_iou"] = rank

    save_csv(ALL_COMBINATIONS_CSV, all_localization_rows)
    save_csv(BEST_BACKBONE_CSV, best_per_backbone)
    save_csv(CLASSIFICATION_CSV, classification_rows)

    iou_matrix = build_matrix(
        all_localization_rows,
        "mean_iou",
    )

    dice_matrix = build_matrix(
        all_localization_rows,
        "mean_dice",
    )

    runtime_matrix = build_matrix(
        all_localization_rows,
        "mean_runtime_sec",
    )

    save_matrix_heatmap(
        iou_matrix,
        "Mean IoU Across Backbones and CAM Methods",
        ".3f",
        IOU_HEATMAP,
    )

    save_matrix_heatmap(
        dice_matrix,
        "Mean Dice Score Across Backbones and CAM Methods",
        ".3f",
        DICE_HEATMAP,
    )

    save_matrix_heatmap(
        runtime_matrix,
        "Mean Runtime Across Backbones and CAM Methods",
        ".3f",
        RUNTIME_HEATMAP,
    )

    backbone_labels = [
        row["backbone"]
        for row in best_per_backbone
    ]

    save_bar(
        backbone_labels,
        [row["mean_iou"] for row in best_per_backbone],
        "Best Localization IoU per Backbone",
        "Best Mean IoU",
        BEST_IOU_BAR,
    )

    classification_sorted = sorted(
        classification_rows,
        key=lambda row: row["accuracy"],
        reverse=True,
    )

    save_bar(
        [row["backbone"] for row in classification_sorted],
        [row["accuracy"] for row in classification_sorted],
        "Test Classification Accuracy by Backbone",
        "Accuracy",
        CLASSIFICATION_BAR,
    )

    plt.figure(figsize=(8, 6))

    for row in best_per_backbone:
        plt.scatter(
            row["classification_accuracy"],
            row["mean_iou"],
            s=90,
        )

        plt.annotate(
            f"{row['backbone']}\n{row['method_display']}",
            (
                row["classification_accuracy"],
                row["mean_iou"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title("Classification Accuracy vs Best Localization IoU")
    plt.xlabel("Test classification accuracy")
    plt.ylabel("Best mean IoU")
    plt.tight_layout()
    plt.savefig(ACCURACY_VS_IOU, dpi=300)
    plt.close()

    best_overall = all_localization_rows[0]
    best_classifier = max(
        classification_rows,
        key=lambda row: row["accuracy"],
    )
    fastest_top_method = min(
        best_per_backbone,
        key=lambda row: row["mean_runtime_sec"],
    )

    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("Cross-Backbone CAM Comparison Report\n")
        file.write("=" * 75 + "\n\n")

        file.write(
            f"Total backbone-CAM combinations: "
            f"{len(all_localization_rows)}\n"
        )
        file.write(
            f"Backbones evaluated: {len(MODELS)}\n"
        )
        file.write(
            f"CAM methods per backbone: "
            f"{len(METHOD_ORDER)}\n"
        )
        file.write(
            "Edited test images per combination: 533\n\n"
        )

        file.write("Best CAM per backbone\n")
        file.write("=" * 75 + "\n\n")

        for row in best_per_backbone:
            file.write(
                f"Rank {row['backbone_rank_by_best_iou']}: "
                f"{row['backbone']} + {row['method_display']}\n"
            )
            file.write("-" * 75 + "\n")
            file.write(
                f"Best threshold: {row['best_threshold']}\n"
            )
            file.write(
                f"Mean IoU: {row['mean_iou']:.4f} "
                f"± {row['std_iou']:.4f}\n"
            )
            file.write(
                f"Mean Dice: {row['mean_dice']:.4f} "
                f"± {row['std_dice']:.4f}\n"
            )
            file.write(
                f"Pixel Precision: "
                f"{row['mean_pixel_precision']:.4f}\n"
            )
            file.write(
                f"Pixel Recall: "
                f"{row['mean_pixel_recall']:.4f}\n"
            )
            file.write(
                f"Runtime: "
                f"{row['mean_runtime_sec']:.4f} sec/image\n"
            )
            file.write(
                f"Classification accuracy: "
                f"{row['classification_accuracy']:.4f}\n"
            )
            file.write(
                f"Classification F1: "
                f"{row['classification_weighted_f1']:.4f}\n\n"
            )

        file.write("Major findings\n")
        file.write("=" * 75 + "\n")
        file.write(
            f"Best overall localization combination: "
            f"{best_overall['backbone']} + "
            f"{best_overall['method_display']} "
            f"(IoU={best_overall['mean_iou']:.4f}, "
            f"Dice={best_overall['mean_dice']:.4f}).\n"
        )
        file.write(
            f"Best classification backbone: "
            f"{best_classifier['backbone']} "
            f"(Accuracy={best_classifier['accuracy']:.4f}).\n"
        )
        file.write(
            f"Fastest best-per-backbone CAM combination: "
            f"{fastest_top_method['backbone']} + "
            f"{fastest_top_method['method_display']} "
            f"({fastest_top_method['mean_runtime_sec']:.4f} sec/image).\n"
        )
        file.write(
            "The backbone with the highest classification accuracy "
            "did not produce the strongest localization result, "
            "demonstrating that classification performance and "
            "explanation localization quality should be evaluated "
            "independently.\n"
        )

    print("\nBest CAM per backbone")
    print("=" * 75)

    for row in best_per_backbone:
        print(
            f"Rank {row['backbone_rank_by_best_iou']}: "
            f"{row['backbone']} + {row['method_display']} | "
            f"IoU={row['mean_iou']:.4f} | "
            f"Dice={row['mean_dice']:.4f} | "
            f"Runtime={row['mean_runtime_sec']:.4f}s | "
            f"Cls Acc={row['classification_accuracy']:.4f}"
        )

    print("\nBest overall combination:")
    print(
        f"{best_overall['backbone']} + "
        f"{best_overall['method_display']} | "
        f"IoU={best_overall['mean_iou']:.4f} | "
        f"Dice={best_overall['mean_dice']:.4f}"
    )

    print("\nSaved outputs:")
    print("All combinations :", ALL_COMBINATIONS_CSV)
    print("Best per backbone:", BEST_BACKBONE_CSV)
    print("Classification   :", CLASSIFICATION_CSV)
    print("Report           :", REPORT_TXT)
    print("Figures          :", FIGURES_DIR)

    print("\nStep 20 completed successfully.")


if __name__ == "__main__":
    main()