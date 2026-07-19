import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# PROJECT CONFIGURATION
# =========================================================
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

METHOD_DISPLAY = {
    "gradcam": "Grad-CAM",
    "gradcamplusplus": "Grad-CAM++",
    "layercam": "LayerCAM",
    "eigencam": "EigenCAM",
    "scorecam": "ScoreCAM",
    "hirescam": "HiResCAM",
    "xgradcam": "XGrad-CAM",
    "ablationcam": "Ablation-CAM",
}

OUTPUT_DIR = ROOT / "outputs" / "publication_figures"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


# =========================================================
# FILE PATHS
# =========================================================
CROSS_BACKBONE_DIR = ROOT / "outputs" / "cross_backbone_comparison"

ALL_COMBINATIONS_CSV = (
    CROSS_BACKBONE_DIR / "all_40_combinations.csv"
)

BEST_PER_BACKBONE_CSV = (
    CROSS_BACKBONE_DIR / "best_cam_per_backbone.csv"
)

CLASSIFICATION_CSV = (
    CROSS_BACKBONE_DIR / "backbone_classification_summary.csv"
)

STATISTICAL_DIR = ROOT / "outputs" / "statistical_analysis"

BACKBONE_FRIEDMAN_CSV = (
    STATISTICAL_DIR / "backbone_friedman_tests.csv"
)

BEST_COMBINATION_PAIRWISE_CSV = (
    STATISTICAL_DIR / "best_combination_pairwise_wilcoxon.csv"
)

SUMMARY_REPORT = OUTPUT_DIR / "publication_figures_summary.txt"


# =========================================================
# HELPERS
# =========================================================
def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No data available for: {path}")

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def annotate_bars(ax, values, decimals=3, offset=0.01):
    for index, value in enumerate(values):
        ax.text(
            index,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def get_model_order_index(model_name):
    for index, model in enumerate(MODELS):
        if model["model_name"] == model_name:
            return index
    return 999


def get_method_order_index(method):
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return 999


# =========================================================
# LOAD AND PARSE DATA
# =========================================================
def load_all_data():
    all_combinations_raw = read_csv(ALL_COMBINATIONS_CSV)
    best_per_backbone_raw = read_csv(BEST_PER_BACKBONE_CSV)
    classification_raw = read_csv(CLASSIFICATION_CSV)
    friedman_raw = read_csv(BACKBONE_FRIEDMAN_CSV)
    pairwise_raw = read_csv(BEST_COMBINATION_PAIRWISE_CSV)

    all_combinations = []

    for row in all_combinations_raw:
        all_combinations.append({
            **row,
            "overall_rank_by_iou": int(row["overall_rank_by_iou"]),
            "rank_within_backbone": int(row["rank_within_backbone"]),
            "best_threshold": int(float(row["best_threshold"])),
            "num_images": int(row["num_images"]),
            "mean_iou": float(row["mean_iou"]),
            "std_iou": float(row["std_iou"]),
            "mean_dice": float(row["mean_dice"]),
            "std_dice": float(row["std_dice"]),
            "mean_pixel_precision": float(row["mean_pixel_precision"]),
            "mean_pixel_recall": float(row["mean_pixel_recall"]),
            "mean_runtime_sec": float(row["mean_runtime_sec"]),
        })

    best_per_backbone = []

    for row in best_per_backbone_raw:
        best_per_backbone.append({
            **row,
            "backbone_rank_by_best_iou": int(
                row["backbone_rank_by_best_iou"]
            ),
            "mean_iou": float(row["mean_iou"]),
            "std_iou": float(row["std_iou"]),
            "mean_dice": float(row["mean_dice"]),
            "std_dice": float(row["std_dice"]),
            "mean_pixel_precision": float(row["mean_pixel_precision"]),
            "mean_pixel_recall": float(row["mean_pixel_recall"]),
            "mean_runtime_sec": float(row["mean_runtime_sec"]),
            "classification_accuracy": float(
                row["classification_accuracy"]
            ),
            "classification_weighted_f1": float(
                row["classification_weighted_f1"]
            ),
            "classification_macro_f1": float(
                row["classification_macro_f1"]
            ),
            "classification_roc_auc": float(
                row["classification_roc_auc"]
            ),
        })

    classification = []

    for row in classification_raw:
        classification.append({
            **row,
            "test_size": int(row["test_size"]),
            "accuracy": float(row["accuracy"]),
            "weighted_f1": float(row["weighted_f1"]),
            "macro_f1": float(row["macro_f1"]),
            "roc_auc": float(row["roc_auc"]),
            "edited_recall": float(row["edited_recall"]),
        })

    friedman = []

    for row in friedman_raw:
        friedman.append({
            **row,
            "num_images": int(row["num_images"]),
            "num_methods": int(row["num_methods"]),
            "friedman_chi_square": float(
                row["friedman_chi_square"]
            ),
            "friedman_p_value": float(
                row["friedman_p_value"]
            ),
            "kendalls_w": float(row["kendalls_w"]),
        })

    pairwise = []

    for row in pairwise_raw:
        pairwise.append({
            **row,
            "mean_iou_a": float(row["mean_iou_a"]),
            "mean_iou_b": float(row["mean_iou_b"]),
            "mean_iou_difference_a_minus_b": float(
                row["mean_iou_difference_a_minus_b"]
            ),
            "raw_p_value": float(row["raw_p_value"]),
            "holm_adjusted_p_value": float(
                row["holm_adjusted_p_value"]
            ),
            "rank_biserial_effect_size": float(
                row["rank_biserial_effect_size"]
            ),
        })

    return {
        "all_combinations": all_combinations,
        "best_per_backbone": best_per_backbone,
        "classification": classification,
        "friedman": friedman,
        "pairwise": pairwise,
    }


# =========================================================
# FIGURE 1 — CLASSIFICATION COMPARISON
# =========================================================
def create_classification_comparison(classification):
    rows = sorted(
        classification,
        key=lambda row: get_model_order_index(row["model_name"]),
    )

    labels = [row["backbone"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    weighted_f1 = [row["weighted_f1"] for row in rows]
    roc_auc = [row["roc_auc"] for row in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - width,
        accuracy,
        width,
        label="Accuracy",
    )

    ax.bar(
        x,
        weighted_f1,
        width,
        label="Weighted F1",
    )

    ax.bar(
        x + width,
        roc_auc,
        width,
        label="ROC-AUC",
    )

    ax.set_title(
        "Classification Performance Across Backbone Architectures"
    )
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()

    save_figure(
        FIGURE_DIR / "figure_01_classification_comparison.png"
    )


# =========================================================
# FIGURE 2 — BEST LOCALIZATION PER BACKBONE
# =========================================================
def create_best_localization_comparison(best_per_backbone):
    rows = sorted(
        best_per_backbone,
        key=lambda row: row["mean_iou"],
        reverse=True,
    )

    labels = [
        f"{row['backbone']}\n+ {row['method_display']}"
        for row in rows
    ]

    iou = [row["mean_iou"] for row in rows]
    dice = [row["mean_dice"] for row in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_iou = ax.bar(
        x - width / 2,
        iou,
        width,
        label="Mean IoU",
    )

    bars_dice = ax.bar(
        x + width / 2,
        dice,
        width,
        label="Mean Dice",
    )

    ax.set_title(
        "Best CAM Localization Performance for Each Backbone"
    )
    ax.set_xlabel("Backbone–CAM combination")
    ax.set_ylabel("Localization score")
    ax.set_ylim(0.0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()

    for bars in [bars_iou, bars_dice]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_figure(
        FIGURE_DIR / "figure_02_best_localization_per_backbone.png"
    )


# =========================================================
# FIGURE 3 — IoU HEATMAP
# =========================================================
def create_iou_heatmap(all_combinations):
    matrix = np.zeros(
        (len(MODELS), len(METHOD_ORDER)),
        dtype=np.float64,
    )

    for model_index, model in enumerate(MODELS):
        for method_index, method in enumerate(METHOD_ORDER):
            matches = [
                row
                for row in all_combinations
                if row["model_name"] == model["model_name"]
                and row["method"] == method
            ]

            if not matches:
                matrix[model_index, method_index] = np.nan
            else:
                matrix[model_index, method_index] = matches[0]["mean_iou"]

    plt.figure(figsize=(14, 6))
    plt.imshow(matrix, aspect="auto")
    plt.title("Mean IoU Across 40 Backbone–CAM Combinations")
    plt.xlabel("CAM method")
    plt.ylabel("Backbone")

    plt.xticks(
        range(len(METHOD_ORDER)),
        [METHOD_DISPLAY[method] for method in METHOD_ORDER],
        rotation=35,
        ha="right",
    )

    plt.yticks(
        range(len(MODELS)),
        [model["display_name"] for model in MODELS],
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            if not np.isnan(value):
                plt.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.colorbar(label="Mean IoU")

    save_figure(
        FIGURE_DIR / "figure_03_backbone_cam_iou_heatmap.png"
    )


# =========================================================
# FIGURE 4 — DICE HEATMAP
# =========================================================
def create_dice_heatmap(all_combinations):
    matrix = np.zeros(
        (len(MODELS), len(METHOD_ORDER)),
        dtype=np.float64,
    )

    for model_index, model in enumerate(MODELS):
        for method_index, method in enumerate(METHOD_ORDER):
            matches = [
                row
                for row in all_combinations
                if row["model_name"] == model["model_name"]
                and row["method"] == method
            ]

            if not matches:
                matrix[model_index, method_index] = np.nan
            else:
                matrix[model_index, method_index] = matches[0]["mean_dice"]

    plt.figure(figsize=(14, 6))
    plt.imshow(matrix, aspect="auto")
    plt.title("Mean Dice Score Across 40 Backbone–CAM Combinations")
    plt.xlabel("CAM method")
    plt.ylabel("Backbone")

    plt.xticks(
        range(len(METHOD_ORDER)),
        [METHOD_DISPLAY[method] for method in METHOD_ORDER],
        rotation=35,
        ha="right",
    )

    plt.yticks(
        range(len(MODELS)),
        [model["display_name"] for model in MODELS],
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            if not np.isnan(value):
                plt.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.colorbar(label="Mean Dice")

    save_figure(
        FIGURE_DIR / "figure_04_backbone_cam_dice_heatmap.png"
    )


# =========================================================
# FIGURE 5 — RUNTIME HEATMAP
# =========================================================
def create_runtime_heatmap(all_combinations):
    matrix = np.zeros(
        (len(MODELS), len(METHOD_ORDER)),
        dtype=np.float64,
    )

    for model_index, model in enumerate(MODELS):
        for method_index, method in enumerate(METHOD_ORDER):
            matches = [
                row
                for row in all_combinations
                if row["model_name"] == model["model_name"]
                and row["method"] == method
            ]

            if not matches:
                matrix[model_index, method_index] = np.nan
            else:
                matrix[model_index, method_index] = matches[0][
                    "mean_runtime_sec"
                ]

    plt.figure(figsize=(14, 6))
    plt.imshow(matrix, aspect="auto")
    plt.title("Mean Heatmap Runtime Across Backbones and CAM Methods")
    plt.xlabel("CAM method")
    plt.ylabel("Backbone")

    plt.xticks(
        range(len(METHOD_ORDER)),
        [METHOD_DISPLAY[method] for method in METHOD_ORDER],
        rotation=35,
        ha="right",
    )

    plt.yticks(
        range(len(MODELS)),
        [model["display_name"] for model in MODELS],
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            if not np.isnan(value):
                plt.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.colorbar(label="Seconds per image")

    save_figure(
        FIGURE_DIR / "figure_05_runtime_heatmap.png"
    )


# =========================================================
# FIGURE 6 — ACCURACY VS LOCALIZATION
# =========================================================
def create_accuracy_vs_localization(best_per_backbone):
    plt.figure(figsize=(9, 6))

    for row in best_per_backbone:
        plt.scatter(
            row["classification_accuracy"],
            row["mean_iou"],
            s=100,
        )

        plt.annotate(
            f"{row['backbone']}\n{row['method_display']}",
            (
                row["classification_accuracy"],
                row["mean_iou"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title(
        "Classification Accuracy vs Best Localization Performance"
    )
    plt.xlabel("Test classification accuracy")
    plt.ylabel("Best mean IoU")
    plt.grid(alpha=0.25)

    save_figure(
        FIGURE_DIR / "figure_06_accuracy_vs_localization.png"
    )


# =========================================================
# FIGURE 7 — PRECISION VS RECALL
# =========================================================
def create_precision_recall_scatter(all_combinations):
    plt.figure(figsize=(10, 7))

    for model in MODELS:
        model_rows = [
            row
            for row in all_combinations
            if row["model_name"] == model["model_name"]
        ]

        precision = [
            row["mean_pixel_precision"]
            for row in model_rows
        ]

        recall = [
            row["mean_pixel_recall"]
            for row in model_rows
        ]

        plt.scatter(
            precision,
            recall,
            s=70,
            label=model["display_name"],
        )

        for row in model_rows:
            plt.annotate(
                METHOD_DISPLAY[row["method"]],
                (
                    row["mean_pixel_precision"],
                    row["mean_pixel_recall"],
                ),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )

    plt.title(
        "Pixel Precision–Recall Trade-off Across CAM Methods"
    )
    plt.xlabel("Mean pixel precision")
    plt.ylabel("Mean pixel recall")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure(
        FIGURE_DIR / "figure_07_precision_recall_tradeoff.png"
    )


# =========================================================
# FIGURE 8 — CAM AVERAGE ACROSS BACKBONES
# =========================================================
def create_cam_average_comparison(all_combinations):
    grouped = defaultdict(lambda: {
        "iou": [],
        "dice": [],
        "runtime": [],
    })

    for row in all_combinations:
        grouped[row["method"]]["iou"].append(row["mean_iou"])
        grouped[row["method"]]["dice"].append(row["mean_dice"])
        grouped[row["method"]]["runtime"].append(
            row["mean_runtime_sec"]
        )

    summary = []

    for method in METHOD_ORDER:
        summary.append({
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "mean_iou_across_backbones": float(
                np.mean(grouped[method]["iou"])
            ),
            "std_iou_across_backbones": float(
                np.std(grouped[method]["iou"], ddof=1)
            ),
            "mean_dice_across_backbones": float(
                np.mean(grouped[method]["dice"])
            ),
            "mean_runtime_across_backbones": float(
                np.mean(grouped[method]["runtime"])
            ),
        })

    summary = sorted(
        summary,
        key=lambda row: row["mean_iou_across_backbones"],
        reverse=True,
    )

    labels = [row["method_display"] for row in summary]
    values = [
        row["mean_iou_across_backbones"]
        for row in summary
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Average CAM Localization Performance Across All Backbones"
    )
    ax.set_xlabel("CAM method")
    ax.set_ylabel("Mean IoU across backbones")
    ax.set_xticklabels(labels, rotation=35, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.008,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        FIGURE_DIR / "figure_08_average_cam_performance.png"
    )

    save_csv(
        TABLE_DIR / "average_cam_performance_across_backbones.csv",
        summary,
    )

    return summary


# =========================================================
# FIGURE 9 — KENDALL'S W
# =========================================================
def create_kendalls_w_comparison(friedman):
    rows = sorted(
        friedman,
        key=lambda row: get_model_order_index(row["model_name"]),
    )

    labels = [row["backbone"] for row in rows]
    values = [row["kendalls_w"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Effect Size of CAM Differences Within Each Backbone"
    )
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Kendall's W")
    ax.set_ylim(0.0, max(values) + 0.1)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        FIGURE_DIR / "figure_09_kendalls_w_comparison.png"
    )


# =========================================================
# FIGURE 10 — TOP 10 COMBINATIONS
# =========================================================
def create_top_10_combinations(all_combinations):
    rows = sorted(
        all_combinations,
        key=lambda row: row["mean_iou"],
        reverse=True,
    )[:10]

    labels = [
        f"{row['backbone']} + {row['method_display']}"
        for row in rows
    ]

    values = [row["mean_iou"] for row in rows]

    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(labels, values)

    ax.set_title("Top 10 Backbone–CAM Combinations by Mean IoU")
    ax.set_xlabel("Mean IoU")
    ax.set_ylabel("Combination")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=8,
        )

    save_figure(
        FIGURE_DIR / "figure_10_top_10_combinations.png"
    )


# =========================================================
# PUBLICATION TABLES
# =========================================================
def create_publication_tables(
    classification,
    best_per_backbone,
    all_combinations,
    friedman,
    cam_average_summary,
):
    classification_table = []

    for row in sorted(
        classification,
        key=lambda item: item["accuracy"],
        reverse=True,
    ):
        classification_table.append({
            "Rank": len(classification_table) + 1,
            "Backbone": row["backbone"],
            "Accuracy": f"{row['accuracy']:.4f}",
            "Weighted_F1": f"{row['weighted_f1']:.4f}",
            "Macro_F1": f"{row['macro_f1']:.4f}",
            "ROC_AUC": f"{row['roc_auc']:.4f}",
            "Edited_Recall": f"{row['edited_recall']:.4f}",
        })

    best_table = []

    for row in sorted(
        best_per_backbone,
        key=lambda item: item["mean_iou"],
        reverse=True,
    ):
        best_table.append({
            "Rank": len(best_table) + 1,
            "Backbone": row["backbone"],
            "Best_CAM": row["method_display"],
            "Threshold": row["best_threshold"],
            "Mean_IoU": f"{row['mean_iou']:.4f}",
            "Std_IoU": f"{row['std_iou']:.4f}",
            "Mean_Dice": f"{row['mean_dice']:.4f}",
            "Precision": f"{row['mean_pixel_precision']:.4f}",
            "Recall": f"{row['mean_pixel_recall']:.4f}",
            "Runtime_sec": f"{row['mean_runtime_sec']:.4f}",
            "Classification_Accuracy": (
                f"{row['classification_accuracy']:.4f}"
            ),
        })

    top_10_table = []

    for row in sorted(
        all_combinations,
        key=lambda item: item["mean_iou"],
        reverse=True,
    )[:10]:
        top_10_table.append({
            "Rank": len(top_10_table) + 1,
            "Backbone": row["backbone"],
            "CAM": row["method_display"],
            "Threshold": row["best_threshold"],
            "Mean_IoU": f"{row['mean_iou']:.4f}",
            "Mean_Dice": f"{row['mean_dice']:.4f}",
            "Precision": f"{row['mean_pixel_precision']:.4f}",
            "Recall": f"{row['mean_pixel_recall']:.4f}",
            "Runtime_sec": f"{row['mean_runtime_sec']:.4f}",
        })

    statistical_table = []

    for row in sorted(
        friedman,
        key=lambda item: get_model_order_index(item["model_name"]),
    ):
        statistical_table.append({
            "Backbone": row["backbone"],
            "Friedman_ChiSquare": (
                f"{row['friedman_chi_square']:.4f}"
            ),
            "P_Value": f"{row['friedman_p_value']:.6e}",
            "Kendalls_W": f"{row['kendalls_w']:.4f}",
        })

    save_csv(
        TABLE_DIR / "table_01_classification_results.csv",
        classification_table,
    )

    save_csv(
        TABLE_DIR / "table_02_best_cam_per_backbone.csv",
        best_table,
    )

    save_csv(
        TABLE_DIR / "table_03_top_10_combinations.csv",
        top_10_table,
    )

    save_csv(
        TABLE_DIR / "table_04_friedman_results.csv",
        statistical_table,
    )

    publication_cam_table = []

    for rank, row in enumerate(
        cam_average_summary,
        start=1,
    ):
        publication_cam_table.append({
            "Rank": rank,
            "CAM_Method": row["method_display"],
            "Mean_IoU_Across_Backbones": (
                f"{row['mean_iou_across_backbones']:.4f}"
            ),
            "Std_IoU_Across_Backbones": (
                f"{row['std_iou_across_backbones']:.4f}"
            ),
            "Mean_Dice_Across_Backbones": (
                f"{row['mean_dice_across_backbones']:.4f}"
            ),
            "Mean_Runtime_sec": (
                f"{row['mean_runtime_across_backbones']:.4f}"
            ),
        })

    save_csv(
        TABLE_DIR / "table_05_cam_average_across_backbones.csv",
        publication_cam_table,
    )


# =========================================================
# SUMMARY REPORT
# =========================================================
def create_summary_report(data, cam_average_summary):
    all_combinations = data["all_combinations"]
    best_per_backbone = data["best_per_backbone"]
    classification = data["classification"]
    friedman = data["friedman"]
    pairwise = data["pairwise"]

    best_overall = max(
        all_combinations,
        key=lambda row: row["mean_iou"],
    )

    best_classifier = max(
        classification,
        key=lambda row: row["accuracy"],
    )

    fastest_best = min(
        best_per_backbone,
        key=lambda row: row["mean_runtime_sec"],
    )

    best_cam_average = max(
        cam_average_summary,
        key=lambda row: row["mean_iou_across_backbones"],
    )

    significant_pairs = sum(
        str(row["significant_after_holm"]).lower() == "true"
        for row in pairwise
    )

    with open(SUMMARY_REPORT, "w", encoding="utf-8") as file:
        file.write("Publication Figures and Tables Summary\n")
        file.write("=" * 80 + "\n\n")

        file.write("Experimental scale\n")
        file.write("-" * 80 + "\n")
        file.write("Backbones: 5\n")
        file.write("CAM methods: 8\n")
        file.write("Backbone-CAM combinations: 40\n")
        file.write("Edited test images per combination: 533\n")
        file.write("Total generated heatmaps: 21,320\n\n")

        file.write("Main findings\n")
        file.write("-" * 80 + "\n")

        file.write(
            f"Best localization combination: "
            f"{best_overall['backbone']} + "
            f"{best_overall['method_display']} "
            f"(IoU={best_overall['mean_iou']:.4f}, "
            f"Dice={best_overall['mean_dice']:.4f}).\n"
        )

        file.write(
            f"Best classification backbone: "
            f"{best_classifier['backbone']} "
            f"(Accuracy={best_classifier['accuracy']:.4f}, "
            f"Weighted F1={best_classifier['weighted_f1']:.4f}).\n"
        )

        file.write(
            f"Fastest best-per-backbone combination: "
            f"{fastest_best['backbone']} + "
            f"{fastest_best['method_display']} "
            f"({fastest_best['mean_runtime_sec']:.4f} sec/image).\n"
        )

        file.write(
            f"Best CAM averaged across all backbones: "
            f"{best_cam_average['method_display']} "
            f"(Mean IoU="
            f"{best_cam_average['mean_iou_across_backbones']:.4f}).\n"
        )

        file.write(
            f"Significant pairwise comparisons among best "
            f"backbone-CAM combinations: "
            f"{significant_pairs}/{len(pairwise)}.\n\n"
        )

        file.write("Generated figures\n")
        file.write("-" * 80 + "\n")

        for figure in sorted(FIGURE_DIR.glob("*.png")):
            file.write(f"{figure.name}\n")

        file.write("\nGenerated tables\n")
        file.write("-" * 80 + "\n")

        for table in sorted(TABLE_DIR.glob("*.csv")):
            file.write(f"{table.name}\n")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("Publication-Quality Figures Generator")
    print("=" * 80)

    data = load_all_data()

    print("\nCreating classification comparison...")
    create_classification_comparison(
        data["classification"]
    )

    print("Creating best localization comparison...")
    create_best_localization_comparison(
        data["best_per_backbone"]
    )

    print("Creating IoU heatmap...")
    create_iou_heatmap(
        data["all_combinations"]
    )

    print("Creating Dice heatmap...")
    create_dice_heatmap(
        data["all_combinations"]
    )

    print("Creating runtime heatmap...")
    create_runtime_heatmap(
        data["all_combinations"]
    )

    print("Creating accuracy-vs-localization figure...")
    create_accuracy_vs_localization(
        data["best_per_backbone"]
    )

    print("Creating precision-recall trade-off figure...")
    create_precision_recall_scatter(
        data["all_combinations"]
    )

    print("Creating average CAM comparison...")
    cam_average_summary = create_cam_average_comparison(
        data["all_combinations"]
    )

    print("Creating Kendall's W comparison...")
    create_kendalls_w_comparison(
        data["friedman"]
    )

    print("Creating top-10 combinations chart...")
    create_top_10_combinations(
        data["all_combinations"]
    )

    print("Creating publication tables...")
    create_publication_tables(
        classification=data["classification"],
        best_per_backbone=data["best_per_backbone"],
        all_combinations=data["all_combinations"],
        friedman=data["friedman"],
        cam_average_summary=cam_average_summary,
    )

    print("Creating summary report...")
    create_summary_report(
        data,
        cam_average_summary,
    )

    print("\nGenerated figures:")
    for figure in sorted(FIGURE_DIR.glob("*.png")):
        print(" -", figure.name)

    print("\nGenerated tables:")
    for table in sorted(TABLE_DIR.glob("*.csv")):
        print(" -", table.name)

    print("\nSummary report:")
    print(SUMMARY_REPORT)

    print("\nStep 22 completed successfully.")


if __name__ == "__main__":
    main()