import csv
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon


# =========================================================
# CONFIGURATION
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

ALPHA = 0.05

OUTPUT_DIR = ROOT / "outputs" / "statistical_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_FRIEDMAN_CSV = OUTPUT_DIR / "backbone_friedman_tests.csv"
BACKBONE_PAIRWISE_CSV = OUTPUT_DIR / "backbone_pairwise_wilcoxon.csv"
BEST_COMBINATION_FRIEDMAN_CSV = OUTPUT_DIR / "best_combination_friedman_test.csv"
BEST_COMBINATION_PAIRWISE_CSV = OUTPUT_DIR / "best_combination_pairwise_wilcoxon.csv"
REPORT_TXT = OUTPUT_DIR / "statistical_analysis_report.txt"

PAIRWISE_HEATMAP_DIR = FIGURES_DIR / "pairwise_pvalue_heatmaps"
PAIRWISE_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

BEST_COMBINATION_PVALUE_HEATMAP = (
    FIGURES_DIR / "best_combination_adjusted_pvalue_heatmap.png"
)

BEST_COMBINATION_IOU_BOXPLOT = (
    FIGURES_DIR / "best_combination_iou_boxplot.png"
)


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
        print(f"[WARNING] No rows to save: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def holm_correction(p_values):
    """
    Holm-Bonferroni correction.

    Returns adjusted p-values in the original order.
    """
    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return []

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * number_of_tests
    running_maximum = 0.0

    for sorted_position, (original_index, p_value) in enumerate(indexed):
        multiplier = number_of_tests - sorted_position
        corrected = min(1.0, multiplier * p_value)

        running_maximum = max(
            running_maximum,
            corrected,
        )

        adjusted[original_index] = running_maximum

    return adjusted


def rank_biserial_effect_size(values_a, values_b):
    """
    Paired rank-biserial correlation.

    Positive value:
        values_a tend to be larger than values_b.

    Negative value:
        values_b tend to be larger than values_a.
    """
    differences = np.asarray(values_a) - np.asarray(values_b)

    differences = differences[differences != 0]

    if len(differences) == 0:
        return 0.0

    absolute_differences = np.abs(differences)
    ranks = rank_absolute_values(absolute_differences)

    positive_rank_sum = float(
        ranks[differences > 0].sum()
    )

    negative_rank_sum = float(
        ranks[differences < 0].sum()
    )

    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / denominator


def rank_absolute_values(values):
    """
    Rank values using average ranks for ties.
    """
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=np.float64)

    index = 0

    while index < len(values):
        end = index

        while (
            end + 1 < len(values)
            and values[order[end + 1]]
            == values[order[index]]
        ):
            end += 1

        average_rank = (
            index + 1 + end + 1
        ) / 2.0

        for position in range(index, end + 1):
            ranks[order[position]] = average_rank

        index = end + 1

    return ranks


def interpret_effect_size(effect_size):
    magnitude = abs(effect_size)

    if magnitude < 0.10:
        return "negligible"
    if magnitude < 0.30:
        return "small"
    if magnitude < 0.50:
        return "medium"
    return "large"


def safe_wilcoxon(values_a, values_b):
    """
    Perform paired Wilcoxon signed-rank test safely.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)

    differences = values_a - values_b

    if np.allclose(differences, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        values_a,
        values_b,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def save_pvalue_heatmap(
    labels,
    pairwise_rows,
    adjusted_p_field,
    title,
    save_path,
):
    size = len(labels)

    matrix = np.ones(
        (size, size),
        dtype=np.float64,
    )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    for row in pairwise_rows:
        label_a = row["display_a"]
        label_b = row["display_b"]

        if (
            label_a not in label_to_index
            or label_b not in label_to_index
        ):
            continue

        index_a = label_to_index[label_a]
        index_b = label_to_index[label_b]

        value = float(row[adjusted_p_field])

        matrix[index_a, index_b] = value
        matrix[index_b, index_a] = value

    # Transform p-values for clearer visualization.
    transformed = -np.log10(
        np.clip(matrix, 1e-12, 1.0)
    )

    plt.figure(figsize=(9, 7))
    plt.imshow(transformed, aspect="auto")
    plt.title(title)
    plt.xlabel("Method / combination")
    plt.ylabel("Method / combination")

    plt.xticks(
        range(size),
        labels,
        rotation=40,
        ha="right",
    )
    plt.yticks(
        range(size),
        labels,
    )

    for row_index in range(size):
        for column_index in range(size):
            if row_index == column_index:
                text = "-"
            else:
                text = f"{matrix[row_index, column_index]:.3f}"

            plt.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.colorbar(label="-log10(adjusted p-value)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_backbone_iou_vectors(model_info):
    """
    Load each method's per-image IoU vector at that method's
    selected best threshold.
    """
    localization_dir = (
        ROOT
        / "outputs"
        / "localization"
        / model_info["experiment_name"]
    )

    per_image_path = (
        localization_dir
        / "per_image_localization_results.csv"
    )

    best_summary_path = (
        localization_dir
        / "best_method_summary.csv"
    )

    per_image_rows = read_csv(per_image_path)
    best_rows = read_csv(best_summary_path)

    best_thresholds = {
        row["method"]: int(float(row["threshold"]))
        for row in best_rows
    }

    vectors = {}

    for method in METHOD_ORDER:
        if method not in best_thresholds:
            raise ValueError(
                f"Missing best threshold for "
                f"{model_info['display_name']} / {method}"
            )

        threshold = best_thresholds[method]

        method_rows = [
            row
            for row in per_image_rows
            if row["method"] == method
            and int(float(row["threshold"])) == threshold
        ]

        method_rows = sorted(
            method_rows,
            key=lambda row: row["image_name"],
        )

        vectors[method] = {
            "threshold": threshold,
            "image_names": [
                row["image_name"]
                for row in method_rows
            ],
            "iou": np.asarray(
                [
                    float(row["iou"])
                    for row in method_rows
                ],
                dtype=np.float64,
            ),
        }

    reference_names = vectors[METHOD_ORDER[0]]["image_names"]

    for method in METHOD_ORDER[1:]:
        if vectors[method]["image_names"] != reference_names:
            raise RuntimeError(
                f"Image alignment mismatch for "
                f"{model_info['display_name']} / {method}"
            )

    return vectors


# =========================================================
# WITHIN-BACKBONE STATISTICS
# =========================================================
def analyze_methods_within_backbone(model_info, vectors):
    method_arrays = [
        vectors[method]["iou"]
        for method in METHOD_ORDER
    ]

    friedman_result = friedmanchisquare(
        *method_arrays
    )

    number_of_images = len(method_arrays[0])
    number_of_methods = len(method_arrays)

    kendalls_w = (
        friedman_result.statistic
        / (
            number_of_images
            * (number_of_methods - 1)
        )
    )

    friedman_row = {
        "model_name": model_info["model_name"],
        "backbone": model_info["display_name"],
        "num_images": number_of_images,
        "num_methods": number_of_methods,
        "friedman_chi_square": float(
            friedman_result.statistic
        ),
        "friedman_p_value": float(
            friedman_result.pvalue
        ),
        "kendalls_w": float(kendalls_w),
        "significant_at_0_05": (
            friedman_result.pvalue < ALPHA
        ),
    }

    pairwise_rows = []
    raw_p_values = []

    for method_a, method_b in combinations(
        METHOD_ORDER,
        2,
    ):
        values_a = vectors[method_a]["iou"]
        values_b = vectors[method_b]["iou"]

        statistic, p_value = safe_wilcoxon(
            values_a,
            values_b,
        )

        effect_size = rank_biserial_effect_size(
            values_a,
            values_b,
        )

        row = {
            "model_name": model_info["model_name"],
            "backbone": model_info["display_name"],
            "method_a": method_a,
            "display_a": METHOD_DISPLAY[method_a],
            "threshold_a": vectors[method_a]["threshold"],
            "mean_iou_a": float(np.mean(values_a)),
            "method_b": method_b,
            "display_b": METHOD_DISPLAY[method_b],
            "threshold_b": vectors[method_b]["threshold"],
            "mean_iou_b": float(np.mean(values_b)),
            "mean_iou_difference_a_minus_b": float(
                np.mean(values_a - values_b)
            ),
            "wilcoxon_statistic": statistic,
            "raw_p_value": p_value,
            "rank_biserial_effect_size": effect_size,
            "effect_size_interpretation": (
                interpret_effect_size(effect_size)
            ),
        }

        pairwise_rows.append(row)
        raw_p_values.append(p_value)

    adjusted_p_values = holm_correction(
        raw_p_values
    )

    for row, adjusted_p in zip(
        pairwise_rows,
        adjusted_p_values,
    ):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_after_holm"] = (
            adjusted_p < ALPHA
        )

    heatmap_path = (
        PAIRWISE_HEATMAP_DIR
        / f"{model_info['model_name']}_pairwise_pvalues.png"
    )

    save_pvalue_heatmap(
        labels=[
            METHOD_DISPLAY[method]
            for method in METHOD_ORDER
        ],
        pairwise_rows=pairwise_rows,
        adjusted_p_field="holm_adjusted_p_value",
        title=(
            f"Adjusted Pairwise P-values — "
            f"{model_info['display_name']}"
        ),
        save_path=heatmap_path,
    )

    return friedman_row, pairwise_rows


# =========================================================
# BEST COMBINATION ACROSS BACKBONES
# =========================================================
def analyze_best_combinations(
    all_vectors,
    best_combination_rows,
):
    labels = []
    arrays = []
    metadata = []

    for combination in best_combination_rows:
        model_name = combination["model_name"]
        method = combination["method"]

        model_info = next(
            model
            for model in MODELS
            if model["model_name"] == model_name
        )

        vector = all_vectors[model_name][method]

        labels.append(
            f"{model_info['display_name']} + "
            f"{METHOD_DISPLAY[method]}"
        )

        arrays.append(vector["iou"])

        metadata.append({
            "model_name": model_name,
            "backbone": model_info["display_name"],
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "threshold": vector["threshold"],
        })

    result = friedmanchisquare(*arrays)

    number_of_images = len(arrays[0])
    number_of_combinations = len(arrays)

    kendalls_w = (
        result.statistic
        / (
            number_of_images
            * (number_of_combinations - 1)
        )
    )

    friedman_row = {
        "comparison": "best_cam_per_backbone",
        "num_images": number_of_images,
        "num_combinations": number_of_combinations,
        "friedman_chi_square": float(result.statistic),
        "friedman_p_value": float(result.pvalue),
        "kendalls_w": float(kendalls_w),
        "significant_at_0_05": (
            result.pvalue < ALPHA
        ),
    }

    pairwise_rows = []
    raw_p_values = []

    for index_a, index_b in combinations(
        range(len(arrays)),
        2,
    ):
        values_a = arrays[index_a]
        values_b = arrays[index_b]

        statistic, p_value = safe_wilcoxon(
            values_a,
            values_b,
        )

        effect_size = rank_biserial_effect_size(
            values_a,
            values_b,
        )

        metadata_a = metadata[index_a]
        metadata_b = metadata[index_b]

        row = {
            "model_a": metadata_a["model_name"],
            "backbone_a": metadata_a["backbone"],
            "method_a": metadata_a["method"],
            "display_a": labels[index_a],
            "threshold_a": metadata_a["threshold"],
            "mean_iou_a": float(np.mean(values_a)),

            "model_b": metadata_b["model_name"],
            "backbone_b": metadata_b["backbone"],
            "method_b": metadata_b["method"],
            "display_b": labels[index_b],
            "threshold_b": metadata_b["threshold"],
            "mean_iou_b": float(np.mean(values_b)),

            "mean_iou_difference_a_minus_b": float(
                np.mean(values_a - values_b)
            ),
            "wilcoxon_statistic": statistic,
            "raw_p_value": p_value,
            "rank_biserial_effect_size": effect_size,
            "effect_size_interpretation": (
                interpret_effect_size(effect_size)
            ),
        }

        pairwise_rows.append(row)
        raw_p_values.append(p_value)

    adjusted_p_values = holm_correction(
        raw_p_values
    )

    for row, adjusted_p in zip(
        pairwise_rows,
        adjusted_p_values,
    ):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_after_holm"] = (
            adjusted_p < ALPHA
        )

    save_pvalue_heatmap(
        labels=labels,
        pairwise_rows=pairwise_rows,
        adjusted_p_field="holm_adjusted_p_value",
        title=(
            "Adjusted P-values for Best "
            "Backbone–CAM Combinations"
        ),
        save_path=BEST_COMBINATION_PVALUE_HEATMAP,
    )

    plt.figure(figsize=(11, 6))
    plt.boxplot(
        arrays,
        tick_labels=labels,
        showmeans=True,
    )
    plt.title(
        "Per-image IoU Distribution of "
        "Best Backbone–CAM Combinations"
    )
    plt.xlabel("Backbone–CAM combination")
    plt.ylabel("IoU")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        BEST_COMBINATION_IOU_BOXPLOT,
        dpi=300,
    )
    plt.close()

    return friedman_row, pairwise_rows


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("Statistical Significance Analysis")
    print("=" * 80)

    all_vectors = {}

    backbone_friedman_rows = []
    backbone_pairwise_rows = []

    for model_info in MODELS:
        print(
            f"\nAnalyzing CAM methods for "
            f"{model_info['display_name']}..."
        )

        vectors = load_backbone_iou_vectors(
            model_info
        )

        all_vectors[
            model_info["model_name"]
        ] = vectors

        friedman_row, pairwise_rows = (
            analyze_methods_within_backbone(
                model_info,
                vectors,
            )
        )

        backbone_friedman_rows.append(
            friedman_row
        )

        backbone_pairwise_rows.extend(
            pairwise_rows
        )

        significant_pairs = sum(
            row["significant_after_holm"]
            for row in pairwise_rows
        )

        print(
            f"Friedman χ²="
            f"{friedman_row['friedman_chi_square']:.4f}, "
            f"p={friedman_row['friedman_p_value']:.6g}, "
            f"Kendall's W="
            f"{friedman_row['kendalls_w']:.4f}"
        )

        print(
            f"Significant pairwise comparisons "
            f"after Holm correction: "
            f"{significant_pairs}/{len(pairwise_rows)}"
        )

    best_combination_path = (
        ROOT
        / "outputs"
        / "cross_backbone_comparison"
        / "best_cam_per_backbone.csv"
    )

    best_combination_rows = read_csv(
        best_combination_path
    )

    best_friedman_row, best_pairwise_rows = (
        analyze_best_combinations(
            all_vectors,
            best_combination_rows,
        )
    )

    save_csv(
        BACKBONE_FRIEDMAN_CSV,
        backbone_friedman_rows,
    )

    save_csv(
        BACKBONE_PAIRWISE_CSV,
        backbone_pairwise_rows,
    )

    save_csv(
        BEST_COMBINATION_FRIEDMAN_CSV,
        [best_friedman_row],
    )

    save_csv(
        BEST_COMBINATION_PAIRWISE_CSV,
        best_pairwise_rows,
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Statistical Significance Analysis Report\n"
        )
        file.write("=" * 80 + "\n\n")

        file.write(
            f"Significance level: α={ALPHA}\n"
        )

        file.write(
            "Multiple comparison correction: "
            "Holm-Bonferroni\n"
        )

        file.write(
            "Effect size: paired rank-biserial "
            "correlation\n\n"
        )

        file.write(
            "Within-backbone CAM comparisons\n"
        )
        file.write("=" * 80 + "\n\n")

        for friedman_row in backbone_friedman_rows:
            file.write(
                f"{friedman_row['backbone']}\n"
            )
            file.write("-" * 80 + "\n")

            file.write(
                f"Friedman chi-square: "
                f"{friedman_row['friedman_chi_square']:.6f}\n"
            )

            file.write(
                f"P-value: "
                f"{friedman_row['friedman_p_value']:.10g}\n"
            )

            file.write(
                f"Kendall's W: "
                f"{friedman_row['kendalls_w']:.6f}\n"
            )

            file.write(
                f"Statistically significant: "
                f"{friedman_row['significant_at_0_05']}\n\n"
            )

            model_pairs = [
                row
                for row in backbone_pairwise_rows
                if row["model_name"]
                == friedman_row["model_name"]
            ]

            significant_model_pairs = [
                row
                for row in model_pairs
                if row["significant_after_holm"]
            ]

            file.write(
                f"Significant pairwise comparisons "
                f"after Holm correction: "
                f"{len(significant_model_pairs)}"
                f"/{len(model_pairs)}\n\n"
            )

        file.write(
            "Best backbone–CAM combination comparison\n"
        )
        file.write("=" * 80 + "\n\n")

        file.write(
            f"Friedman chi-square: "
            f"{best_friedman_row['friedman_chi_square']:.6f}\n"
        )

        file.write(
            f"P-value: "
            f"{best_friedman_row['friedman_p_value']:.10g}\n"
        )

        file.write(
            f"Kendall's W: "
            f"{best_friedman_row['kendalls_w']:.6f}\n"
        )

        file.write(
            f"Statistically significant: "
            f"{best_friedman_row['significant_at_0_05']}\n\n"
        )

        file.write(
            "Pairwise best-combination comparisons\n"
        )
        file.write("-" * 80 + "\n\n")

        for row in sorted(
            best_pairwise_rows,
            key=lambda item: float(
                item["holm_adjusted_p_value"]
            ),
        ):
            file.write(
                f"{row['display_a']} vs "
                f"{row['display_b']}\n"
            )

            file.write(
                f"Mean IoU difference: "
                f"{row['mean_iou_difference_a_minus_b']:.6f}\n"
            )

            file.write(
                f"Adjusted p-value: "
                f"{row['holm_adjusted_p_value']:.10g}\n"
            )

            file.write(
                f"Effect size: "
                f"{row['rank_biserial_effect_size']:.6f} "
                f"({row['effect_size_interpretation']})\n"
            )

            file.write(
                f"Significant: "
                f"{row['significant_after_holm']}\n\n"
            )

    print("\nBest-combination Friedman test")
    print("=" * 80)

    print(
        f"χ²="
        f"{best_friedman_row['friedman_chi_square']:.4f}"
    )

    print(
        f"p="
        f"{best_friedman_row['friedman_p_value']:.8g}"
    )

    print(
        f"Kendall's W="
        f"{best_friedman_row['kendalls_w']:.4f}"
    )

    significant_best_pairs = sum(
        row["significant_after_holm"]
        for row in best_pairwise_rows
    )

    print(
        f"Significant best-combination pairs: "
        f"{significant_best_pairs}/"
        f"{len(best_pairwise_rows)}"
    )

    print("\nSaved outputs:")
    print(
        "Backbone Friedman CSV :",
        BACKBONE_FRIEDMAN_CSV,
    )
    print(
        "Backbone pairwise CSV :",
        BACKBONE_PAIRWISE_CSV,
    )
    print(
        "Best Friedman CSV     :",
        BEST_COMBINATION_FRIEDMAN_CSV,
    )
    print(
        "Best pairwise CSV     :",
        BEST_COMBINATION_PAIRWISE_CSV,
    )
    print(
        "Report                :",
        REPORT_TXT,
    )
    print(
        "Figures               :",
        FIGURES_DIR,
    )

    print(
        "\nStep 21 completed successfully."
    )


if __name__ == "__main__":
    main()