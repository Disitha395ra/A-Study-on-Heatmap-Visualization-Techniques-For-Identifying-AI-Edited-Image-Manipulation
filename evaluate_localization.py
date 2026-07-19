import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from metrics.localization_metrics import evaluate_heatmap_against_mask


THRESHOLDS = [20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 220]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def stats(values):
    arr = np.array(values, dtype=np.float64)

    if len(arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median = float(np.median(arr))

    if len(arr) > 1:
        margin = 1.96 * std / math.sqrt(len(arr))
    else:
        margin = 0.0

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def save_bar(labels, values, title, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("CAM method")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_boxplot(labels, values_by_method, title, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    plt.boxplot(values_by_method, tick_labels=labels, showmeans=True)
    plt.title(title)
    plt.xlabel("CAM method")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--heatmap-csv", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    if args.heatmap_csv is None:
        heatmap_csv = root / "outputs" / "heatmaps" / experiment_name / "heatmap_generation.csv"
    else:
        heatmap_csv = Path(args.heatmap_csv)

    output_dir = root / "outputs" / "localization" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = output_dir / "per_image_localization_results.csv"
    threshold_summary_csv = output_dir / "threshold_summary.csv"
    best_method_summary_csv = output_dir / "best_method_summary.csv"
    report_txt = output_dir / "localization_report.txt"

    iou_bar = figures_dir / "mean_iou_bar.png"
    dice_bar = figures_dir / "mean_dice_bar.png"
    precision_bar = figures_dir / "mean_precision_bar.png"
    recall_bar = figures_dir / "mean_recall_bar.png"
    runtime_bar = figures_dir / "runtime_bar.png"
    iou_boxplot = figures_dir / "iou_boxplot.png"
    dice_boxplot = figures_dir / "dice_boxplot.png"
    threshold_iou_line = figures_dir / "threshold_vs_iou.png"

    print("=" * 70)
    print(f"Localization Evaluation: {experiment_name}")
    print("=" * 70)

    print("Heatmap CSV:", heatmap_csv)

    with open(heatmap_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["status"] == "success"]

    print("Successful heatmap records:", len(rows))

    if len(rows) == 0:
        raise RuntimeError("No successful heatmap records found.")

    per_image_results = []
    grouped = defaultdict(lambda: {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "fpr": [],
        "pixel_accuracy": [],
        "runtime": [],
    })

    for row in rows:
        method = row["method"]

        heatmap_path = root / row["heatmap_path"]
        mask_path = root / row["mask_path"]

        heatmap = read_gray(heatmap_path)
        gt_mask = read_gray(mask_path)

        for threshold in THRESHOLDS:
            metrics, _, _ = evaluate_heatmap_against_mask(
                heatmap=heatmap,
                gt_mask=gt_mask,
                threshold=threshold,
            )

            runtime = float(row["runtime_sec"])

            result = {
                "experiment_name": experiment_name,
                "model_name": row["model_name"],
                "image_name": row["image_name"],
                "method": method,
                "threshold": threshold,
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "is_correct_classification": row["is_correct"],
                "runtime_sec": runtime,
                "heatmap_path": row["heatmap_path"],
                "mask_path": row["mask_path"],
                **metrics,
            }

            per_image_results.append(result)

            key = (method, threshold)
            grouped[key]["iou"].append(metrics["iou"])
            grouped[key]["dice"].append(metrics["dice"])
            grouped[key]["precision"].append(metrics["pixel_precision"])
            grouped[key]["recall"].append(metrics["pixel_recall"])
            grouped[key]["specificity"].append(metrics["specificity"])
            grouped[key]["fpr"].append(metrics["fpr"])
            grouped[key]["pixel_accuracy"].append(metrics["pixel_accuracy"])
            grouped[key]["runtime"].append(runtime)

    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_image_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_results)

    threshold_summary = []

    for (method, threshold), values in grouped.items():
        iou_s = stats(values["iou"])
        dice_s = stats(values["dice"])
        precision_s = stats(values["precision"])
        recall_s = stats(values["recall"])
        specificity_s = stats(values["specificity"])
        fpr_s = stats(values["fpr"])
        pixel_acc_s = stats(values["pixel_accuracy"])
        runtime_s = stats(values["runtime"])

        threshold_summary.append({
            "experiment_name": experiment_name,
            "model_name": config["model_name"],
            "method": method,
            "threshold": threshold,
            "num_images": len(values["iou"]),

            "mean_iou": iou_s["mean"],
            "std_iou": iou_s["std"],
            "median_iou": iou_s["median"],
            "ci95_iou_low": iou_s["ci95_low"],
            "ci95_iou_high": iou_s["ci95_high"],

            "mean_dice": dice_s["mean"],
            "std_dice": dice_s["std"],
            "median_dice": dice_s["median"],
            "ci95_dice_low": dice_s["ci95_low"],
            "ci95_dice_high": dice_s["ci95_high"],

            "mean_pixel_precision": precision_s["mean"],
            "std_pixel_precision": precision_s["std"],
            "median_pixel_precision": precision_s["median"],

            "mean_pixel_recall": recall_s["mean"],
            "std_pixel_recall": recall_s["std"],
            "median_pixel_recall": recall_s["median"],

            "mean_specificity": specificity_s["mean"],
            "mean_fpr": fpr_s["mean"],
            "mean_pixel_accuracy": pixel_acc_s["mean"],

            "mean_runtime_sec": runtime_s["mean"],
            "std_runtime_sec": runtime_s["std"],
            "median_runtime_sec": runtime_s["median"],
        })

    with open(threshold_summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(threshold_summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(threshold_summary)

    best_by_method = {}

    for row in threshold_summary:
        method = row["method"]
        if method not in best_by_method or row["mean_iou"] > best_by_method[method]["mean_iou"]:
            best_by_method[method] = row

    best_rows = list(best_by_method.values())
    best_rows = sorted(best_rows, key=lambda r: r["mean_iou"], reverse=True)

    for rank, row in enumerate(best_rows, start=1):
        row["rank_by_iou"] = rank

    with open(best_method_summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank_by_iou"] + [k for k in best_rows[0].keys() if k != "rank_by_iou"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    labels = [r["method"] for r in best_rows]

    save_bar(labels, [r["mean_iou"] for r in best_rows], "Mean IoU by CAM Method", "Mean IoU", iou_bar)
    save_bar(labels, [r["mean_dice"] for r in best_rows], "Mean Dice by CAM Method", "Mean Dice", dice_bar)
    save_bar(labels, [r["mean_pixel_precision"] for r in best_rows], "Mean Pixel Precision by CAM Method", "Precision", precision_bar)
    save_bar(labels, [r["mean_pixel_recall"] for r in best_rows], "Mean Pixel Recall by CAM Method", "Recall", recall_bar)
    save_bar(labels, [r["mean_runtime_sec"] for r in best_rows], "Runtime by CAM Method", "Seconds/image", runtime_bar)

    iou_values_by_method = []
    dice_values_by_method = []

    for row in best_rows:
        method = row["method"]
        threshold = int(row["threshold"])

        iou_vals = [
            float(x["iou"])
            for x in per_image_results
            if x["method"] == method and int(x["threshold"]) == threshold
        ]

        dice_vals = [
            float(x["dice"])
            for x in per_image_results
            if x["method"] == method and int(x["threshold"]) == threshold
        ]

        iou_values_by_method.append(iou_vals)
        dice_values_by_method.append(dice_vals)

    save_boxplot(labels, iou_values_by_method, "IoU Distribution by CAM Method", "IoU", iou_boxplot)
    save_boxplot(labels, dice_values_by_method, "Dice Distribution by CAM Method", "Dice", dice_boxplot)

    plt.figure(figsize=(10, 6))

    for method in sorted(set(r["method"] for r in threshold_summary)):
        method_rows = sorted(
            [r for r in threshold_summary if r["method"] == method],
            key=lambda r: int(r["threshold"])
        )

        plt.plot(
            [r["threshold"] for r in method_rows],
            [r["mean_iou"] for r in method_rows],
            marker="o",
            label=method,
        )

    plt.title("Threshold Sensitivity: Mean IoU")
    plt.xlabel("Heatmap threshold")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(threshold_iou_line, dpi=300)
    plt.close()

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Localization Evaluation Report: {experiment_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Heatmap CSV: {heatmap_csv}\n")
        f.write(f"Successful heatmaps: {len(rows)}\n")
        f.write(f"Thresholds tested: {THRESHOLDS}\n\n")

        f.write("Best CAM methods ranked by Mean IoU\n")
        f.write("=" * 70 + "\n\n")

        for row in best_rows:
            f.write(f"Rank {row['rank_by_iou']}: {row['method']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best threshold: {row['threshold']}\n")
            f.write(f"Mean IoU: {row['mean_iou']:.4f} ± {row['std_iou']:.4f}\n")
            f.write(f"95% CI IoU: [{row['ci95_iou_low']:.4f}, {row['ci95_iou_high']:.4f}]\n")
            f.write(f"Median IoU: {row['median_iou']:.4f}\n")
            f.write(f"Mean Dice: {row['mean_dice']:.4f} ± {row['std_dice']:.4f}\n")
            f.write(f"95% CI Dice: [{row['ci95_dice_low']:.4f}, {row['ci95_dice_high']:.4f}]\n")
            f.write(f"Median Dice: {row['median_dice']:.4f}\n")
            f.write(f"Mean Pixel Precision: {row['mean_pixel_precision']:.4f}\n")
            f.write(f"Mean Pixel Recall: {row['mean_pixel_recall']:.4f}\n")
            f.write(f"Mean Specificity: {row['mean_specificity']:.4f}\n")
            f.write(f"Mean FPR: {row['mean_fpr']:.4f}\n")
            f.write(f"Mean Pixel Accuracy: {row['mean_pixel_accuracy']:.4f}\n")
            f.write(f"Mean Runtime: {row['mean_runtime_sec']:.4f} sec/image\n\n")

    print("\nBest CAM methods ranked by Mean IoU")
    print("=" * 70)

    for row in best_rows:
        print(
            f"Rank {row['rank_by_iou']}: {row['method']} | "
            f"thr={row['threshold']} | "
            f"IoU={row['mean_iou']:.4f}±{row['std_iou']:.4f} | "
            f"Dice={row['mean_dice']:.4f}±{row['std_dice']:.4f} | "
            f"Prec={row['mean_pixel_precision']:.4f} | "
            f"Recall={row['mean_pixel_recall']:.4f} | "
            f"Runtime={row['mean_runtime_sec']:.4f}s"
        )

    print("\nSaved outputs:")
    print("Per-image CSV        :", per_image_csv)
    print("Threshold summary CSV:", threshold_summary_csv)
    print("Best summary CSV     :", best_method_summary_csv)
    print("Report TXT           :", report_txt)
    print("Figures dir          :", figures_dir)

    print("\nStep 18 completed successfully.")


if __name__ == "__main__":
    main()