import numpy as np
from metrics.localization_metrics import evaluate_heatmap_against_mask


def main():
    print("=" * 70)
    print("Localization Metrics Test")
    print("=" * 70)

    heatmap = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    heatmap[2:6, 2:6] = 200
    mask[4:8, 4:8] = 255

    metrics, pred_binary, gt_binary = evaluate_heatmap_against_mask(
        heatmap,
        mask,
        threshold=100,
    )

    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nPredicted positive pixels:", (pred_binary > 0).sum())
    print("GT positive pixels       :", (gt_binary > 0).sum())

    print("\nStep 17B completed successfully.")


if __name__ == "__main__":
    main()