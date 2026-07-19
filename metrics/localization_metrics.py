import cv2
import numpy as np


def binarize_heatmap(heatmap, threshold):
    """
    Convert grayscale heatmap to binary predicted mask.
    heatmap must be uint8 or float image.
    threshold range usually 0-255.
    """
    if heatmap.dtype != np.uint8:
        heatmap = normalize_to_uint8(heatmap)

    _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    return binary


def binarize_mask(mask):
    """
    Convert ground-truth mask to binary mask.
    Any pixel > 0 is treated as manipulated region.
    """
    if mask.dtype != np.uint8:
        mask = normalize_to_uint8(mask)

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return binary


def normalize_to_uint8(image):
    """
    Normalize image/heatmap to uint8 [0, 255].
    """
    img = image.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)

    return img


def compute_confusion_pixels(pred_binary, gt_binary):
    """
    Compute pixel-level TP, FP, TN, FN.
    """
    pred = pred_binary > 0
    gt = gt_binary > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    return int(tp), int(fp), int(tn), int(fn)


def compute_localization_metrics(pred_binary, gt_binary):
    """
    Compute segmentation/localization metrics.

    Returns:
        dict with IoU, Dice, Precision, Recall, Specificity, FPR, Accuracy
    """
    tp, fp, tn, fn = compute_confusion_pixels(pred_binary, gt_binary)

    union = tp + fp + fn
    pred_positive = tp + fp
    gt_positive = tp + fn
    total = tp + fp + tn + fn

    iou = tp / union if union > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / pred_positive if pred_positive > 0 else 0.0
    recall = tp / gt_positive if gt_positive > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "iou": iou,
        "dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "pixel_accuracy": accuracy,
    }


def evaluate_heatmap_against_mask(heatmap, gt_mask, threshold):
    """
    Full evaluation for one heatmap and one GT mask.
    """
    if heatmap.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (heatmap.shape[1], heatmap.shape[0]))

    pred_binary = binarize_heatmap(heatmap, threshold)
    gt_binary = binarize_mask(gt_mask)

    metrics = compute_localization_metrics(pred_binary, gt_binary)

    return metrics, pred_binary, gt_binary