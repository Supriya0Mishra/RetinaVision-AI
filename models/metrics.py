"""
models/metrics.py
=================
Custom loss functions and evaluation metrics for medical image segmentation.

WHY NOT USE BINARY CROSS-ENTROPY?
----------------------------------
Standard BCE treats every pixel equally.
In retinal images, ~85% of pixels are BACKGROUND (not vessels).
→ A model that predicts "all background" gets 85% accuracy — but is useless.
→ We need metrics that reward correct vessel detection specifically.

DICE COEFFICIENT: The gold standard for segmentation overlap.
IoU (Jaccard Index): Another overlap metric, stricter than Dice.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


# ─────────────────────────────────────────────────────────────────
# DICE COEFFICIENT
# ─────────────────────────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Measures overlap between predicted mask and ground truth mask.

    Formula: Dice = (2 * |A ∩ B|) / (|A| + |B|)
    Where A = predicted pixels, B = ground truth pixels

    Range: 0 (no overlap) to 1 (perfect overlap)
    smooth: small value added to numerator/denominator to avoid division by zero

    Why smooth=1e-6? If both masks are empty (no vessels), 0/0 → NaN.
    Adding smooth makes it 1e-6 / 2e-6 = 0.5 → stable.

    Interview answer: "Dice is twice the intersection over the sum of both sets.
    It's equivalent to F1 score applied pixel-wise."
    """
    # Flatten both tensors from 2D → 1D for element-wise comparison
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Intersection: pixels that are vessel in BOTH prediction and ground truth
    intersection = K.sum(y_true_flat * y_pred_flat)

    # Dice formula
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

    return dice


# ─────────────────────────────────────────────────────────────────
# DICE LOSS
# ─────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    """
    Loss function based on Dice Coefficient.

    Formula: Dice Loss = 1 - Dice Coefficient

    WHY USE THIS INSTEAD OF BCE?
    → Directly optimizes the metric we care about (overlap)
    → Naturally handles class imbalance (vessels are rare pixels)
    → A model must actually segment vessels correctly to minimize this loss

    Interview answer: "I replaced binary cross-entropy with Dice loss because
    retinal images have severe class imbalance — about 85% background.
    Dice loss penalizes missed vessels more directly."
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────
# IoU (Intersection over Union) — also called Jaccard Index
# ─────────────────────────────────────────────────────────────────
def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Measures what fraction of the combined vessel area is correctly predicted.

    Formula: IoU = |A ∩ B| / |A ∪ B|
                 = intersection / (|A| + |B| - intersection)

    Range: 0 to 1. Stricter than Dice (denominator is larger).
    A Dice of 0.9 ≈ IoU of 0.82

    Interview answer: "IoU divides intersection by union — it penalizes both
    false positives and false negatives. It's the standard metric in
    segmentation benchmarks like DRIVE."
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou
