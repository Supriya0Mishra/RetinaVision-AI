"""
utils/visualizer.py
===================
Visualization tools for the RetinaVision-AI project.

Produces side-by-side plots of:
  [Original Image] | [Ground Truth Mask] | [Predicted Mask] | [Overlay]

These visualizations serve two purposes:
  1. Qualitative evaluation: visually inspect segmentation quality
  2. Portfolio/presentation: compelling figures for README and demos
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────
# SAVE SINGLE PREDICTION IMAGE
# ─────────────────────────────────────────────────────────────────
def save_prediction_image(predicted_mask, save_path):
    """
    Saves a predicted mask to disk as a grayscale PNG.

    The model outputs values in [0, 1] (probabilities).
    We threshold at 0.5 and convert to 0-255 for saving.

    Args:
        predicted_mask : numpy array of shape (H, W) or (H, W, 1), values in [0, 1]
        save_path      : full path including filename, e.g. 'outputs/predictions/01.png'
    """
    # Remove channel dimension if present: (512, 512, 1) → (512, 512)
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask[:, :, 0]

    # Convert probability map to binary mask (threshold at 0.5)
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    # Save using OpenCV
    cv2.imwrite(save_path, binary_mask)


# ─────────────────────────────────────────────────────────────────
# VISUALIZE SINGLE SAMPLE (4-panel figure)
# ─────────────────────────────────────────────────────────────────
def visualize_segmentation_result(original_image, ground_truth_mask, predicted_mask,
                                   sample_name="sample", save_dir="outputs/plots"):
    """
    Creates a 4-panel visualization figure and saves it to disk.

    Panels:
    1. Original retinal image (RGB)
    2. Ground truth mask (manual annotation by ophthalmologist)
    3. Predicted mask (binary, thresholded at 0.5)
    4. Overlay: green = true positive vessels on original image

    Args:
        original_image   : numpy array (H, W, 3), float32 in [0, 1]
        ground_truth_mask: numpy array (H, W) or (H, W, 1), float32 in [0, 1]
        predicted_mask   : numpy array (H, W) or (H, W, 1), float32 in [0, 1]
        sample_name      : identifier for filename (e.g. "01_test")
        save_dir         : directory where plot is saved

    Returns:
        save_path : full path of the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Normalize inputs ──────────────────────────────────────
    # Squeeze channel dim from masks: (H, W, 1) → (H, W)
    if ground_truth_mask.ndim == 3:
        ground_truth_mask = ground_truth_mask[:, :, 0]
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask[:, :, 0]

    # Ensure image is in [0, 1] for matplotlib display
    if original_image.max() > 1.0:
        original_image = original_image / 255.0

    # Binary predicted mask (threshold at 0.5)
    binary_prediction = (predicted_mask > 0.5).astype(np.float32)

    # ── Create overlay ────────────────────────────────────────
    # overlay: original image with predicted vessels highlighted in bright green
    overlay = original_image.copy()
    vessel_pixels = binary_prediction > 0.5   # boolean mask: True where vessel

    # Paint vessel pixels green (R=0, G=1, B=0)
    overlay[vessel_pixels, 0] = 0.0   # Red channel → 0
    overlay[vessel_pixels, 1] = 1.0   # Green channel → 1 (bright green)
    overlay[vessel_pixels, 2] = 0.0   # Blue channel → 0

    # ── Calculate metrics for display ────────────────────────
    gt_flat   = ground_truth_mask.flatten()
    pred_flat = binary_prediction.flatten()

    # Dice coefficient: 2*TP / (2*TP + FP + FN)
    intersection = np.sum(gt_flat * pred_flat)
    dice_score = (2.0 * intersection + 1e-6) / (np.sum(gt_flat) + np.sum(pred_flat) + 1e-6)

    # IoU: TP / (TP + FP + FN)
    union = np.sum(gt_flat) + np.sum(pred_flat) - intersection
    iou_score = (intersection + 1e-6) / (union + 1e-6)

    # ── Build Figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle(
        f"RetinaVision-AI — Segmentation Results | {sample_name}\n"
        f"Dice: {dice_score:.4f}   IoU: {iou_score:.4f}",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    # Define 4 equal columns
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (original_image,       "Original Fundus Image",      "viridis"),
        (ground_truth_mask,    "Ground Truth Mask\n(Manual Annotation)", "gray"),
        (binary_prediction,    "Predicted Mask\n(U-Net Output)", "gray"),
        (overlay,              "Overlay\n(Green = Detected Vessels)", "viridis"),
    ]

    for i, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if data.ndim == 3:
            ax.imshow(data)          # RGB image
        else:
            ax.imshow(data, cmap=cmap)  # Grayscale mask

        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.axis("off")  # Hide axis ticks

        # Add thin border around each panel
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(1)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f"{sample_name}_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  [SAVED] Visualization → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────
# VISUALIZE TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────
def plot_training_history(csv_log_path, save_dir="outputs/plots"):
    """
    Reads the training CSV log and plots loss + metrics over epochs.

    Shows:
    - Training vs Validation Dice Loss
    - Training vs Validation Dice Coefficient
    - Training vs Validation IoU

    Args:
        csv_log_path : path to 'training_log.csv' generated by CSVLogger callback
        save_dir     : directory to save the plot
    """
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    # Load training log
    history = pd.read_csv(csv_log_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle("Training History — RetinaVision-AI", color="white",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("loss",             "val_loss",             "Dice Loss",        "#ef4444", "#f97316"),
        ("dice_coefficient", "val_dice_coefficient", "Dice Coefficient", "#22c55e", "#86efac"),
        ("iou_metric",       "val_iou_metric",       "IoU Score",        "#3b82f6", "#93c5fd"),
    ]

    for ax, (train_col, val_col, title, train_color, val_color) in zip(axes, metrics):
        ax.set_facecolor("#1a1a2e")

        if train_col in history.columns:
            ax.plot(history["epoch"], history[train_col],
                    color=train_color, linewidth=2, label="Train")
        if val_col in history.columns:
            ax.plot(history["epoch"], history[val_col],
                    color=val_color, linewidth=2, linestyle="--", label="Validation")

        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        ax.legend(facecolor="#222222", labelcolor="white")
        ax.grid(alpha=0.2, color="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  [SAVED] Training history plot → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────
# BATCH VISUALIZE: visualize multiple test samples at once
# ─────────────────────────────────────────────────────────────────
def visualize_batch_results(images, gt_masks, pred_masks, save_dir="outputs/plots",
                             max_samples=5):
    """
    Runs visualize_segmentation_result on up to max_samples from a batch.

    Useful for quickly reviewing model performance across several test images.

    Args:
        images      : numpy array (N, H, W, 3)
        gt_masks    : numpy array (N, H, W, 1)
        pred_masks  : numpy array (N, H, W, 1)
        save_dir    : output directory
        max_samples : limit to avoid generating too many files
    """
    n = min(len(images), max_samples)
    print(f"\n  [INFO] Generating visualizations for {n} samples...")

    saved_paths = []
    for i in range(n):
        path = visualize_segmentation_result(
            original_image    = images[i],
            ground_truth_mask = gt_masks[i],
            predicted_mask    = pred_masks[i],
            sample_name       = f"sample_{i+1:02d}",
            save_dir          = save_dir
        )
        saved_paths.append(path)

    return saved_paths
