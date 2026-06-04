"""
predict.py
==========
Inference pipeline for RetinaVision-AI.

Run this AFTER training to segment blood vessels in new retinal images:
    python predict.py --image_path path/to/image.tif

Or run on the full test set:
    python predict.py --test_dir dataset/DRIVE/test

What this script does:
1. Loads the saved trained model
2. Preprocesses input image(s)
3. Runs model inference to get predicted probability maps
4. Thresholds predictions → binary mask
5. Calculates Dice and IoU metrics (if ground truth available)
6. Saves predicted mask + visualization side-by-side

WHY A SEPARATE PREDICT SCRIPT?
--------------------------------
Training and inference are different workflows.
- Training: needs data pipelines, callbacks, optimizers
- Inference: needs only a single forward pass + visualization
Keeping them separate makes each script cleaner and more focused.
This is standard production practice.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Our modules
from dataset.loader import (
    preprocess_retinal_image,
    preprocess_vessel_mask,
    load_dataset_paths,
    IMAGE_HEIGHT, IMAGE_WIDTH
)
from models.metrics import dice_loss, dice_coefficient, iou_metric
from utils.visualizer import (
    visualize_segmentation_result,
    save_prediction_image,
    visualize_batch_results
)


# ─────────────────────────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────────────────────────
def load_trained_model(model_path):
    """
    Loads a saved Keras model from disk.

    custom_objects: we must tell Keras about our custom loss/metrics
    because it doesn't know 'dice_loss', 'dice_coefficient', 'iou_metric'
    by default. Without this, model loading would throw an error.

    Args:
        model_path : path to the .h5 model file

    Returns:
        model : loaded Keras model ready for inference
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Please run 'python train.py' first to train and save the model."
        )

    print(f"  [INFO] Loading model from: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_loss"        : dice_loss,
            "dice_coefficient" : dice_coefficient,
            "iou_metric"       : iou_metric
        }
    )

    print(f"  [INFO] Model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────
# PREDICT ON SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────
def predict_single_image(model, image_path):
    """
    Runs inference on a single retinal image.

    Steps:
    1. Preprocess image (resize, CLAHE, normalize)
    2. Add batch dimension: (H, W, 3) → (1, H, W, 3)
       WHY? Keras always expects a batch, even for single image
    3. model.predict() → probability map
    4. Remove batch dimension: (1, H, W, 1) → (H, W, 1)

    Args:
        model      : trained Keras U-Net model
        image_path : path to input image

    Returns:
        original_image    : preprocessed image (H, W, 3) float32
        predicted_mask    : probability map (H, W, 1) float32, values in [0, 1]
        binary_mask       : thresholded mask (H, W, 1) uint8, values 0 or 1
    """
    print(f"  [PREDICT] Processing: {os.path.basename(image_path)}")

    # Step 1: Preprocess
    original_image = preprocess_retinal_image(image_path)  # (H, W, 3)

    # Step 2: Add batch dimension
    # np.expand_dims(..., axis=0): inserts a new axis at position 0
    # (512, 512, 3) → (1, 512, 512, 3) — batch of 1
    image_batch = np.expand_dims(original_image, axis=0)

    # Step 3: Forward pass (inference)
    # model.predict() runs the model without gradient computation (faster than model())
    # Output shape: (1, 512, 512, 1) — batch of 1 single-channel probability map
    prediction_batch = model.predict(image_batch, verbose=0)

    # Step 4: Remove batch dimension
    predicted_mask = prediction_batch[0]   # (512, 512, 1)

    # Step 5: Threshold at 0.5 → binary mask
    # Values > 0.5 → vessel (1), else → background (0)
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    return original_image, predicted_mask, binary_mask


# ─────────────────────────────────────────────────────────────────
# EVALUATE PREDICTIONS
# ─────────────────────────────────────────────────────────────────
def evaluate_prediction(binary_mask, ground_truth_mask):
    """
    Computes Dice and IoU between prediction and ground truth.

    These metrics are computed in NumPy (not TensorFlow) for simplicity
    during inference — we just need scalar numbers, not gradients.

    Args:
        binary_mask       : predicted binary mask (H, W, 1) or (H, W)
        ground_truth_mask : ground truth mask (H, W, 1) or (H, W)

    Returns:
        dict with "dice" and "iou" float values
    """
    # Flatten to 1D for element-wise computation
    pred_flat = binary_mask.flatten().astype(np.float32)
    gt_flat   = ground_truth_mask.flatten().astype(np.float32)

    # Intersection: pixels that are vessel in BOTH
    intersection = np.sum(pred_flat * gt_flat)

    # Dice
    dice = (2.0 * intersection + 1e-6) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-6)

    # IoU
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    iou   = (intersection + 1e-6) / (union + 1e-6)

    return {"dice": float(dice), "iou": float(iou)}


# ─────────────────────────────────────────────────────────────────
# PREDICT ON FULL TEST SET
# ─────────────────────────────────────────────────────────────────
def predict_test_set(model, test_dir, output_dir="outputs"):
    """
    Runs inference on all images in the test directory.

    Also computes average Dice and IoU across the test set.
    DRIVE benchmark: good models achieve Dice ≈ 0.81, IoU ≈ 0.68

    Args:
        model      : trained model
        test_dir   : path containing 'images/' and 'masks/' subdirectories
        output_dir : where to save results
    """
    pred_dir = os.path.join(output_dir, "predictions")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load test image and mask paths
    image_paths, mask_paths = load_dataset_paths(test_dir)

    if len(image_paths) == 0:
        print(f"  ⚠️  No test images found in {test_dir}")
        return

    print(f"\n  Running inference on {len(image_paths)} test images...")
    print("  " + "─" * 50)

    all_dice = []
    all_iou  = []

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        sample_name = os.path.splitext(os.path.basename(img_path))[0]

        # Predict
        original_image, predicted_mask, binary_mask = predict_single_image(model, img_path)

        # Load ground truth mask
        ground_truth = preprocess_vessel_mask(mask_path)

        # Evaluate
        metrics = evaluate_prediction(binary_mask, ground_truth)
        all_dice.append(metrics["dice"])
        all_iou.append(metrics["iou"])

        print(f"  [{idx+1:02d}/{len(image_paths)}] {sample_name:<30} "
              f"Dice: {metrics['dice']:.4f}  IoU: {metrics['iou']:.4f}")

        # Save predicted mask as PNG
        mask_save_path = os.path.join(pred_dir, f"{sample_name}_pred.png")
        save_prediction_image(binary_mask, mask_save_path)

        # Save visualization (4-panel figure)
        visualize_segmentation_result(
            original_image    = original_image,
            ground_truth_mask = ground_truth,
            predicted_mask    = predicted_mask,
            sample_name       = sample_name,
            save_dir          = plot_dir
        )

    # ── Summary Statistics ──────────────────────────────────
    print("\n" + "═" * 50)
    print("  TEST SET RESULTS")
    print("═" * 50)
    print(f"  Mean Dice Coefficient : {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"  Mean IoU Score        : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"  Best  Dice            : {np.max(all_dice):.4f}")
    print(f"  Worst Dice            : {np.min(all_dice):.4f}")
    print("═" * 50)
    print(f"\n  Predictions saved to : {pred_dir}/")
    print(f"  Visualizations saved to: {plot_dir}/")


# ─────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────
def parse_arguments():
    """
    Parses command-line arguments for flexible inference.

    Usage examples:
        python predict.py --test_dir dataset/DRIVE/test
        python predict.py --image_path myimage.tif --mask_path mymask.png
        python predict.py --model_path outputs/my_model.h5 --test_dir dataset/DRIVE/test
    """
    parser = argparse.ArgumentParser(
        description="RetinaVision-AI — Retinal Blood Vessel Segmentation Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/retinavision_best_model.h5",
        help="Path to the trained .h5 model file"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="dataset/DRIVE/test",
        help="Directory with images/ and masks/ subdirectories (for batch inference)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image (for single-image inference)"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to ground truth mask (optional, for evaluation)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save predictions and visualizations"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  RetinaVision-AI — Inference Pipeline")
    print("═" * 60)

    args = parse_arguments()

    # Load model
    model = load_trained_model(args.model_path)

    if args.image_path:
        # ── Single image mode ──────────────────────────────
        original_image, predicted_mask, binary_mask = predict_single_image(
            model, args.image_path
        )

        os.makedirs(args.output_dir, exist_ok=True)
        sample_name = os.path.splitext(os.path.basename(args.image_path))[0]

        if args.mask_path:
            # Evaluate against ground truth if provided
            ground_truth = preprocess_vessel_mask(args.mask_path)
            metrics = evaluate_prediction(binary_mask, ground_truth)
            print(f"\n  Dice: {metrics['dice']:.4f}   IoU: {metrics['iou']:.4f}")

            visualize_segmentation_result(
                original_image    = original_image,
                ground_truth_mask = ground_truth,
                predicted_mask    = predicted_mask,
                sample_name       = sample_name,
                save_dir          = os.path.join(args.output_dir, "plots")
            )
        else:
            # No ground truth — save prediction only
            save_prediction_image(
                binary_mask,
                os.path.join(args.output_dir, f"{sample_name}_pred.png")
            )
            print(f"  Prediction saved to: {args.output_dir}/{sample_name}_pred.png")
    else:
        # ── Full test set mode ─────────────────────────────
        predict_test_set(model, args.test_dir, args.output_dir)

    print("\n  ✅ Inference complete.\n")


if __name__ == "__main__":
    main()
