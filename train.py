"""
train.py
========
Main training script for RetinaVision-AI.

Run this from the project root:
    python train.py

What this script does:
1. Loads DRIVE dataset image/mask paths
2. Builds the U-Net model
3. Sets up training callbacks (model saving, early stopping, LR reduction)
4. Runs training and saves the best model
5. Plots training history

WHAT CHANGED vs ORIGINAL:
---------------------------
Original issues:                  Our fixes:
─────────────────────────────────────────────────────────────────
resize commented out            → Always resize (consistent inputs)
No augmentation                 → Added flips + rotations (augment=True)
Hard-coded paths                → Config dict at top (easy to change)
No CLAHE preprocessing          → Added contrast enhancement
Typo in variable name           → Fixed (valid_setps → valid_steps)
No structured logging           → Clear section headers + progress prints
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs (show errors only)

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,    # Saves model whenever val_loss improves
    CSVLogger,          # Writes metrics to a CSV file after each epoch
    ReduceLROnPlateau,  # Cuts learning rate when training plateaus
    EarlyStopping,      # Stops training early to prevent overfitting
    TensorBoard         # Enables TensorBoard visualization (optional)
)
from tensorflow.keras.optimizers import Adam

# Our custom modules
from dataset.loader import (
    load_dataset_paths,
    shuffle_dataset,
    build_tf_dataset,
    ensure_directory_exists,
    IMAGE_HEIGHT, IMAGE_WIDTH
)
from models.unet import build_retina_unet
from models.metrics import dice_loss, dice_coefficient, iou_metric
from utils.visualizer import plot_training_history


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — Change these settings to adjust training
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # Dataset paths (update to match your folder structure)
    "dataset_root"  : "dataset/DRIVE",
    "train_dir"     : "dataset/DRIVE/train",
    "valid_dir"     : "dataset/DRIVE/test",   # DRIVE test split used for validation

    # Output paths
    "output_dir"    : "outputs",
    "model_path"    : "outputs/retinavision_best_model.h5",
    "csv_log_path"  : "outputs/training_log.csv",
    "tensorboard_dir": "outputs/tensorboard_logs",

    # Hyperparameters
    "batch_size"    : 2,       # Small batch due to limited VRAM
    "learning_rate" : 1e-3,    # Adam optimizer starting learning rate
    "num_epochs"    : 150,     # Max epochs (EarlyStopping will likely stop earlier)
    "input_shape"   : (IMAGE_HEIGHT, IMAGE_WIDTH, 3),

    # Training stability
    "seed"          : 42,      # Reproducibility seed
}


def set_global_seeds(seed):
    """
    Sets seeds for NumPy and TensorFlow to ensure reproducible results.
    Same seed → same weight initialization → same training outcome.
    Important for debugging and fair comparison of experiments.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"  [INFO] Random seed set to {seed}")


def compute_dataset_steps(num_samples, batch_size):
    """
    Calculates how many gradient update steps per epoch.

    Formula: steps = ceil(num_samples / batch_size)
    Example: 20 images, batch_size=2 → 10 steps/epoch
             21 images, batch_size=2 → 11 steps/epoch (last batch has 1 image)

    The ceiling (not floor) ensures the last partial batch is included.
    """
    return (num_samples + batch_size - 1) // batch_size


def build_training_callbacks(config):
    """
    Creates Keras training callbacks.

    Callbacks = functions that run automatically at the end of each epoch.

    1. ModelCheckpoint: saves the model file only when val_loss improves
       → Ensures we always have the BEST model, not the final (potentially overfit) one

    2. ReduceLROnPlateau: if val_loss doesn't improve for 5 epochs, multiply LR by 0.1
       → Helps escape local minima; large LR for fast early learning, small LR for fine-tuning

    3. CSVLogger: writes a row to CSV after each epoch
       → Lets us plot training history later without rerunning training

    4. EarlyStopping: stops if val_loss doesn't improve for 15 epochs
       → Prevents wasting compute on epochs that won't improve the model

    5. TensorBoard: enables rich training visualization in browser
       → Run: tensorboard --logdir outputs/tensorboard_logs
    """
    callbacks = [

        # Save best model checkpoint
        ModelCheckpoint(
            filepath        = config["model_path"],
            monitor         = "val_loss",    # Watch validation loss
            save_best_only  = True,          # Only save if it's the best so far
            verbose         = 1,             # Print when saving
            mode            = "min"          # Lower loss = better
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,          # New LR = current LR × 0.5
            patience = 10,           # Wait 10 epochs before reducing
            min_lr   = 1e-7,         # Never go below this LR
            verbose  = 1
        ),

        # Log metrics to CSV
        CSVLogger(
            filename = config["csv_log_path"],
            append   = False         # Overwrite old log (set True to continue training)
        ),

        # Early stopping
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 25,       # Stop after 25 epochs without improvement
            restore_best_weights = True,     # Load best weights when stopping
            verbose              = 1
        ),

        # TensorBoard visualization
        TensorBoard(
            log_dir    = config["tensorboard_dir"],
            histogram_freq = 0       # Don't compute weight histograms (saves time)
        ),
    ]
    return callbacks


def main():
    print("\n" + "═" * 60)
    print("  RetinaVision-AI — Training Pipeline")
    print("═" * 60)

    # ── Step 1: Seeds & Directories ───────────────────────────
    set_global_seeds(CONFIG["seed"])
    ensure_directory_exists(CONFIG["output_dir"])
    ensure_directory_exists(CONFIG["tensorboard_dir"])

    # ── Step 2: Load Dataset Paths ────────────────────────────
    print("\n[1/5] Loading dataset paths...")
    train_images, train_masks = load_dataset_paths(CONFIG["train_dir"])
    valid_images, valid_masks = load_dataset_paths(CONFIG["valid_dir"])

    # Shuffle training data (important: shuffles both lists in sync)
    train_images, train_masks = shuffle_dataset(train_images, train_masks, seed=CONFIG["seed"])

    print(f"  Training samples : {len(train_images)}")
    print(f"  Validation samples: {len(valid_images)}")

    if len(train_images) == 0:
        print("\n  ⚠️  ERROR: No training images found.")
        print("  Expected structure: dataset/DRIVE/train/images/ and dataset/DRIVE/train/masks/")
        print("  Please download the DRIVE dataset and organize it as above.")
        return

    # ── Step 3: Build TF Datasets ─────────────────────────────
    print("\n[2/5] Building TensorFlow data pipelines...")
    train_dataset = build_tf_dataset(
        train_images, train_masks,
        batch_size=CONFIG["batch_size"],
        augment=True           # ← Apply augmentation during training
    )
    valid_dataset = build_tf_dataset(
        valid_images, valid_masks,
        batch_size=CONFIG["batch_size"],
        augment=False          # ← Never augment validation data
    )

    # Steps per epoch
    train_steps = compute_dataset_steps(len(train_images), CONFIG["batch_size"])
    valid_steps = compute_dataset_steps(len(valid_images), CONFIG["batch_size"])
    print(f"  Train steps/epoch: {train_steps} | Valid steps/epoch: {valid_steps}")

    # ── Step 4: Build & Compile Model ─────────────────────────
    print("\n[3/5] Building U-Net model...")
    model = build_retina_unet(input_shape=CONFIG["input_shape"])

    # Compile: specify optimizer, loss function, and evaluation metrics
    model.compile(
        optimizer = Adam(learning_rate=CONFIG["learning_rate"]),
        loss      = dice_loss,                    # Our custom Dice loss
        metrics   = [
            dice_coefficient,                     # Tracks segmentation overlap
            iou_metric,                           # Tracks Intersection over Union
            tf.keras.metrics.Recall(name="recall"),     # TP / (TP + FN)
            tf.keras.metrics.Precision(name="precision") # TP / (TP + FP)
        ]
    )

    print(f"  Total parameters: {model.count_params():,}")

    # ── Step 5: Train ─────────────────────────────────────────
    print("\n[4/5] Starting training...")
    print(f"  Batch size   : {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Max epochs   : {CONFIG['num_epochs']}")
    print(f"  Best model → : {CONFIG['model_path']}\n")

    callbacks = build_training_callbacks(CONFIG)

    history = model.fit(
        train_dataset,
        epochs            = CONFIG["num_epochs"],
        validation_data   = valid_dataset,
        steps_per_epoch   = train_steps,
        validation_steps  = valid_steps,
        callbacks         = callbacks,
        verbose           = 1
    )

    # ── Step 6: Plot Training History ─────────────────────────
    print("\n[5/5] Generating training history plots...")
    if os.path.exists(CONFIG["csv_log_path"]):
        plot_training_history(
            csv_log_path = CONFIG["csv_log_path"],
            save_dir     = CONFIG["output_dir"] + "/plots"
        )

    print("\n" + "═" * 60)
    print("  ✅ Training complete!")
    print(f"  Best model saved to: {CONFIG['model_path']}")
    print(f"  Visualizations in  : {CONFIG['output_dir']}/plots/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()

