"""
models/unet.py
==============
Defines the U-Net architecture for retinal blood vessel segmentation.

WHY U-NET?
----------
U-Net was originally designed for biomedical image segmentation (Ronneberger et al., 2015).
It works extremely well on small datasets (like DRIVE's 40 images) because:
  1. Skip connections preserve fine spatial detail (thin vessel edges)
  2. Encoder-decoder structure captures both context and location
  3. Pixel-wise output → ideal for segmentation masks

ARCHITECTURE OVERVIEW:
  Input Image (512x512x3)
       ↓
  Encoder (4 blocks) → progressively shrinks image, learns "what"
       ↓
  Bottleneck            → deepest representation
       ↓
  Decoder (4 blocks) → progressively grows image back, learns "where"
       ↓
  Output Mask (512x512x1) with sigmoid → pixel probability of being a vessel
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,           # 2D convolution: extracts spatial features
    BatchNormalization,  # Normalizes layer outputs → stable, faster training
    Activation,       # Applies non-linearity (ReLU here)
    MaxPool2D,        # Downsampling: shrinks spatial size, keeps dominant features
    Conv2DTranspose,  # Upsampling (transposed conv): grows spatial size back
    Concatenate,      # Skip connection: merges encoder + decoder feature maps
    Input             # Entry point for the model
)
from tensorflow.keras.models import Model


# ─────────────────────────────────────────────────────────────────
# BUILDING BLOCK: Convolutional Block
# ─────────────────────────────────────────────────────────────────
def convolutional_block(input_tensor, num_filters):
    """
    Applies two rounds of: Conv2D → BatchNorm → ReLU

    This is the fundamental learning unit of U-Net.
    Two convolutions are used (not one) to give the model more
    capacity to learn complex patterns at the same scale.

    Args:
        input_tensor : the feature map coming in
        num_filters  : how many feature detectors to use (e.g. 64, 128…)

    Returns:
        x : processed feature map of same spatial size but `num_filters` channels
    """

    # --- First convolution ---
    # Conv2D(num_filters, kernel_size=3, padding="same"):
    #   • kernel_size=3 → 3x3 filter slides over image
    #   • padding="same" → output height/width = input height/width (no shrinkage)
    #   • num_filters → number of different patterns we're detecting
    x = Conv2D(num_filters, kernel_size=3, padding="same")(input_tensor)

    # BatchNormalization: normalizes outputs so mean≈0, std≈1
    # Prevents exploding/vanishing gradients; speeds up training
    x = BatchNormalization()(x)

    # ReLU: sets all negative values to 0 → introduces non-linearity
    # Without this, stacking layers is mathematically equivalent to one layer
    x = Activation("relu")(x)

    # --- Second convolution (same idea, deeper feature extraction) ---
    x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# ─────────────────────────────────────────────────────────────────
# ENCODER BLOCK (Contracting Path)
# ─────────────────────────────────────────────────────────────────
def encoder_block(input_tensor, num_filters):
    """
    One step DOWN the encoder (left side of U-Net).

    Two outputs:
      - skip_features: full-resolution features saved for the skip connection
      - pooled_output: halved-resolution features passed to next encoder block

    Think of it as: "learn features at this scale, then zoom out"

    Args:
        input_tensor : incoming feature map
        num_filters  : number of feature detectors

    Returns:
        skip_features : feature map BEFORE downsampling (used in decoder)
        pooled_output : feature map AFTER downsampling (fed to next layer)
    """

    # Learn features at the current spatial resolution
    skip_features = convolutional_block(input_tensor, num_filters)

    # MaxPool2D((2,2)): takes the max value in each 2x2 window
    # → halves height and width (512→256→128→64→32)
    # → forces the network to look at larger context
    pooled_output = MaxPool2D(pool_size=(2, 2))(skip_features)

    return skip_features, pooled_output


# ─────────────────────────────────────────────────────────────────
# DECODER BLOCK (Expanding Path)
# ─────────────────────────────────────────────────────────────────
def decoder_block(input_tensor, skip_features, num_filters):
    """
    One step UP the decoder (right side of U-Net).

    The KEY idea: merge upsampled decoder features WITH encoder skip features.
    This is what makes U-Net special — the skip connection "reminds" the
    decoder of fine spatial details the encoder saw early on.

    Args:
        input_tensor  : feature map from the previous (deeper) decoder layer
        skip_features : saved feature map from the matching encoder block
        num_filters   : number of feature detectors

    Returns:
        x : upsampled + merged + refined feature map
    """

    # Conv2DTranspose: "reverse convolution" — doubles spatial size
    # strides=2 means the output is 2x bigger in both height and width
    # This is the learned upsampling (better than simple bilinear resize)
    x = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=2, padding="same")(input_tensor)

    # Concatenate: glue the upsampled map WITH the skip features along channel axis
    # Example: (256x256x128) + (256x256x128) → (256x256x256)
    # This is what makes the "U" shape — information flows across the U
    x = Concatenate()([x, skip_features])

    # Refine the merged features with another convolutional block
    x = convolutional_block(x, num_filters)

    return x


# ─────────────────────────────────────────────────────────────────
# FULL U-NET MODEL
# ─────────────────────────────────────────────────────────────────
def build_retina_unet(input_shape=(512, 512, 3)):
    """
    Constructs the complete U-Net model for retinal vessel segmentation.

    Filter progression: 64 → 128 → 256 → 512 → 1024 (bottleneck)
    Each encoder level doubles the filters while halving the spatial size.
    Each decoder level halves the filters while doubling the spatial size.

    Args:
        input_shape : (H, W, C) — default 512x512 RGB image

    Returns:
        model : compiled-ready Keras Model
    """

    # ── Input ──────────────────────────────────────────────────
    # Input() defines the entry tensor. Shape = (512, 512, 3) for RGB.
    inputs = Input(shape=input_shape, name="retina_input")

    # ── Encoder (Contracting Path) ─────────────────────────────
    # Each block: learn features → save skip → downsample
    # s = skip features (saved for decoder)
    # p = pooled output (sent to next encoder block)

    s1, p1 = encoder_block(inputs, num_filters=64)   # s1: 512x512x64,  p1: 256x256x64
    s2, p2 = encoder_block(p1,     num_filters=128)  # s2: 256x256x128, p2: 128x128x128
    s3, p3 = encoder_block(p2,     num_filters=256)  # s3: 128x128x256, p3: 64x64x256
    s4, p4 = encoder_block(p3,     num_filters=512)  # s4: 64x64x512,   p4: 32x32x512

    # ── Bottleneck ─────────────────────────────────────────────
    # Deepest level — no pooling here, just convolutions
    # The model sees the most abstract, global representation here
    # Shape: 32x32x1024
    bottleneck = convolutional_block(p4, num_filters=1024)

    # ── Decoder (Expanding Path) ────────────────────────────────
    # Each block: upsample → concat with skip → refine
    # Notice the skip connections are used in REVERSE order (s4→s3→s2→s1)

    d1 = decoder_block(bottleneck, s4, num_filters=512)  # 64x64x512
    d2 = decoder_block(d1,         s3, num_filters=256)  # 128x128x256
    d3 = decoder_block(d2,         s2, num_filters=128)  # 256x256x128
    d4 = decoder_block(d3,         s1, num_filters=64)   # 512x512x64

    # ── Output Layer ────────────────────────────────────────────
    # Conv2D with 1 filter → produces a single-channel (grayscale) output
    # sigmoid activation → squishes output to [0, 1]
    # Each pixel value = probability that pixel belongs to a blood vessel
    # > 0.5 → vessel (white in mask), < 0.5 → background (black in mask)
    outputs = Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="segmentation_output"
    )(d4)

    # Build and return the Keras Model
    model = Model(inputs=inputs, outputs=outputs, name="RetinaVision_UNet")
    return model


# ─────────────────────────────────────────────────────────────────
# Quick test: run this file directly to verify architecture
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_retina_unet(input_shape=(512, 512, 3))
    model.summary()
    print("\n✅ Model built successfully.")
    print(f"   Total parameters: {model.count_params():,}")
