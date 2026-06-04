content = '''import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from PIL import Image as PILImage

IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512
BATCH_SIZE   = 2
AUTOTUNE     = tf.data.AUTOTUNE


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"  [INFO] Created directory: {path}")


def load_dataset_paths(dataset_root):
    image_paths = sorted(
        glob(os.path.join(dataset_root, "images", "*.tif")) +
        glob(os.path.join(dataset_root, "images", "*.jpg")) +
        glob(os.path.join(dataset_root, "images", "*.png"))
    )
    mask_paths = sorted(
        glob(os.path.join(dataset_root, "masks", "*.gif")) +
        glob(os.path.join(dataset_root, "masks", "*.png")) +
        glob(os.path.join(dataset_root, "masks", "*.jpg"))
    )
    return image_paths, mask_paths


def preprocess_retinal_image(image_path):
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_vessel_mask(mask_path):
    if isinstance(mask_path, bytes):
        mask_path = mask_path.decode("utf-8")
    mask = np.array(PILImage.open(mask_path).convert("L"))
    mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def parse_image_mask_pair(image_path, mask_path):
    def _load_pair(img_path, msk_path):
        image = preprocess_retinal_image(img_path)
        mask  = preprocess_vessel_mask(msk_path)
        return image, mask

    image, mask = tf.numpy_function(
        func=_load_pair,
        inp=[image_path, mask_path],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    mask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return image, mask


def augment_image_mask_pair(image, mask):
    combined = tf.concat([image, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    num_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k=num_rotations)
    image = combined[:, :, :3]
    mask  = combined[:, :, 3:]
    return image, mask


def build_tf_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image_mask_pair, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image_mask_pair, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def shuffle_dataset(image_paths, mask_paths, seed=42):
    image_paths, mask_paths = shuffle(image_paths, mask_paths, random_state=seed)
    return image_paths, mask_paths
'''

with open('dataset/loader.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("loader.py fixed successfully")
