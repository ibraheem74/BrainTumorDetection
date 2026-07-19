import cv2
import numpy as np

# Image size for the model
IMAGE_SIZE = (256, 256)


def preprocess_image(image_path):
    """
    Load and preprocess an MRI image.
    """

    # Read image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize pixel values (0-255 → 0-1)
    image = image.astype(np.float32) / 255.0

    return image


def preprocess_mask(mask_path):
    """
    Load and preprocess a tumor mask.
    """

    # Read mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize mask
    mask = cv2.resize(mask, IMAGE_SIZE)

    # Convert mask to binary
    mask = (mask > 0).astype(np.float32)

    # Add channel dimension
    mask = np.expand_dims(mask, axis=-1)

    return mask