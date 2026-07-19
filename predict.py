import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models.losses import bce_dice_loss
from models.metrics import dice_coef, iou_score

# -----------------------------------
# Configuration
# -----------------------------------

MODEL_PATH = "saved_models/best_attention_unet.keras"

IMAGE_PATH = "dataset/test/images/TCGA_CS_4944_20010208_13.tif"
# Change the above path to any test MRI image

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

# -----------------------------------
# Load Model
# -----------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_score": iou_score
    }
)

print("✅ Model Loaded Successfully!")

# -----------------------------------
# Read Image
# -----------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(original, IMG_SIZE)
image = image.astype(np.float32) / 255.0

input_image = np.expand_dims(image, axis=0)

# -----------------------------------
# Prediction
# -----------------------------------

prediction = model.predict(input_image, verbose=0)

mask = prediction[0, :, :, 0]
mask = (mask > 0.5).astype(np.uint8)

# -----------------------------------
# Save Predicted Mask
# -----------------------------------

mask_path = os.path.join(OUTPUT_DIR, "predicted_mask.png")

cv2.imwrite(mask_path, mask * 255)

print("✅ Predicted mask saved to:", mask_path)

# -----------------------------------
# Display Results
# -----------------------------------

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()