import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================================
# Configuration
# =====================================
MODEL_PATH = "saved_models/best_attention_unet.keras"
IMAGE_DIR = "dataset/test/images"
IMG_SIZE = (256, 256)

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model Loaded Successfully!")

# =====================================
# Load Test Images
# =====================================
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
])

print(f"\nFound {len(image_files)} test images.\n")

# =====================================
# Debug First Image
# =====================================
for file in image_files:

    image_path = os.path.join(IMAGE_DIR, file)

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read image.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32) / 255.0

    pred = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0]

    print("\n==============================")
    print("Image :", file)
    print("Shape :", pred.shape)
    print("Min   :", pred.min())
    print("Max   :", pred.max())
    print("Mean  :", pred.mean())
    print("==============================")

    pred = pred.squeeze()

    # Save raw prediction
    cv2.imwrite(
        "raw_prediction.png",
        (pred * 255).astype(np.uint8)
    )

    print("\n✅ Raw prediction saved as raw_prediction.png")