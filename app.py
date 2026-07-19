import os
import cv2
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# =====================================================
# Flask App Configuration
# =====================================================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PRED_FOLDER = "static/predictions"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PRED_FOLDER"] = PRED_FOLDER

# =====================================================
# Load Trained Model
# =====================================================
MODEL_PATH = "saved_models/best_attention_unet.keras"

print("Loading Attention U-Net Model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model Loaded Successfully!")

IMG_SIZE = (256, 256)

# Prediction Threshold
PIXEL_THRESHOLD = 0.5

# Minimum number of predicted pixels to classify as tumor
TUMOR_AREA_THRESHOLD = 200


# =====================================================
# Home Page
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# Prediction
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():

    # Check file
    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No image selected."

    filename = secure_filename(file.filename)

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(upload_path)

    # -----------------------------------
    # Read Image
    # -----------------------------------
    image = cv2.imread(upload_path)

    if image is None:
        return "Unable to read uploaded image."

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, IMG_SIZE)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    # -----------------------------------
    # Predict
    # -----------------------------------
    prediction = model.predict(
        np.expand_dims(image_rgb, axis=0),
        verbose=0
    )[0]

    prediction = prediction.squeeze()

    # Debug
    print("\n========== Prediction ==========")
    print("Shape :", prediction.shape)
    print("Min   :", prediction.min())
    print("Max   :", prediction.max())
    print("Mean  :", prediction.mean())
    print("===============================\n")

    # Save raw probability map (optional)
    cv2.imwrite(
        "raw_prediction.png",
        (prediction * 255).astype(np.uint8)
    )

    # -----------------------------------
    # Binary Mask
    # -----------------------------------
    binary_mask = (prediction > PIXEL_THRESHOLD).astype(np.uint8)

    tumor_pixels = int(np.sum(binary_mask))

    # -----------------------------------
    # Decision
    # -----------------------------------
    if tumor_pixels > TUMOR_AREA_THRESHOLD:

        result = "Tumor Detected"

        confidence = float(
            np.mean(prediction[binary_mask == 1])
        ) * 100

    else:

        result = "No Tumor Detected"

        confidence = (1 - float(np.max(prediction))) * 100

    confidence = round(confidence, 2)

    # -----------------------------------
    # Save Predicted Mask
    # -----------------------------------
    mask = binary_mask * 255

    pred_name = "pred_" + filename

    pred_path = os.path.join(
        app.config["PRED_FOLDER"],
        pred_name
    )

    cv2.imwrite(pred_path, mask)

    # -----------------------------------
    # Display Result
    # -----------------------------------
    return render_template(
        "index.html",
        uploaded_image=upload_path,
        predicted_image=pred_path,
        result=result,
        confidence=confidence,
        tumor_pixels=tumor_pixels
    )


# =====================================================
# Run Flask App
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)