import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
LOG_FILE = "logs/training_log.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Check CSV
# ==========================
if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"{LOG_FILE} not found!")

# Load CSV
history = pd.read_csv(LOG_FILE)

print("Columns found:")
print(history.columns.tolist())

# ==========================
# Loss Graph
# ==========================
if "loss" in history.columns and "val_loss" in history.columns:
    plt.figure(figsize=(8,5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

# ==========================
# Dice Score
# ==========================
if "dice_coef" in history.columns and "val_dice_coef" in history.columns:
    plt.figure(figsize=(8,5))
    plt.plot(history["dice_coef"], label="Training Dice")
    plt.plot(history["val_dice_coef"], label="Validation Dice")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "dice_curve.png"))
    plt.close()

# ==========================
# IoU
# ==========================
if "iou_score" in history.columns and "val_iou_score" in history.columns:
    plt.figure(figsize=(8,5))
    plt.plot(history["iou_score"], label="Training IoU")
    plt.plot(history["val_iou_score"], label="Validation IoU")
    plt.title("IoU Score")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_curve.png"))
    plt.close()

# ==========================
# Accuracy (if available)
# ==========================
if "accuracy" in history.columns and "val_accuracy" in history.columns:
    plt.figure(figsize=(8,5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
    plt.close()

print("\n✅ Graphs generated successfully!")
print(f"Saved in: {OUTPUT_DIR}")