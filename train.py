import os
import tensorflow as tf

from utils.dataloader import BrainTumorDataset
from models.attention_unet import build_attention_unet
from models.losses import bce_dice_loss
from models.metrics import dice_coef, iou_score

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4

# -----------------------------
# Create folders
# -----------------------------
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
train_dataset = BrainTumorDataset(
    image_dir="dataset/train/images",
    mask_dir="dataset/train/masks",
    batch_size=BATCH_SIZE
)

val_dataset = BrainTumorDataset(
    image_dir="dataset/val/images",
    mask_dir="dataset/val/masks",
    batch_size=BATCH_SIZE
)

# -----------------------------
# Build model
# -----------------------------
model = build_attention_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_score, "accuracy"]
)

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="saved_models/best_attention_unet.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    "logs/training_log.csv"
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        csv_logger
    ]

)
# -----------------------------
# Save final model
# -----------------------------
model.save("saved_models/final_attention_unet.keras")

print("\n✅ Training Completed Successfully!")