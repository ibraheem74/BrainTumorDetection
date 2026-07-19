import os
import math
import random
import tensorflow as tf

from utils.preprocessing import preprocess_image, preprocess_mask


class BrainTumorDataset(tf.keras.utils.Sequence):

    def __init__(self, image_dir, mask_dir, batch_size=8):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size

        self.images = sorted([
            file for file in os.listdir(image_dir)
            if file.endswith(".tif")
        ])

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.images))

        batch_images = self.images[start:end]

        X = []
        Y = []

        for image_name in batch_images:

            mask_name = image_name.replace(".tif", "_mask.tif")

            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = preprocess_image(image_path)
            mask = preprocess_mask(mask_path)

            X.append(image)
            Y.append(mask)

        return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)

    # 👇 Add this method here
    def on_epoch_end(self):
        random.shuffle(self.images)