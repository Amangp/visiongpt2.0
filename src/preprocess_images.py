import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

SIZE = (224, 224)


def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    for file in tqdm(files):
        img = tf.io.read_file(os.path.join(input_dir, file))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, SIZE)

        img = tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(img, tf.float32),
            mode="torch",
        )

        np.save(
            os.path.join(output_dir, file.replace(".jpg", ".npy")),
            img.numpy(),
        )


preprocess_folder(
    "data/images/train2014",
    "data/processed_images/train2014",
)

preprocess_folder(
    "data/images/val2014",
    "data/processed_images/val2014",
)