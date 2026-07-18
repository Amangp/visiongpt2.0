import tensorflow as tf
from PIL import Image
import os
import pickle
import numpy as np


class VQADataset:
    def __init__(
        self,
        data_path,
        image_dir,
        word2idx,
        ans2idx,
        transform=None,
        max_len=20,
    ):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.image_dir = image_dir
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.transform = transform
        self.max_len = max_len

    def encode_question(self, question):
        words = question.split()

        ids = [
            self.word2idx.get(word, self.word2idx["<unk>"])
            for word in words
        ]

        if len(ids) < self.max_len:
            ids += [self.word2idx["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]

        return np.array(ids, dtype=np.int32)

    def encode_answer(self, answers):
        counts = {}

        for answer in answers:
            idx = self.ans2idx.get(answer, self.ans2idx["<unk>"])
            counts[idx] = counts.get(idx, 0) + 1

        return np.int32(max(counts, key=counts.get))

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(
            self.image_dir,
            f"COCO_train2014_{item['image_id']:012d}.jpg",
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = image.resize((224, 224))
            image = np.array(image).astype(np.float32) / 255.0

        question = self.encode_question(item["question"])
        answer = self.encode_answer(item["answers"])

        return image, question, answer

    def __len__(self):
        return len(self.data)

    def generator(self):
        for i in range(len(self)):
            yield self[i]

    def get_tf_dataset(
        self,
        batch_size=32,
        shuffle=True,
        buffer_size=1000,
    ):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.max_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset