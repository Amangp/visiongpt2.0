import tensorflow as tf
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
        max_samples=None,
    ):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)      
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.image_dir = image_dir
        print("Image dir:", self.image_dir)
        print("Split:", os.path.basename(os.path.normpath(self.image_dir)))
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.transform = transform
        self.max_len = max_len
    def load_sample(self, image_path, question, answer):

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.efficientnet.preprocess_input(
            image * 255.0
        )

        return image, question, answer
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

        split = os.path.basename(os.path.normpath(self.image_dir))

        image_path = os.path.join(
            self.image_dir,
            f"COCO_{split}_{item['image_id']:012d}.jpg",
        )

        try:
            # Read image
            image = tf.io.read_file(image_path)

            # Decode JPEG directly
            image = tf.image.decode_jpeg(image, channels=3)

            # Convert to float32
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Resize
            image = tf.image.resize(image, (224, 224))

            # EfficientNet preprocessing
            image = tf.keras.applications.efficientnet.preprocess_input(
                image * 255.0
            )

        except Exception as e:
            print(f"\nError loading image: {image_path}")
            print(f"Reason: {e}")
            raise

        question = self.encode_question(item["question"])
        answer = self.encode_answer(item["answers"])

        return image, question, answer
    def __len__(self):
        return len(self.data)


    def get_tf_dataset(
        self,
        batch_size=32,
        shuffle=True,
    ):

        image_paths = []
        questions = []
        answers = []

        for item in self.data:

            split = os.path.basename(os.path.normpath(self.image_dir))

            image_paths.append(
                os.path.join(
                    self.image_dir,
                    f"COCO_{split}_{item['image_id']:012d}.jpg",
                )
            )

            questions.append(
                self.encode_question(item["question"])
            )

            answers.append(
                self.encode_answer(item["answers"])
            )

        image_paths = tf.constant(image_paths)

        questions = tf.constant(
            np.array(questions, dtype=np.int32)
        )

        answers = tf.constant(
            np.array(answers, dtype=np.int32)
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, questions, answers)
        )

        if shuffle:
            dataset = dataset.shuffle(
                len(image_paths),
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            self.load_sample,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Cache processed images to SSD
        cache_dir = "data/cache/train_cache" if shuffle else "data/cache/val_cache"
        dataset = dataset.cache(cache_dir)

        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset