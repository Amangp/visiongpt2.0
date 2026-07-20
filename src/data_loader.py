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
        self.split = os.path.basename(os.path.normpath(image_dir))

        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.transform = transform
        self.max_len = max_len

    def load_sample(
        self,
        image_path,
        question,
        answer,
    ):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_jpeg(
            image,
            channels=3,
        )

        image = tf.image.resize(
            image,
            (224, 224),
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        image = tf.keras.applications.efficientnet.preprocess_input(
            image
        )

        return image, question, answer

    def encode_question(self, question):

        ids = [
            self.word2idx.get(
                word,
                self.word2idx["<unk>"],
            )
            for word in question.split()
        ]

        if len(ids) < self.max_len:

            ids += [
                self.word2idx["<pad>"]
            ] * (
                self.max_len - len(ids)
            )

        else:

            ids = ids[: self.max_len]

        return np.array(
            ids,
            dtype=np.int32,
        )

    def encode_answer(
        self,
        answers,
    ):

        counts = {}

        for ans in answers:

            idx = self.ans2idx.get(
                ans,
                self.ans2idx["<unk>"],
            )

            counts[idx] = counts.get(idx, 0) + 1

        return np.int32(
            max(
                counts,
                key=counts.get,
            )
        )

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

            image_paths.append(

                os.path.join(

                    self.image_dir,

                    f"COCO_{self.split}_{item['image_id']:012d}.jpg",
                )
            )

            questions.append(
                self.encode_question(
                    item["question"]
                )
            )

            answers.append(
                self.encode_answer(
                    item["answers"]
                )
            )

        dataset = tf.data.Dataset.from_tensor_slices(

            (

                tf.constant(image_paths),

                tf.constant(
                    questions,
                    dtype=tf.int32,
                ),

                tf.constant(
                    answers,
                    dtype=tf.int32,
                ),
            )
        )

        if shuffle:

            dataset = dataset.shuffle(

                min(
                    len(image_paths),
                    20000,
                ),

                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            self.load_sample,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(
            batch_size,
            drop_remainder=False,
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        options = tf.data.Options()

        options.experimental_deterministic = False
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_parallelization = True

        dataset = dataset.with_options(options)

        return dataset