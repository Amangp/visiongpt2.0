import tensorflow as tf
import pickle
import numpy as np


class VQADataset:

    def __init__(
        self,
        tfrecord_pattern,
        word2idx,
        ans2idx,
        batch_size=32,
        shuffle=True,
    ):

        self.tfrecord_pattern = tfrecord_pattern
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.batch_size = batch_size
        self.shuffle = shuffle

    def parse_example(self, example):

        feature_description = {

            "image": tf.io.FixedLenFeature([], tf.string),

            "question": tf.io.FixedLenFeature([], tf.string),

            "answer": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(
            example,
            feature_description,
        )

        image = tf.io.parse_tensor(
            example["image"],
            out_type=tf.float32,
        )

        image.set_shape((224, 224, 3))

        question = tf.io.parse_tensor(
            example["question"],
            out_type=tf.int32,
        )

        question.set_shape((20,))

        answer = tf.cast(
            example["answer"],
            tf.int32,
        )

        return image, question, answer

    def get_tf_dataset(self):

        files = tf.io.gfile.glob(
            self.tfrecord_pattern
        )

        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        dataset = dataset.map(
            self.parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.shuffle:

            dataset = dataset.shuffle(
                8192,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=False,
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE,
        )

        options = tf.data.Options()

        options.experimental_deterministic = False

        dataset = dataset.with_options(
            options
        )

        return dataset