import os
import math
import tensorflow as tf
from tqdm import tqdm

IMAGE_SIZE = (224, 224)
SHARD_SIZE = 5000


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(value)])
    )


def serialize_example(image, question, answer):

    feature = {
        "image": _bytes_feature(tf.io.serialize_tensor(image).numpy()),
        "question": _bytes_feature(tf.io.serialize_tensor(question).numpy()),
        "answer": _int64_feature(answer),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )

    return example.SerializeToString()


def create_tfrecords(
    data,
    image_dir,
    output_dir,
    word2idx,
    ans2idx,
    max_len=20,
):

    os.makedirs(output_dir, exist_ok=True)

    split = os.path.basename(os.path.normpath(image_dir))

    num_shards = math.ceil(len(data) / SHARD_SIZE)

    for shard in range(num_shards):

        start = shard * SHARD_SIZE
        end = min((shard + 1) * SHARD_SIZE, len(data))

        filename = os.path.join(
            output_dir,
            f"{split}-{shard:05d}-of-{num_shards:05d}.tfrecord",
        )

        with tf.io.TFRecordWriter(filename) as writer:

            for item in tqdm(
                data[start:end],
                desc=f"Shard {shard+1}/{num_shards}",
            ):

                image_path = os.path.join(
                    image_dir,
                    f"COCO_{split}_{item['image_id']:012d}.jpg",
                )

                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, IMAGE_SIZE)
                image = tf.cast(image, tf.float32)

                image = tf.keras.applications.efficientnet.preprocess_input(
                    image
                )

                words = item["question"].split()

                question = [
                    word2idx.get(
                        w,
                        word2idx["<unk>"],
                    )
                    for w in words
                ]

                if len(question) < max_len:

                    question += [
                        word2idx["<pad>"]
                    ] * (max_len - len(question))

                else:

                    question = question[:max_len]

                counts = {}

                for ans in item["answers"]:

                    idx = ans2idx.get(
                        ans,
                        ans2idx["<unk>"],
                    )

                    counts[idx] = counts.get(idx, 0) + 1

                answer = max(
                    counts,
                    key=counts.get,
                )

                example = serialize_example(
                    image,
                    tf.constant(question, dtype=tf.int32),
                    answer,
                )

                writer.write(example)

    print(f"\n✅ TFRecords written to {output_dir}")