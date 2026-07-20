import json
import re
import os
import pickle
from collections import Counter
import tensorflow as tf
from src.tfrecord_builder import create_tfrecords

# ---------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize question/answer text.

    - lowercase
    - remove punctuation
    - remove leading/trailing spaces
    """

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text


# ---------------------------------------------------------
# Load COCO VQA Data
# ---------------------------------------------------------

def load_data(
    questions_path,
    annotations_path,
):
    """
    Merge COCO question and annotation files.
    """

    with open(questions_path, "r") as fq:
        questions = json.load(fq)["questions"]

    with open(annotations_path, "r") as fa:
        annotations = json.load(fa)["annotations"]

    question_dict = {
        q["question_id"]: q
        for q in questions
    }

    merged = []

    for ann in annotations:

        question = question_dict.get(
            ann["question_id"]
        )

        if question is None:
            continue

        merged.append({

            "image_id":
                ann["image_id"],

            "question":
                normalize_text(
                    question["question"]
                ),

            "answers":
                [
                    normalize_text(a["answer"])
                    for a in ann["answers"]
                ]
        })

    return merged


# ---------------------------------------------------------
# Build Question Vocabulary
# ---------------------------------------------------------

def build_vocab(
    data,
    min_word_count=3,
):
    """
    Create word vocabulary from training questions.
    """

    counter = Counter()

    for item in data:
        counter.update(
            item["question"].split()
        )

    vocab = [

        word

        for word, count in counter.items()

        if count >= min_word_count
    ]

    word2idx = {
        word: idx + 1
        for idx, word in enumerate(vocab)
    }

    word2idx["<pad>"] = 0
    word2idx["<unk>"] = len(word2idx)

    return word2idx


# ---------------------------------------------------------
# Build Answer Vocabulary
# ---------------------------------------------------------

def build_answer_vocab(
    data,
    top_n=3000,
):
    """
    Create answer vocabulary using
    the most frequent answers.
    """

    counter = Counter()

    for item in data:
        counter.update(
            item["answers"]
        )

    most_common = counter.most_common(top_n)

    ans2idx = {

        answer: idx

        for idx, (answer, _)
        in enumerate(most_common)

    }

    ans2idx["<unk>"] = len(ans2idx)

    return ans2idx


# ---------------------------------------------------------
# Save Processed Files
# ---------------------------------------------------------

def preprocess():

    print("Loading training dataset...")

    train_data = load_data(
        "data/questions/v2_OpenEnded_mscoco_train2014_questions.json",
        "data/annotations/v2_mscoco_train2014_annotations.json",
    )

    print("Loading validation dataset...")

    val_data = load_data(
        "data/questions/v2_OpenEnded_mscoco_val2014_questions.json",
        "data/annotations/v2_mscoco_val2014_annotations.json",
    )

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save processed datasets
    # -----------------------------------------------------

    with open(
        "data/processed/train_data.pkl",
        "wb",
    ) as f:
        pickle.dump(train_data, f)

    with open(
        "data/processed/val_data.pkl",
        "wb",
    ) as f:
        pickle.dump(val_data, f)
    
    print("Creating TFRecords...")

    create_tfrecords(
        train_data,
        image_dir="data/images/train2014",
        output_dir="data/tfrecords/train",
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_len=20,
    )

    create_tfrecords(
        val_data,
        image_dir="data/images/val2014",
        output_dir="data/tfrecords/val",
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_len=20,
    )
    print("\n✅ Preprocessing complete.")
    # -----------------------------------------------------
    # Build vocabularies ONLY from training data
    # -----------------------------------------------------

    word2idx = build_vocab(train_data)

    ans2idx = build_answer_vocab(train_data)

    with open(
        "data/processed/word2idx.pkl",
        "wb",
    ) as f:
        pickle.dump(word2idx, f)

    with open(
        "data/processed/ans2idx.pkl",
        "wb",
    ) as f:
        pickle.dump(ans2idx, f)

    print("\n✅ Preprocessing complete.")
    print(f"Training Questions   : {len(train_data)}")
    print(f"Validation Questions : {len(val_data)}")
    print(f"Vocabulary Size      : {len(word2idx)}")
    print(f"Answer Classes       : {len(ans2idx)}")
# ---------------------------------------------------------

if __name__ == "__main__":
    preprocess()