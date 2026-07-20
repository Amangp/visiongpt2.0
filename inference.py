import tensorflow as tf
import numpy as np
import pickle
import yaml
from PIL import Image

from src.model import VQATransformer
from src.preprocess import normalize_text


class VQAInference:

    def __init__(
        self,
        model_path,
        word2idx_path,
        ans2idx_path,
    ):

        # -----------------------------
        # Load vocabularies
        # -----------------------------

        with open(word2idx_path, "rb") as f:
            self.word2idx = pickle.load(f)

        with open(ans2idx_path, "rb") as f:
            self.ans2idx = pickle.load(f)

        self.idx2ans = {
            v: k for k, v in self.ans2idx.items()
        }

        print(f"Vocabulary Size : {len(self.word2idx)}")
        print(f"Answer Classes  : {len(self.ans2idx)}")

        self.max_question_len = 20
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        # -----------------------------
        # Create Model
        # -----------------------------

        self.model = VQATransformer(
            vocab_size=len(self.word2idx),
            ans_size=len(self.ans2idx),
            embed_dim=model_cfg["embed_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["num_heads"],
            n_layers=model_cfg["n_layers"],
        )

        # -----------------------------
        # Build model
        # -----------------------------

        dummy_image = tf.random.normal((1, 224, 224, 3))
        dummy_question = tf.zeros(
            (1, self.max_question_len),
            dtype=tf.int32,
        )

        self.model(
            dummy_image,
            dummy_question,
            training=False,
        )

        # -----------------------------
        # Load weights
        # -----------------------------

        try:
            self.model.load_weights(model_path)
            print("Loaded successfully")
            total = 0
            for w in self.model.trainable_variables:
                total += tf.reduce_sum(tf.abs(w)).numpy()

            print(f"Total weight magnitude: {total:.2f}")
        except Exception as e:
            print(e)

            print("\n========================")
            for layer in self.model.layers:
                print(layer.name, type(layer))

        print("Model loaded successfully!")

    # ----------------------------------------------------
    # Image preprocessing
    # ----------------------------------------------------

    def preprocess_image(self, image):

        if isinstance(image, str):

            image = tf.io.read_file(image)
            image = tf.image.decode_jpeg(
                image,
                channels=3,
            )

        else:

            image = np.array(image)
            image = tf.convert_to_tensor(image)

        image = tf.image.convert_image_dtype(
            image,
            tf.float32,
        )

        image = tf.image.resize(
            image,
            (224, 224),
        )

        image = tf.keras.applications.efficientnet.preprocess_input(
            image * 255.0
        )

        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image

    # ----------------------------------------------------
    # Question preprocessing
    # ----------------------------------------------------

    def preprocess_question(self, question):

        question = normalize_text(question)

        words = question.split()

        words = words[:self.max_question_len]

        indices = [

            self.word2idx.get(
                w,
                self.word2idx["<unk>"],
            )

            for w in words
        ]

        while len(indices) < self.max_question_len:
            indices.append(
                self.word2idx["<pad>"]
            )

        question = tf.convert_to_tensor(
            [indices],
            dtype=tf.int32,
        )

        return question

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------

    def predict(
        self,
        image,
        question,
        top_k=5,
    ):

        image = self.preprocess_image(image)

        question = self.preprocess_question(question)

        logits = self.model(
            image,
            question,
            training=False,
        )

        probs = tf.nn.softmax(
            logits,
            axis=-1,
        )[0]

        values, indices = tf.math.top_k(
            probs,
            k=top_k,
        )

        predictions = []

        for score, idx in zip(
            values.numpy(),
            indices.numpy(),
        ):

            predictions.append(
                (
                    self.idx2ans[int(idx)],
                    float(score),
                )
            )

        return predictions


# ----------------------------------------------------
# Interactive Testing
# ----------------------------------------------------

if __name__ == "__main__":

    vqa = VQAInference(

        model_path="best_model.weights.h5",

        word2idx_path="data/processed/word2idx.pkl",

        ans2idx_path="data/processed/ans2idx.pkl",

    )

    while True:

        print("\n" + "=" * 60)

        image = input("Image Path (q to quit): ")

        if image.lower() == "q":
            break

        question = input("Question: ")

        predictions = vqa.predict(
            image,
            question,
            top_k=5,
        )

        print("\nTop Predictions\n")

        for i, (answer, confidence) in enumerate(
            predictions,
            start=1,
        ):

            print(
                f"{i}. {answer:<20} {confidence:.2%}"
            )

        print("=" * 60)