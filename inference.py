import tensorflow as tf
import numpy as np
import pickle

from PIL import Image

from src.model import VQATransformer
from src.model_fusion import VQATransformerFusion
from src.preprocess import normalize_text


class VQAInference:

    def __init__(
        self,
        model_path,
        word2idx_path,
        ans2idx_path,
        model_type="fusion",
    ):

        # -------------------------------------------------
        # Load vocabularies
        # -------------------------------------------------

        with open(word2idx_path, "rb") as f:
            self.word2idx = pickle.load(f)

        with open(ans2idx_path, "rb") as f:
            self.ans2idx = pickle.load(f)

        self.idx2ans = {
            v: k
            for k, v in self.ans2idx.items()
        }

        vocab_size = len(self.word2idx)
        ans_size = len(self.ans2idx)

        print(f"📊 Vocabulary Size : {vocab_size}")
        print(f"📊 Answer Classes  : {ans_size}")

        # -------------------------------------------------
        # Create Model
        # -------------------------------------------------

        if model_type.lower() == "fusion":

            self.max_question_len = 24

            self.model = VQATransformerFusion(
                vocab_size=vocab_size,
                ans_size=ans_size,
                image_size=224,
                patch_size=16,
                vision_dim=768,
                vision_depth=14,
                embed_dim=384,
                hidden_dim=768,
                num_heads=12,
                text_layers=8,
                fusion_layers=4,
                fusion_mlp_ratio=4.0,
                dropout=0.0,
                max_question_len=self.max_question_len,
            )

        else:

            self.max_question_len = 20

            self.model = VQATransformer(
                vocab_size=vocab_size,
                ans_size=ans_size,
                embed_dim=256,
                hidden_dim=512,
                num_heads=8,
            )

        # -------------------------------------------------
        # Build Model
        # -------------------------------------------------

        dummy_image = tf.random.normal(
            (
                1,
                224,
                224,
                3,
            )
        )

        dummy_question = tf.zeros(
            (
                1,
                self.max_question_len,
            ),
            dtype=tf.int32,
        )

        _ = self.model(
            dummy_image,
            dummy_question,
            training=False,
        )

        # -------------------------------------------------
        # Load TensorFlow Weights
        # -------------------------------------------------

        self.model.load_weights(model_path)

        print("✅ TensorFlow model loaded successfully.")

        # -------------------------------------------------
        # ImageNet normalization
        # -------------------------------------------------

        self.mean = tf.constant(
            [0.485, 0.456, 0.406],
            dtype=tf.float32,
        )

        self.std = tf.constant(
            [0.229, 0.224, 0.225],
            dtype=tf.float32,
        )
    # -------------------------------------------------
    # Image Preprocessing
    # -------------------------------------------------

    def preprocess_image(self, image):
        """
        Convert a PIL image into a TensorFlow tensor.

        Input:
            PIL.Image or image path

        Output:
            Tensor of shape [1, 224, 224, 3]
        """

        # Load image if a path is provided
        if isinstance(image, str):
            image = Image.open(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        image = image.resize((224, 224))

        # PIL -> NumPy
        image = np.array(image).astype(np.float32)

        # Scale to [0,1]
        image /= 255.0

        # Normalize using ImageNet statistics
        image = (image - self.mean.numpy()) / self.std.numpy()

        # NumPy -> Tensor
        image = tf.convert_to_tensor(
            image,
            dtype=tf.float32,
        )

        # Add batch dimension
        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image
    # -------------------------------------------------
    # Question Preprocessing
    # -------------------------------------------------

    def preprocess_question(
        self,
        question,
        max_len=None,
    ):
        """
        Convert a natural language question into a TensorFlow tensor.

        Input:
            "What color is the car?"

        Output:
            Tensor of shape [1, max_len]
        """

        if max_len is None:
            max_len = self.max_question_len

        # Normalize question
        question = normalize_text(question)

        # Split into words
        words = question.split()

        # Truncate
        words = words[:max_len]

        # Convert words to indices
        indices = [
            self.word2idx.get(
                word,
                self.word2idx["<unk>"],
            )
            for word in words
        ]

        # Pad if necessary
        if len(indices) < max_len:
            indices.extend(
                [
                    self.word2idx["<pad>"]
                ]
                * (max_len - len(indices))
            )

        # Convert to Tensor
        question_tensor = tf.convert_to_tensor(
            indices,
            dtype=tf.int32,
        )

        # Add batch dimension
        question_tensor = tf.expand_dims(
            question_tensor,
            axis=0,
        )

        return question_tensor
    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(
        self,
        image,
        question,
        top_k=5,
    ):
        """
        Predict answers for an image-question pair.

        Args:
            image : PIL Image or image path
            question : str
            top_k : number of predictions

        Returns:
            List[(answer, confidence)]
        """

        # -----------------------------
        # Preprocess
        # -----------------------------

        image_tensor = self.preprocess_image(image)

        question_tensor = self.preprocess_question(question)

        # -----------------------------
        # Forward Pass
        # -----------------------------

        logits = self.model(
            image_tensor,
            question_tensor,
            training=False,
        )

        # -----------------------------
        # Softmax
        # -----------------------------

        probs = tf.nn.softmax(
            logits,
            axis=-1,
        )

        probs = tf.squeeze(
            probs,
            axis=0,
        )

        # -----------------------------
        # Top-K
        # -----------------------------

        top_k = min(
            top_k,
            probs.shape[0],
        )

        values, indices = tf.math.top_k(
            probs,
            k=top_k,
            sorted=True,
        )

        values = values.numpy()
        indices = indices.numpy()

        # -----------------------------
        # Decode Answers
        # -----------------------------

        predictions = []

        for score, idx in zip(values, indices):

            answer = self.idx2ans.get(
                int(idx),
                "<unk>",
            )

            predictions.append(
                (
                    answer,
                    float(score),
                )
            )

        return predictions


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":

    vqa = VQAInference(

        model_path="vqa_transformer.keras",

        word2idx_path="word2idx.pkl",

        ans2idx_path="ans2idx.pkl",

        model_type="fusion",

    )

    # Example usage

    # image = "sample.jpg"
    # question = "What color is the car?"

    # predictions = vqa.predict(
    #     image,
    #     question,
    # )

    # print(question)

    # for answer, confidence in predictions:
    #     print(
    #         answer,
    #         f"{confidence:.2%}"
    #     )