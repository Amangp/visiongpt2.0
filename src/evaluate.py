import pickle
import yaml
import tensorflow as tf

from src.model import VQAModel


def evaluate():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Select device
    gpus = tf.config.list_physical_devices("GPU")
    device = "/GPU:0" if len(gpus) > 0 else "/CPU:0"

    # Load vocabularies
    with open("data/processed/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("data/processed/ans2idx.pkl", "rb") as f:
        ans2idx = pickle.load(f)

    # Create model
    model = VQAModel(
        vocab_size=len(word2idx),
        ans_size=len(ans2idx),
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        image_feat_dim=config["model"]["image_feat_dim"],
        attention_dim=config["model"]["attention_dim"],
    )

    # Build model
    dummy_image = tf.random.normal(
        (1, 224, 224, 3),
        dtype=tf.float32,
    )

    dummy_question = tf.zeros(
        (1, config["data"]["max_question_len"]),
        dtype=tf.int32,
    )

    model(dummy_image, dummy_question, training=False)

    # Load trained TensorFlow weights
    model.load_weights("vqa_model.keras")

    print("✅ TensorFlow model loaded successfully.")
    print(f"🖥 Using device: {device}")


if __name__ == "__main__":
    evaluate()