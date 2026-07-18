import pickle
import yaml
import tensorflow as tf
from tqdm import tqdm

from src.data_loader import VQADataset
from src.model import VQATransformer


def train():

    # -----------------------------------------------------
    # Load Config
    # -----------------------------------------------------

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("🖥 Using device:",
          "GPU" if tf.config.list_physical_devices("GPU") else "CPU")

    # -----------------------------------------------------
    # Load Vocab
    # -----------------------------------------------------

    with open("data/processed/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("data/processed/ans2idx.pkl", "rb") as f:
        ans2idx = pickle.load(f)

    # -----------------------------------------------------
    # Image Transform
    # -----------------------------------------------------

    def transform(image):

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        image = tf.image.resize(image, (224, 224))

        image = image / 255.0

        image = tf.keras.applications.imagenet_utils.preprocess_input(
            image * 255.0,
            mode="torch",
        )

        return image

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = VQADataset(
        "data/processed/train_data.pkl",
        config["data"]["image_dir"],
        word2idx,
        ans2idx,
        transform=transform,
    )

    loader = dataset.get_tf_dataset(
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = VQATransformer(
        vocab_size=len(word2idx),
        ans_size=len(ans2idx),
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
    )

    # Build model

    dummy_img = tf.random.normal((1, 224, 224, 3))

    dummy_q = tf.zeros(
        (
            1,
            config["data"]["max_question_len"],
        ),
        dtype=tf.int32,
    )

    model(dummy_img, dummy_q)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config["training"]["lr"]
    )

    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["training"]["lr"],
        decay_steps=len(dataset),
        decay_rate=config["training"]["lr_decay"],
        staircase=False,
    )

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for epoch in range(config["training"]["epochs"]):

        total_loss = 0.0

        total = 0

        correct = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}",
        )

        for images, questions, answers in progress:

            optimizer.learning_rate = scheduler(
                optimizer.iterations
            )

            with tf.GradientTape() as tape:

                logits = model(
                    images,
                    questions,
                    training=True,
                )

                loss = loss_fn(
                    answers,
                    logits,
                )

            gradients = tape.gradient(
                loss,
                model.trainable_variables,
            )

            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables,
                )
            )

            total_loss += float(loss)

            pred = tf.argmax(
                logits,
                axis=1,
                output_type=tf.int32,
            )

            correct += int(
                tf.reduce_sum(
                    tf.cast(
                        pred == answers,
                        tf.int32,
                    )
                )
            )

            total += int(answers.shape[0])

        accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch+1}: "
            f"Loss={total_loss/len(loader):.4f}, "
            f"Acc={accuracy:.2f}%"
        )

    # -----------------------------------------------------
    # Save Model
    # -----------------------------------------------------

    model.save_weights(
        "vqa_transformer.keras"
    )

    print("✅ Saved model as vqa_transformer.keras")


if __name__ == "__main__":
    train()