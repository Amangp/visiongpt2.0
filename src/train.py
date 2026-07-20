import pickle
import yaml
import tensorflow as tf
from tqdm import tqdm
import math
from src.data_loader import VQADataset
from src.model import VQATransformer
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)
mixed_precision.set_global_policy("mixed_float16")
@tf.function(reduce_retracing=True)
def train_step(
    model,
    images,
    questions,
    answers,
    optimizer,
    loss_fn,
):
    with tf.GradientTape() as tape:

        logits = model(
            images,
            questions,
            training=True,
        )

        loss = loss_fn(answers, logits)


    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    pred = tf.argmax(
        logits,
        axis=1,
        output_type=tf.int32,
    )

    correct = tf.reduce_sum(
        tf.cast(pred == answers, tf.int32)
    )

    return loss, correct
@tf.function(reduce_retracing=True)
def val_step(
    model,
    images,
    questions,
    answers,
    loss_fn,
):
    logits = model(
        images,
        questions,
        training=False,
    )

    loss = loss_fn(
        answers,
        logits,
    )

    pred = tf.argmax(
        logits,
        axis=1,
        output_type=tf.int32,
    )

    correct = tf.reduce_sum(
        tf.cast(pred == answers, tf.int32)
    )

    return loss, correct
def train():

    # -----------------------------------------------------
    # Load Config
    # -----------------------------------------------------
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(
        "🖥 Using device:",
        "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    )

    # -----------------------------------------------------
    # Load Vocabulary
    # -----------------------------------------------------
    with open("data/processed/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("data/processed/ans2idx.pkl", "rb") as f:
        ans2idx = pickle.load(f)

    # -----------------------------------------------------
    # Training Dataset
    # -----------------------------------------------------

    train_dataset = VQADataset(
        data_path=config["data"]["train_data"],
        image_dir=config["data"]["image_dir"],
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_len=config["data"]["max_question_len"],
        max_samples=config["data"].get("train_samples"),
    )

    train_loader = train_dataset.get_tf_dataset(
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    train_steps = math.ceil(
        config["data"]["train_samples"]
        / config["training"]["batch_size"]
    )
    # -----------------------------------------------------
    # Validation Dataset
    # -----------------------------------------------------

    val_dataset = VQADataset(
        data_path=config["data"]["val_data"],
        image_dir=config["data"]["val_image_dir"],
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_len=config["data"]["max_question_len"],
        max_samples=config["data"].get("val_samples"),
    )

    val_loader = val_dataset.get_tf_dataset(
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    val_steps = math.ceil(
        config["data"]["val_samples"]
        / config["training"]["batch_size"]
    )

    print("\n========== DATASET ==========")
    print(f"Training Samples   : {config['data']['train_samples']}")
    print(f"Validation Samples : {config['data']['val_samples']}")
    print(f"Batch Size         : {config['training']['batch_size']}")
    print(f"Train Steps        : {train_steps}")
    print(f"Validation Steps   : {val_steps}")
    print("=============================\n")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = VQATransformer(
        vocab_size=len(word2idx),
        ans_size=len(ans2idx),
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        n_layers=config["model"]["n_layers"],
    )

    dummy_img = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    dummy_q = tf.zeros(
        (1, config["data"]["max_question_len"]),
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
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["training"]["lr"],
        decay_steps=train_steps,
        decay_rate=config["training"]["lr_decay"],
        staircase=False,
    )

    base_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduler,
        clipnorm=1.0,
    )

    optimizer = mixed_precision.LossScaleOptimizer(
        base_optimizer
    )
    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------
    best_val_acc = 0.0
    for epoch in range(config["training"]["epochs"]):

        total_loss = 0.0
        total = 0
        correct = 0

        progress = tqdm(
            train_loader,
            total=train_steps,
            desc=f"Epoch {epoch+1}/{config['training']['epochs']}",
            dynamic_ncols=True,
        )

        for images, questions, answers in progress:

            
            loss, batch_correct = train_step(
                model,
                images,
                questions,
                answers,
                optimizer,
                loss_fn,
            )

            total_loss += float(loss)

            correct += int(batch_correct)

            total += int(answers.shape[0])

            if progress.n % 20 == 0:
                current_lr = float(
                    scheduler(optimizer.iterations)
                )

                progress.set_postfix(
                    loss=f"{total_loss/(progress.n+1):.4f}",
                    acc=f"{100*correct/total:.2f}%",
                    lr=f"{current_lr:.2e}",
                )

        epoch_loss = total_loss / train_steps
        epoch_acc = 100.0 * correct / total

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss : {epoch_loss:.4f}")
        print(f"Train Acc  : {epoch_acc:.2f}%")
        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_loss = 0.0
        val_total = 0
        val_correct = 0

        val_progress = tqdm(val_loader,
            total=val_steps,
            desc="Validation",
            dynamic_ncols=True,
        )

        for images, questions, answers in val_progress:

            loss, batch_correct = val_step(
                model,
                images,
                questions,
                answers,
                loss_fn,
            )

            val_loss += float(loss)
            val_correct += int(batch_correct)
            val_total += int(answers.shape[0])

            val_progress.set_postfix(
                loss=f"{val_loss/(val_progress.n+1):.4f}",
                acc=f"{100*val_correct/val_total:.2f}%"
            )

        # ← THIS PART MUST BE AFTER THE for LOOP

        val_loss /= val_steps
        val_acc = 100 * val_correct / val_total

        print(f"Val Loss : {val_loss:.4f}")
        print(f"Val Acc  : {val_acc:.2f}%")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            model.save_weights(
                "best_model.weights.h5"
            )

            print("✅ Best model updated.")

    model.save_weights("vqa_transformer.weights.h5")
    print("\n✅ Model saved as vqa_transformer.weights.h5")
    
if __name__ == "__main__":
    train()