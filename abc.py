import pickle
import yaml
import tensorflow as tf
from src.model import VQATransformer

with open("config.yaml") as f:
    config = yaml.safe_load(f)

with open("data/processed/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("data/processed/ans2idx.pkl", "rb") as f:
    ans2idx = pickle.load(f)

model = VQATransformer(
    vocab_size=len(word2idx),
    ans_size=len(ans2idx),
    embed_dim=config["model"]["embed_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_heads=config["model"]["num_heads"],
    n_layers=config["model"]["n_layers"],
)

dummy_img = tf.zeros((1, 224, 224, 3))
dummy_q = tf.zeros((1, 20), dtype=tf.int32)

model(dummy_img, dummy_q)

model.save_weights("temp.weights.h5")

print("Saved!")

model.load_weights("temp.weights.h5")

print("Loaded!")