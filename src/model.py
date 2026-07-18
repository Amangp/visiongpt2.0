import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

# ---------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------

class VisionEncoder(tf.keras.Model):

    def __init__(self, out_dim=512):
        super().__init__()

        # ViT-B16 feature extractor
        self.vit = hub.KerasLayer(
            "https://tfhub.dev/sayakpaul/vit_b16_fe/1",
            trainable=False
        )

        self.proj = layers.Dense(out_dim)

    def call(self, images):

        # shape -> [B,197,768]
        features = self.vit(images)

        features = self.proj(features)

        return features


# ---------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------

class TransformerBlock(layers.Layer):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 rate=0.1):

        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):

        attn = self.att(x, x)

        attn = self.drop1(attn, training=training)

        x = self.norm1(x + attn)

        ffn = self.ffn(x)

        ffn = self.drop2(ffn, training=training)

        return self.norm2(x + ffn)


class TextEncoder(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_dim=512,
        n_heads=4,
        n_layers=2
    ):

        super().__init__()

        self.embedding = layers.Embedding(
            vocab_size,
            embed_dim
        )

        self.blocks = [
            TransformerBlock(
                embed_dim,
                n_heads,
                hidden_dim
            )
            for _ in range(n_layers)
        ]

        self.proj = layers.Dense(hidden_dim)

    def call(self, question, training=False):

        x = self.embedding(question)

        for block in self.blocks:
            x = block(x, training)

        x = tf.reduce_mean(x, axis=1)

        return self.proj(x)


# ---------------------------------------------------------
# VQA Transformer
# ---------------------------------------------------------

class VQATransformer(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        ans_size,
        embed_dim=300,
        hidden_dim=512,
        num_heads=4
    ):

        super().__init__()

        self.vision = VisionEncoder(hidden_dim)

        self.text = TextEncoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            num_heads
        )

        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim
        )

        self.fc1 = layers.Dense(
            hidden_dim,
            activation="relu"
        )

        self.dropout = layers.Dropout(0.3)

        self.fc2 = layers.Dense(ans_size)

    def call(self,
             images,
             questions,
             training=False):

        image_features = self.vision(images)

        question_features = self.text(
            questions,
            training
        )

        question_features = tf.expand_dims(
            question_features,
            axis=1
        )

        attended = self.cross_attention(
            query=question_features,
            key=image_features,
            value=image_features
        )

        fused = tf.squeeze(attended, axis=1)

        fused = self.fc1(fused)

        fused = self.dropout(
            fused,
            training=training
        )

        logits = self.fc2(fused)

        return logits