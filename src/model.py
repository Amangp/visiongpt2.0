import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class TransformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.1
    ):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):

        attn = self.attention(
            query=x,
            key=x,
            value=x,
            training=training
        )

        attn = self.dropout1(
            attn,
            training=training
        )

        x = self.norm1(x + attn)

        ffn = self.ffn(x)

        ffn = self.dropout2(
            ffn,
            training=training
        )

        return self.norm2(x + ffn) 

class VisionEncoder(tf.keras.Model):

    def __init__(self, hidden_dim=512):
        super().__init__()

        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224,224,3)
        )

        self.backbone.trainable = False

        self.proj = layers.Dense(hidden_dim)

    def call(self, images, training=False):

        images = tf.keras.applications.efficientnet.preprocess_input(images)

        features = self.backbone(
            images,
            training=training
        )

        # (B,7,7,1280)
        B = tf.shape(features)[0]

        features = tf.reshape(
            features,
            (B,49,1280)
        )

        features = self.proj(features)

        return features
# ---------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------

class TextEncoder(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_dim=512,
        n_heads=8,
        n_layers=2,
        max_len=50,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = layers.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

        self.dropout = layers.Dropout(dropout)

        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=n_heads,
                ff_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ]

        self.projection = layers.Dense(hidden_dim)

    def call(
        self,
        questions,
        training=False
    ):

        seq_len = tf.shape(questions)[1]

        positions = tf.range(seq_len)
        positions = self.position_embedding(positions)

        x = self.embedding(questions)

        x = x + positions

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                training=training
            )

        x = self.projection(x)

        return x
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

        image_features = self.vision(
            images,
            training=training
        )

        question_features = self.text(
            questions,
            training=training
        )

        attended = self.cross_attention(
            query=question_features,
            key=image_features,
            value=image_features,
            training=training
        )

        # Pool over question tokens
        fused = tf.reduce_mean(attended, axis=1)

        fused = self.fc1(fused)

        fused = self.dropout(
            fused,
            training=training
        )

        logits = self.fc2(fused)

        return logits