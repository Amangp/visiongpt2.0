import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------

class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
        )

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])

    def call(self, x, training=False):

        # Self Attention
        attn = self.attn(
            query=self.norm1(x),
            value=self.norm1(x),
            key=self.norm1(x),
            training=training,
        )

        x = x + attn

        # Feed Forward
        ff = self.mlp(
            self.norm2(x),
            training=training,
        )

        x = x + ff

        return x


# ---------------------------------------------------------
# Vision Transformer Encoder
# ---------------------------------------------------------

class VisionTransformerEncoder(tf.keras.Model):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        depth=14,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        # Patch Embedding
        self.patch_embed = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=True,
        )

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        # Learnable CLS Token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, dim),
            initializer="zeros",
            trainable=True,
        )

        # Learnable Position Embeddings
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_patches + 1, dim),
            initializer="zeros",
            trainable=True,
        )

        # Transformer Encoder
        self.encoder = [
            TransformerEncoderBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.proj = layers.Dense(dim)

    def call(self, x, training=False):
        """
        x : [B,H,W,3]
        Returns:
            [B,197,768]
        """

        # Patch Embedding
        # [B,14,14,768]
        x = self.patch_embed(x)

        batch = tf.shape(x)[0]

        # Flatten patches
        # [B,196,768]
        x = tf.reshape(
            x,
            [
                batch,
                self.num_patches,
                self.dim,
            ],
        )

        # CLS Token
        cls = tf.broadcast_to(
            self.cls_token,
            [batch, 1, self.dim],
        )

        x = tf.concat(
            [cls, x],
            axis=1,
        )

        # Position Embedding
        x = x + self.pos_embed[:, : tf.shape(x)[1], :]

        # Transformer Layers
        for layer in self.encoder:
            x = layer(
                x,
                training=training,
            )

        x = self.norm(x)

        x = self.proj(x)

        return x
    
# ---------------------------------------------------------
# Text Transformer Encoder
# ---------------------------------------------------------

class TextTransformerEncoder(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        max_len=24,
        embed_dim=384,
        out_dim=768,
        depth=8,
        num_heads=12,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Learnable CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
        )

        # Token Embedding
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
        )

        # Position Embedding
        self.pos_embedding = layers.Embedding(
            input_dim=max_len + 1,
            output_dim=embed_dim,
        )

        # Transformer Encoder Stack
        self.encoder = [
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        # Final projection
        self.proj = layers.Dense(out_dim)

    def call(
        self,
        token_ids,
        training=False,
    ):
        """
        token_ids : [B,L]

        Returns:
            [B,L+1,out_dim]
        """

        token_ids = tf.cast(token_ids, tf.int32)

        batch = tf.shape(token_ids)[0]
        length = tf.shape(token_ids)[1]

        # Truncate if necessary
        token_ids = token_ids[:, : self.max_len]

        length = tf.shape(token_ids)[1]

        # Word Embeddings
        x = self.embedding(token_ids)

        # CLS Token
        cls = tf.broadcast_to(
            self.cls_token,
            [batch, 1, self.embed_dim],
        )

        x = tf.concat(
            [cls, x],
            axis=1,
        )

        # Position IDs
        pos_ids = tf.range(length + 1)
        pos_ids = tf.expand_dims(pos_ids, 0)
        pos_ids = tf.tile(pos_ids, [batch, 1])

        # Position Embeddings
        x = x + self.pos_embedding(pos_ids)

        # Transformer Encoder
        for block in self.encoder:
            x = block(
                x,
                training=training,
            )

        # Projection to hidden dimension
        x = self.proj(x)

        return x
# ---------------------------------------------------------
# Feed Forward Network
# ---------------------------------------------------------

class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.Activation(tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])

    def call(
        self,
        x,
        training=False,
    ):
        return self.net(
            x,
            training=training,
        )


# ---------------------------------------------------------
# Bidirectional Fusion Layer
# ---------------------------------------------------------

class FusionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        # LayerNorms
        self.q_norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.v_norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.q_norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.v_norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        # Cross Attention
        self.q_to_v = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
        )

        self.v_to_q = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
        )

        # Feed Forward Networks
        self.q_ff = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

        self.v_ff = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def call(
        self,
        q,
        v,
        training=False,
    ):
        """
        q : [B,Tq,D]
        v : [B,Tv,D]

        Returns
        -------
        q : [B,Tq,D]
        v : [B,Tv,D]
        """

        # -----------------------------
        # First LayerNorm
        # -----------------------------
        q1 = self.q_norm1(q)
        v1 = self.v_norm1(v)

        # -----------------------------
        # Question attends to Vision
        # -----------------------------
        q2 = self.q_to_v(
            query=q1,
            key=v1,
            value=v1,
            training=training,
        )

        # -----------------------------
        # Vision attends to Question
        # -----------------------------
        v2 = self.v_to_q(
            query=v1,
            key=q1,
            value=q1,
            training=training,
        )

        # -----------------------------
        # Residual
        # -----------------------------
        q = q + q2
        v = v + v2

        # -----------------------------
        # Second LayerNorm
        # -----------------------------
        q3 = self.q_norm2(q)
        v3 = self.v_norm2(v)

        # -----------------------------
        # Feed Forward
        # -----------------------------
        q = q + self.q_ff(
            q3,
            training=training,
        )

        v = v + self.v_ff(
            v3,
            training=training,
        )

        return q, v
# ---------------------------------------------------------
# Complete VQA Transformer Fusion Model
# ---------------------------------------------------------

class VQATransformerFusion(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        ans_size,
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
        max_question_len=24,
    ):
        super().__init__()

        self.max_question_len = max_question_len

        # -------------------------------------------------
        # Vision Encoder
        # -------------------------------------------------
        self.vision = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            depth=vision_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
        )

        # -------------------------------------------------
        # Text Encoder
        # -------------------------------------------------
        self.text = TextTransformerEncoder(
            vocab_size=vocab_size,
            max_len=max_question_len,
            embed_dim=embed_dim,
            out_dim=hidden_dim,
            depth=text_layers,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=dropout,
        )

        # -------------------------------------------------
        # Fusion Blocks
        # -------------------------------------------------
        self.fusion_layers = [
            FusionLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=fusion_mlp_ratio,
                dropout=dropout,
            )
            for _ in range(fusion_layers)
        ]

        # -------------------------------------------------
        # Pooling Head
        # -------------------------------------------------
        self.fusion_norm = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.fusion_head = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2),
            layers.Activation(tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim),
        ])

        # -------------------------------------------------
        # Gating
        # -------------------------------------------------
        self.gate = layers.Dense(hidden_dim)

        self.joint_proj = layers.Dense(hidden_dim)

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------
        self.classifier = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim),
            layers.Activation(tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(ans_size),
        ])

    # -----------------------------------------------------
    # Token Pooling
    # -----------------------------------------------------

    def pool_tokens(self, x):
        """
        x : [B,T,D]

        Returns
        -------
        cls
        mean
        max
        """

        cls = x[:, 0]

        if x.shape[1] > 1:
            toks = x[:, 1:]
        else:
            toks = x

        mean = tf.reduce_mean(
            toks,
            axis=1,
        )

        maximum = tf.reduce_max(
            toks,
            axis=1,
        )

        return cls, mean, maximum

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def call(
        self,
        image,
        question,
        training=False,
    ):

        # -----------------------------
        # Encoders
        # -----------------------------

        v = self.vision(
            image,
            training=training,
        )

        q = self.text(
            question,
            training=training,
        )

        # -----------------------------
        # Fusion
        # -----------------------------

        for layer in self.fusion_layers:
            q, v = layer(
                q,
                v,
                training=training,
            )

        # -----------------------------
        # Pool
        # -----------------------------

        q_cls, q_mean, q_max = self.pool_tokens(q)

        v_cls, v_mean, v_max = self.pool_tokens(v)

        # -----------------------------
        # Statistics Vector
        # -----------------------------

        stats = tf.concat([
            q_cls,
            v_cls,
            q_mean,
            v_mean,
            q_max,
            v_max,
        ], axis=-1)

        stats = self.fusion_norm(stats)

        fusion_vec = self.fusion_head(
            stats,
            training=training,
        )

        # -----------------------------
        # Joint Pair
        # -----------------------------

        pair = tf.concat(
            [
                q_cls,
                v_cls,
            ],
            axis=-1,
        )

        gate = tf.sigmoid(
            self.gate(pair)
        )

        joint = self.joint_proj(pair)

        fused = (
            gate * joint
            +
            (1.0 - gate) * fusion_vec
        )

        logits = self.classifier(
            fused,
            training=training,
        )

        return logits