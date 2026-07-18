import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 14,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=None)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed[:, : x.shape[1]]

        x = self.encoder(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 24,
        embed_dim: int = 384,
        out_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=None)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        b, l = token_ids.shape
        if l > self.max_len:
            token_ids = token_ids[:, : self.max_len]
            l = self.max_len

        x = self.embedding(token_ids)  # [B, L, E]
        pos_ids = torch.arange(l + 1, device=token_ids.device).unsqueeze(0).expand(b, -1)

        cls = self.cls_token.expand(b, -1, -1)  # [B, 1, E]
        x = torch.cat([cls, x], dim=1)  # [B, 1+L, E]
        x = x + self.pos_embedding(pos_ids)  # [B, 1+L, E]

        x = self.encoder(x)  # [B, 1+L, E]
        x = self.proj(x)  # [B, 1+L, D]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionLayer(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.q_norm1 = nn.LayerNorm(dim)
        self.v_norm1 = nn.LayerNorm(dim)
        self.q_norm2 = nn.LayerNorm(dim)
        self.v_norm2 = nn.LayerNorm(dim)

        self.q_to_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.v_to_q = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.q_ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
        self.v_ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q_norm1(q)
        v1 = self.v_norm1(v)

        q2, _ = self.q_to_v(q1, v1, v1, need_weights=False)
        v2, _ = self.v_to_q(v1, q1, q1, need_weights=False)

        q = q + q2
        v = v + v2

        q3 = self.q_norm2(q)
        v3 = self.v_norm2(v)

        q = q + self.q_ff(q3)
        v = v + self.v_ff(v3)

        return q, v


class VQATransformerFusion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ans_size: int,
        image_size: int = 224,
        patch_size: int = 16,
        vision_dim: int = 768,
        vision_depth: int = 14,
        embed_dim: int = 384,
        hidden_dim: int = 768,
        num_heads: int = 12,
        text_layers: int = 8,
        fusion_layers: int = 4,
        fusion_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_question_len: int = 24,
    ):
        super().__init__()

        self.max_question_len = max_question_len

        self.vision = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            depth=vision_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
        )
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

        self.fusion_layers = nn.ModuleList(
            [
                FusionLayer(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=fusion_mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(fusion_layers)
            ]
        )

        # Pooling/fusion heads
        self.fusion_norm = nn.LayerNorm(hidden_dim * 6)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim))
        self.joint_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ans_size),
        )

    @staticmethod
    def _pool_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        cls = x[:, 0]
        if x.shape[1] > 1:
            toks = x[:, 1:]
        else:
            toks = x
        mean = toks.mean(dim=1)
        maxv = toks.max(dim=1).values
        return cls, mean, maxv

    def forward(self, img: torch.Tensor, ques: torch.Tensor) -> torch.Tensor:
        v = self.vision(img)  # [B, Tv, D]
        q = self.text(ques)   # [B, Tq, D]

        for layer in self.fusion_layers:
            q, v = layer(q, v)

        q_cls, q_mean, q_max = self._pool_tokens(q)
        v_cls, v_mean, v_max = self._pool_tokens(v)

        stats = torch.cat([q_cls, v_cls, q_mean, v_mean, q_max, v_max], dim=-1)  # [B, 6D]
        stats = self.fusion_norm(stats)
        fusion_vec = self.fusion_head(stats)  # [B, D]

        pair = torch.cat([q_cls, v_cls], dim=-1)  # [B, 2D]
        gate = torch.sigmoid(self.gate(pair))
        joint = self.joint_proj(pair)

        fused = gate * joint + (1.0 - gate) * fusion_vec
        return self.classifier(fused)
