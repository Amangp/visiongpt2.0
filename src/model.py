import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit = vit
        self.proj = nn.Linear(768, out_dim)
        for p in self.vit.parameters():
            p.requires_grad = False  # freeze backbone

    def forward(self, x):
        feats = self.vit._process_input(x)
        b, n, _ = feats.size()
        cls_token = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat((cls_token, feats), dim=1)
        x = x + self.vit.encoder.pos_embedding[:, :(n + 1)]
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.ln(self.vit.encoder.layers(x))
        return self.proj(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                               dim_feedforward=hidden_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, ques):
        x = self.embedding(ques).permute(1, 0, 2)
        x = self.encoder(x)
        x = x.mean(0)
        return self.proj(x)

class VQATransformer(nn.Module):
    def __init__(self, vocab_size, ans_size, embed_dim=300, hidden_dim=512, num_heads=4):
        super().__init__()
        self.vision = VisionEncoder(out_dim=hidden_dim)
        self.text = TextEncoder(vocab_size, embed_dim, hidden_dim, n_heads=num_heads)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, ans_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, img, ques):
        img_feat = self.vision(img)                  # [B, N, D]
        ques_feat = self.text(ques).unsqueeze(1)     # [B, 1, D]
        attn_output, _ = self.cross_attn(ques_feat, img_feat, img_feat)
        fused = F.relu(self.fc1(attn_output.squeeze(1)))
        fused = self.dropout(fused)
        return self.fc2(fused)
