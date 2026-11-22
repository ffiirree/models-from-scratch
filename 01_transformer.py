import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, dropout_rate=0.1):
        super().__init__()

        self.d_k = d_k

        self.Wq = nn.Linear(d_model, d_k)
        self.Wk = nn.Linear(d_model, d_k)
        self.Wv = nn.Linear(d_model, d_k)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        q = self.Wq(q)
        k = self.Wq(k)
        v = self.Wq(v)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        probs = F.softmax(scores, dim=-1)

        return torch.matmul(self.dropout(probs), v)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()

        self.d_k = d_model // num_heads

        self.attentions = nn.ModuleList(ScaledDotProductAttention(
            d_model, self.d_k, dropout_rate) for _ in range(num_heads))

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        torch.cat([attn(q, k, v, mask) for attn in self.attentions], dim=-1)
        x = self.proj(x)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout_rate)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(x)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, num_encoders):
        super().__init__()

        self.enc_layers = nn.ModuleList(EncoderLayer(
            d_model, num_heads) for _ in range(num_encoders))

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()


class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, num_decoders):
        pass

    def forward(self, tgt, enc, tgt_mask, enc_mask):
        pass


class Transfromer(nn.Module):

    def __init__(self, d_model=512, num_heads=8, num_encoders=6, num_decoders=6, vocab_size=10000):
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)

        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = PositionEmbedding(d_model)

        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self):
        pass
