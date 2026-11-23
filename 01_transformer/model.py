import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout_rate=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # compute positional embeddings
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


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

        scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

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

    def __init__(self, d_model, num_heads, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout_rate)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(x, x, x, mask)
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

    def __init__(self, d_model, num_heads, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.masked_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.masked_attn_norm = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = x + self.masked_attn(x, x, x, mask=tgt_mask)
        x = self.masked_attn_norm(x)

        x = x + self.attn(x, memory, memory, mask=src_mask)
        x = self.attn_norm(x)

        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x


class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, num_decoders):
        super().__init__()

        self.dec_layers = nn.ModuleList(DecoderLayer(
            d_model, num_heads) for _ in range(num_decoders))

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class Transfromer(nn.Module):

    def __init__(self, d_model=512, num_heads=8, num_encoders=6, num_decoders=6, vocab_size=10000):
        super().__init__()

        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = PositionalEmbedding(d_model)

        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)

        self.generator = nn.Linear(d_model, vocab_size)

    def create_pad_mask(self, x, pad_idx):
        x = (x != pad_idx).unsqueeze(-2)  # (batch_size, 1, seq_len)
        return x

    def create_subsquent_mask(self, x):
        seq_len = x.size(1)
        subsequent_mask = np.triu(
            np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0).to(x.device)
        return subsequent_mask

    def forward(self, src, tgt, src_pad_token=0, tgt_pad_token=0):
        # create masks
        src_mask = self.create_pad_mask(
            src, src_pad_token)  # (batch_size, 1, seq_len)
        tgt_mask = self.create_pad_mask(
            # (batch_size, seq_len, seq_len)
            tgt, tgt_pad_token) & self.create_subsquent_mask(tgt)

        # embedding with scaling
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        # positional encoding
        src_emb = self.pos_embedding(src_emb)
        tgt_emb = self.pos_embedding(tgt_emb)

        # encoder
        enc_output = self.encoder(src_emb, src_mask)

        # decoder
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        # output projection
        output = self.generator(dec_output)

        return output


if __name__ == '__main__':
    model = Transfromer()
    print(model)
