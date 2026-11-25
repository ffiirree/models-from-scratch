import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i) = sin(pos/10000^{2i/d}) = sin(pos*exp(-log(10000) * 2i/ d))
    PE(pos, 2i+1) = cos(pos/10000^{2i/d}) = cos(pos*exp(-log(10000) * 2i/ d))
    """

    def __init__(self, embed_dim, max_len=5000, dropout_rate=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # compute positional embeddings
        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]  # (batch_size, seq_len, embed_dim)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, embed_dim, d_k, dropout_rate=0.1):
        super().__init__()

        self.d_k = d_k

        self.Wq = nn.Linear(embed_dim, d_k)
        self.Wk = nn.Linear(embed_dim, d_k)
        self.Wv = nn.Linear(embed_dim, d_k)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        probs = F.softmax(scores, dim=-1)

        return torch.matmul(self.dropout(probs), v)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        assert num_heads > 0
        assert embed_dim % num_heads == 0

        self.d_k = embed_dim // num_heads

        self.attentions = nn.ModuleList(ScaledDotProductAttention(
            embed_dim, self.d_k, dropout_rate) for _ in range(num_heads))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        x = torch.cat([attn(q, k, v, mask) for attn in self.attentions], dim=-1)
        x = self.proj(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, embed_dim, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)

        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = PositionwiseFeedForward(embed_dim, d_ff, dropout_rate)

        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        """ x: (batch_size, seq_len, embed_dim)
        """
        x = x + self.attn(x, x, x, mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x


class Encoder(nn.Module):

    def __init__(self, embed_dim, num_heads, num_encoders):
        super().__init__()

        self.enc_layers = nn.ModuleList(EncoderLayer(embed_dim, num_heads) for _ in range(num_encoders))

    def forward(self, x, mask):
        """ x: (batch_size, seq_len, embed_dim)
        """
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, d_ff=2048, dropout_rate=0.1):
        super().__init__()

        self.masked_attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.masked_attn_norm = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = PositionwiseFeedForward(embed_dim, d_ff, dropout_rate)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = x + self.masked_attn(x, x, x, mask=tgt_mask)
        x = self.masked_attn_norm(x)

        x = x + self.attn(x, memory, memory, mask=src_mask)
        x = self.attn_norm(x)

        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x


class Decoder(nn.Module):

    def __init__(self, embed_dim, num_heads, num_decoders):
        super().__init__()

        self.dec_layers = nn.ModuleList(DecoderLayer(
            embed_dim, num_heads) for _ in range(num_decoders))

    def forward(self, x, memory, src_mask, tgt_mask):
        """ x: (batch_size, seq_len, embed_dim)
        """
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):

    def __init__(
            self,
            embed_dim=512,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            vocab_size=10000
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoding = PositionalEncoding(embed_dim)

        self.encoder = Encoder(embed_dim, num_heads, num_encoders)
        self.decoder = Decoder(embed_dim, num_heads, num_decoders)

        self.generator = nn.Linear(embed_dim, vocab_size)

    def create_pad_mask(self, x, pad_idx):
        """ x:  # (batch_size, seq_len)
        """
        x = (x != pad_idx).unsqueeze(-2)   # (batch_size, 1, seq_len)
        return x

    def create_subsquent_mask(self, x):
        """ x:  # (batch_size, seq_len)
        """
        seq_len = x.size(1)
        subsequent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0).to(x.device)
        return subsequent_mask

    def forward(self, src, tgt, pad_idx=0):
        """ src: (batch_size, target_seq_len)
            tgt: (batch_size, target_seq_len)
        """

        # create masks
        src_mask = self.create_pad_mask(src, pad_idx)  # (batch_size, 1, seq_len)
        tgt_mask = self.create_pad_mask(tgt, pad_idx) & self.create_subsquent_mask(
            tgt)  # (batch_size, seq_len, seq_len)

        # embedding with scaling
        src_emb = self.src_embedding(src) * np.sqrt(self.embed_dim)  # (batch_size, seq_len, embed_dim)
        tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.embed_dim)  # (batch_size, seq_len, embed_dim)

        # positional encoding
        src_emb = self.pos_encoding(src_emb)  # (batch_size, seq_len, embed_dim)
        tgt_emb = self.pos_encoding(tgt_emb)  # (batch_size, seq_len, embed_dim)

        # encoder
        enc_output = self.encoder(src_emb, src_mask)  # (batch_size, seq_len, embed_dim)

        # decoder
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)  # (batch_size, seq_len, embed_dim)

        # output projection
        output = self.generator(dec_output)  # (batch_size, seq_len, vocab_size)

        return output


if __name__ == '__main__':
    model = Transformer()
    print(model)
