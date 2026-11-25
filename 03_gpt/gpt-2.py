import torch
import torch.nn as nn


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout_rate

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.size()  # (batch_size, seq_len, embed_dim)

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (batch_size, num_heads, seq_len, embed_dim // num_heads)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=True
        )

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.dropout(self.proj(x))
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.csa_norm = nn.LayerNorm(embed_dim)
        self.csa = CausalMultiHeadSelfAttention(embed_dim, num_heads, dropout_rate=dropout_rate, bias=bias)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = x + self.csa(self.csa_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GenerativePreTraining(nn.Module):

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        vocab_size: int = 50257,
        context_len: int = 1024,
        dropout_rate: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList(TransformerBlock(
            embed_dim, num_heads, dropout_rate, bias) for _ in range(num_layers))

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # weight-tying
        self.tok_embedding.weight = self.lm_head.weight

    def forward(self, x):
        """ x: (batch_size, seq_len)"""
        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(x.shape[1], device=x.device))

        x = self.dropout(tok_emb + pos_emb)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.norm(x)
        x = self.lm_head(x)
        return x


if __name__ == '__main__':
    model = GenerativePreTraining(vocab_size=1000,)
    x = torch.randint(0, 1000, (2, 32))
    y = model(x)
    print(model)
    print(f'input: {x.shape}, output: {y.shape}')
