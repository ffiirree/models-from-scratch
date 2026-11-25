import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """ x: (B, C, H, W)
        """
        x = self.proj(x)  # (batch_size, embed_dim, H/P, W/P)
        x = torch.flatten(x, start_dim=2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)

        return x


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.msa_norm = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True)
        self.msa_drop = nn.Dropout(dropout_rate)

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(embed_dim*mlp_ratio), embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        inputs = x
        x = self.msa_norm(x)
        x, _ = self.msa(x, x, x, need_weights=False)
        x = self.msa_drop(x)
        x = x + inputs

        x = x + self.mlp(self.mlp_norm(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size=224,
        in_channels=3,
        num_classes=1000,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_blocks=12,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        classifier: str = 'token'
    ):
        super().__init__()

        self.classifier = classifier

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embedding.num_patches

        # extra learnable [class] embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # learnable 1D position embedding
        self.pos_embedding = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=[1, num_patches + 1, embed_dim]))

        self.drop = nn.Dropout(dropout_rate)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            embed_dim, num_heads, mlp_ratio, dropout_rate=dropout_rate, attention_dropout_rate=dropout_rate) for _ in range(num_blocks)])

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """ x: (B, C, H, W)
        """

        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # position embedding
        x = x + self.pos_embedding  # (batch_size, num_patches + 1, embed_dim)

        x = self.drop(x)

        for block in self.encoder_blocks:
            x = block(x)  # (batch_size, num_patches + 1, embed_dim)

        x = self.norm(x)

        x = x[:, 0] if self.classifier == 'token' else x[:, 1:].mean(dim=1)
        return self.mlp_head(x)


if __name__ == '__main__':
    model = VisionTransformer(image_size=224, patch_size=16, num_classes=10)
    image = torch.randn(2, 3, 224, 224)
    logits = model(image)

    print(model)
    print(f'input: {image.shape}, output: {logits.shape}')
