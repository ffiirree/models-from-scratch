import torch
import torch.nn as nn
import numpy as np


class VisionTransformer(nn.Module):
    """ ViT
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()

        self.visual = VisionTransformer()
        self.transformer = Transformer()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        image_features = self.visual(image)
        text_features = self.transformer(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text  # (global_batch_size, global_batch_size)
