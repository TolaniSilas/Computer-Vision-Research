"""
Vision Transformer (ViT) architecture Implementation from paper "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).
https://arxiv.org/pdf/2010.11929
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from components import TransformerEncoder


class MLP_Head(nn.Module):
    """mlp head for classification after the transformer encoder."""

    def __init__(self, hidden_size, num_classes):
        """initializes a single linear projection layer."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # linear projection to num_classes.
        self.head_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        """projects cls token representation to class logits."""

        return self.head_layer(x)


class VisionTransformer(nn.Module):
    """vision transformer (vit) architecture."""

    def __init__(self, config, img_size=224, num_classes=5, zero_head=False):
        """initializes the vit with transformer encoder and mlp classification head."""
        super().__init__()

        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = TransformerEncoder()
        self.mlp_head = MLP_Head(config.hidden_size, self.num_classes)


    def forward(self, x, labels=None):
        """encodes input and returns loss if labels provided, else logits."""

        # encode input through transformer, extract cls token.
        x = self.transformer(x)
        logits = self.mlp_head(x[:, 0])

        if labels is not None:
            # compute cross entropy loss.
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            return loss

        return logits






