"""
ViT building blocks: PatchEmbed (Embedded Patches), Transformer Encoder, Multi-Head Attention, MLP.

Reference: "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).
https://arxiv.org/pdf/2010.11929 
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """multi-layer perceptron for the vision transformer (ViT)."""

    def __init__(self, input_size, hidden_size, output_size):
        """initializes the mlp with two linear layers."""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # two-layer mlp with gelu activation.
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Dropout(p=0.5),
        )


    def forward(self, x):
        """applies the multi-layer perceptron to input."""

        return self.layers(x)



# Patch Embedding.
class PatchEmbed(nn.Module):
    """embedding patches for the vision transformer (ViT)."""

    def __init__(self):
        """initializes the patch embedding with a linear layer."""
        super().__init__()

        






# Implement Multi-Head Attention (MHA).











# Implement Transformer Encoder.
