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



class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism with multiple parallel attention heads for the vision transformer (ViT)."""

    def __init__(self, num_heads, head_dim):
        """initialize the multi-head attention layer."""
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = ""






class TransformerBlock(nn.Module):
    """transformer block for the vision transformer (ViT)."""

    def __init__(self):
        """initializes the transformer block with a multi-head attention layer and a mlp layer."""
        super().__init__()

        self.multi_head_attention = MultiHeadAttention()
        self.mlp = MLP()



class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights








class PatchEmbed(nn.Module):
    """embedding patches for the vision transformer (ViT)."""

    def __init__(self):
        """initializes the patch embedding with a linear layer."""
        super().__init__()






class TransformerEncoder(nn.Module):
    """transformer encoder for the vision trasnformer (ViT)."""

    def __init__(self, config, img_size):
        """initializes the transformer encoder with a transformer block (multi-head attention and mlp layers) and embedding patches layer."""
        super().__init__()

        self.patch_embed = PatchEmbed(config, img_size=img_size)
        self.transformer_blocks = TransformerBlock(config)

    
    def forward(self, x):
        """applies the transformer encoder to the input."""
        
        patch_embed_output = self.patch_embed(x)

        encoded = self.transformer_blocks(patch_embed_output)
        
        return encoded