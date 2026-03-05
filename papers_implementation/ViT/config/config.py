"""
configuration for Vision Transformer (ViT)
Reference: "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).
https://arxiv.org/pdf/2010.11929 
"""

class ViTConfig:
    """configuration class for vision transformer (vit) hyperparameters."""

    def __init__(
        self,
        patch_size=16,        # patch size.
        embed_dim=768,       # embedding dimension.
        hidden_size=768,     # hidden size of the transformer.
        num_layers=12,       # number of transformer layers.
        num_heads=12,        # number of attention heads.
        drop_out=0.2,        # dropout rate.
        qkv_bias=False       # query-key-value bias.
    ):
        """initializes vit configuration with hyperparameters."""

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_out = drop_out
        self.qkv_bias = qkv_bias

    def __getitem__(self, key):
        """allows dictionary-style access."""

        return getattr(self, key)

    def __setitem__(self, key, value):
        """allows dictionary-style setting."""

        setattr(self, key, value)



# predefined configurations for different vit variants.
vit_testing = ViTConfig(
    patch_size=16,
    embed_dim=192,
    hidden_size=192,
    num_layers=4,
    num_heads=3,
    drop_out=0.1,
    qkv_bias=False
)

vit_base = ViTConfig(
    patch_size=16,
    embed_dim=768,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    drop_out=0.1,
    qkv_bias=False
)

vit_large = ViTConfig(
    patch_size=16,
    embed_dim=1024,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    drop_out=0.1,
    qkv_bias=False
)

vit_huge = ViTConfig(
    patch_size=16,
    embed_dim=1280,
    hidden_size=1280,
    num_layers=32,
    num_heads=16,
    drop_out=0.1,
    qkv_bias=False
)