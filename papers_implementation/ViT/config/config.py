"""
Named ViT variants (ViT-Tiny, ViT-Small, ViT-Base) for easy import.
Reference: An Image Is Worth 16x16 Words (Dosovitskiy et al.).
"""

VIT_TINY = dict(
    patch_size=16,
    dim=192,
    depth=12,
    heads=3,
    mlp_dim=768,
)

VIT_SMALL = dict(
    patch_size=16,
    dim=384,
    depth=12,
    heads=6,
    mlp_dim=1536,
)

VIT_BASE = dict(
    patch_size=16,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
)
