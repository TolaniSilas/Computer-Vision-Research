"""
unit tests for ViT components (PatchEmbed, TransformerBlock, and all other components).
"""

import pytest
import torch
from config.config import vit_testing
from src.model.components import (
    MLP,
    MultiHeadAttention,
    TransformerBlock,
    EncoderBlock,
    PatchEmbed,
    TransformerEncoder,
)


# shared test config.
config = vit_testing
img_size = 112
batch_size = 2


def test_mlp_output_shape():
    """tests mlp output shape is correct."""

    mlp = MLP(
        input_size=config["embed_dim"],
        hidden_size=config["hidden_size"],
        output_size=config["embed_dim"],
        dropout=config["drop_out"]
    )
    x = torch.randn(batch_size, 10, config["embed_dim"])
    out = mlp(x)

    assert out.shape == x.shape


def test_multi_head_attention_output_shape():
    """tests multi-head attention output shape is correct."""

    mha = MultiHeadAttention(
        d_in=config["embed_dim"],
        d_out=config["embed_dim"],
        dropout=config["drop_out"],
        num_heads=config["num_heads"]
    )
    x = torch.randn(batch_size, 10, config["embed_dim"])
    out = mha(x)

    assert out.shape == x.shape


def test_transformer_block_output_shape():
    """tests transformer block output shape is correct."""

    block = TransformerBlock(config)
    x = torch.randn(batch_size, 10, config["embed_dim"])
    out = block(x)

    assert out.shape == x.shape


def test_encoder_block_output_shape():
    """tests encoder block output shape is correct."""

    encoder = EncoderBlock(config)
    x = torch.randn(batch_size, 10, config["embed_dim"])
    out = encoder(x)

    assert out.shape == x.shape


def test_patch_embed_output_shape():
    """tests patch embed output shape is correct."""

    patch_embed = PatchEmbed(config, img_size=img_size)
    x = torch.randn(batch_size, 3, img_size, img_size)
    out = patch_embed(x)

    # expected: (batch, num_patches + 1, embed_dim).
    num_patches = (img_size // config["patch_size"]) ** 2
    assert out.shape == (batch_size, num_patches + 1, config["embed_dim"])


def test_patch_embed_num_patches():
    """tests static num_patches method returns correct value."""

    num_patches = PatchEmbed.num_patches(img_size, patch_size=config["patch_size"])

    assert num_patches == (img_size // config["patch_size"]) ** 2


def test_transformer_encoder_output_shape():
    """tests transformer encoder output shape is correct."""

    encoder = TransformerEncoder(config, img_size=img_size)
    x = torch.randn(batch_size, 3, img_size, img_size)
    out = encoder(x)

    num_patches = (img_size // config["patch_size"]) ** 2
    assert out.shape == (batch_size, num_patches + 1, config["embed_dim"])


def test_patch_embed_invalid_image_size():
    """tests patch embed raises assertion for invalid image size."""

    patch_embed = PatchEmbed(config, img_size=img_size)
    x = torch.randn(batch_size, 3, 111, 111)

    with pytest.raises(AssertionError):
        patch_embed(x)