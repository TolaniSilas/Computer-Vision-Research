"""
unit tests for ViT model (shape checks, forward pass, param count).
"""

import torch
from config.config import vit_testing
from src.model.vit import VisionTransformer


# shared test config.
config = vit_testing
img_size = 112
num_classes = 10
batch_size = 2


def test_vit_output_shape():
    """tests vit returns correct logits shape."""

    model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    x = torch.randn(batch_size, 3, img_size, img_size)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)


def test_vit_loss_with_labels():
    """tests vit returns a scalar loss when labels are provided."""

    model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    x = torch.randn(batch_size, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = model(x, labels=labels)

    assert loss.ndim == 0
    assert loss.item() > 0


def test_vit_no_labels_returns_logits():
    """tests vit returns logits and not loss when no labels are provided."""

    model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    x = torch.randn(batch_size, 3, img_size, img_size)
    out = model(x)

    assert out.shape == (batch_size, num_classes)


def test_vit_trainable_parameters():
    """tests vit has trainable parameters."""

    model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params > 0


def test_vit_single_image():
    """tests vit handles a single image input correctly."""

    model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    x = torch.randn(1, 3, img_size, img_size)
    logits = model(x)

    assert logits.shape == (1, num_classes)