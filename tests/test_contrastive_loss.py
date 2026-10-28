"""
Unit tests for 3D ResNet Contrastive Encoder, NT-Xent loss, and augmentations.
"""

from __future__ import annotations

import torch

from copd_anomaly.models.augmentations import ContrastiveDualViewTransform
from copd_anomaly.models.encoder import ResNet3DEncoder
from copd_anomaly.models.losses import NTXentLoss


def test_encoder_forward_pass_and_shapes():
    """Verify encoder outputs correct embedding and unit-normalized projection shapes."""
    batch_size = 3
    in_channels = 1
    # Small patch size for fast unit testing
    x = torch.rand(batch_size, in_channels, 32, 32, 32)

    encoder = ResNet3DEncoder(
        in_channels=in_channels,
        base_filters=8,
        embedding_dim=64,
        projection_dim=32,
    )
    z_emb, z_proj = encoder(x)

    assert z_emb.shape == (batch_size, 64)
    assert z_proj.shape == (batch_size, 32)

    # Check projection L2 normalization: ||z_proj||_2 == 1.0
    norms = torch.norm(z_proj, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encoder_dual_channel_input():
    """Verify encoder functions seamlessly with 2-channel registered inspiratory+expiratory input."""
    x = torch.rand(2, 2, 32, 32, 32)
    encoder = ResNet3DEncoder(in_channels=2, base_filters=8, embedding_dim=64, projection_dim=32)
    z_emb, z_proj = encoder(x)
    assert z_emb.shape == (2, 64)
    assert z_proj.shape == (2, 32)


def test_ntxent_loss_values_and_gradient_flow():
    """Verify NT-Xent loss computation, positive pair preference, and backward gradient flow."""
    criterion = NTXentLoss(temperature=0.1)

    # Batch of 4 identical normalized vectors (perfect positive similarity)
    z1 = torch.nn.functional.normalize(torch.randn(4, 32, requires_grad=True), dim=-1)
    # View 2 identical to view 1
    loss_identical = criterion(z1, z1)

    # View 2 completely random
    z2 = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    z1.retain_grad()
    loss_random = criterion(z1, z2)

    # Loss for identical views must be substantially strictly lower than for random views
    assert loss_identical.item() < loss_random.item()

    # Gradient flow test
    loss_random.backward()
    assert z1.grad is not None
    assert not torch.isnan(z1.grad).any()


def test_dual_view_transform():
    """Verify dual view augmentation produces two augmented views with identical shape."""
    x = torch.rand(1, 16, 16, 16)
    dual_trans = ContrastiveDualViewTransform()
    v1, v2 = dual_trans(x)

    assert v1.shape == x.shape
    assert v2.shape == x.shape
    assert torch.all(v1 >= 0.0) and torch.all(v1 <= 1.0)
    assert torch.all(v2 >= 0.0) and torch.all(v2 <= 1.0)
