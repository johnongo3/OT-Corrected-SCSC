"""Evaluation metrics for tracking model collapse across VAE generations.

Model collapse manifests as (1) drift of the generated distribution away from
the real one, and (2) a progressive *loss of variance / diversity* -- later
generations forget the tails and converge onto a few prototypes. The metrics
below are chosen to make both effects visible and are all saved to
``metrics.json`` so they can be plotted against the generation index later.

Key metrics
-----------
* ``fid``                   Frechet distance between generated and real-test
                            features (lower is better; rises as quality degrades).
* ``active_units``          # latent dims with per-dim KL above a threshold. VAE
                            posterior collapse shows up as this number falling.
* ``feature_variance``      Trace of the covariance of generated features; a
                            direct measure of sample diversity that shrinks under
                            collapse.
* ``mean_pairwise_feat_dist`` Average pairwise feature distance among generated
                            samples; another diversity signal.
* ``pixel_std_mean``        Mean per-pixel std of generated images (vs the real
                            reference) -- collapse drives this toward the real
                            mean image / a blur.
* ``class_entropy`` / ``class_kl``  Using the frozen classifier head, how evenly
                            the generated samples cover the 10 characters and how
                            far that histogram is from the real one. Mode collapse
                            drops the entropy and raises the KL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from feature_extractor import FEATURE_DIM, FeatureCNN


class _FIDFeatureModule(nn.Module):
    """Adapts our cached classifier into a feature module for torchmetrics FID.

    torchmetrics' default FID uses InceptionV3 (299x299 RGB), which is
    inappropriate for 28x28 grayscale data. Passing a custom ``feature`` module
    keeps the vetted Frechet computation while measuring distance in a
    domain-trained feature space. Exposing ``num_features`` lets torchmetrics
    skip its 299x299 dummy-forward probe, and because ``used_custom_model`` is
    set internally, it forwards our images through unchanged (no ``*255`` byte
    cast), so the CNN receives the float ``[0, 1]`` tensors it was trained on.
    """

    num_features = FEATURE_DIM

    def __init__(self, extractor: FeatureCNN):
        super().__init__()
        self.extractor = extractor

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.extractor.features(imgs)


@torch.no_grad()
def frechet_distance(
    extractor: FeatureCNN,
    gen_images: torch.Tensor,
    real_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """FID between generated and real images via torchmetrics, in our feature space.

    Both image tensors are ``(N, 1, 28, 28)`` floats in ``[0, 1]``.
    """
    fid = FrechetInceptionDistance(
        feature=_FIDFeatureModule(extractor).to(device).eval(),
        normalize=True,
    ).to(device)
    for start in range(0, len(real_images), batch_size):
        fid.update(real_images[start:start + batch_size].to(device), real=True)
    for start in range(0, len(gen_images), batch_size):
        fid.update(gen_images[start:start + batch_size].to(device), real=False)
    return float(fid.compute())


def mean_pairwise_distance(feats: np.ndarray, max_samples: int = 2000) -> float:
    """Average pairwise Euclidean distance between feature vectors (diversity)."""
    if len(feats) > max_samples:
        idx = np.random.RandomState(0).choice(len(feats), max_samples, replace=False)
        feats = feats[idx]
    # Efficient pairwise-distance mean via the mean of squared norms.
    sq = (feats ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.T)
    d2 = np.clip(d2, 0.0, None)
    n = len(feats)
    # Exclude the diagonal (self-distances) from the mean.
    total = np.sqrt(d2).sum()
    return float(total / (n * n - n))


@torch.no_grad()
def class_distribution(
    extractor: FeatureCNN, images: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Predicted-class histogram (probability vector) over a set of images."""
    extractor.eval()
    counts = np.zeros(10, dtype=np.float64)
    for start in range(0, len(images), 256):
        batch = images[start:start + 256].to(device)
        preds = extractor(batch).argmax(1).cpu().numpy()
        for p in preds:
            counts[p] += 1
    return counts / counts.sum()


def entropy(dist: np.ndarray) -> float:
    dist = np.clip(dist, 1e-12, None)
    return float(-(dist * np.log(dist)).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) between two categorical distributions."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def per_dim_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL contribution of each latent dimension across a dataset.

    Shape in: ``(N, latent_dim)``. Shape out: ``(latent_dim,)``.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean(0)
