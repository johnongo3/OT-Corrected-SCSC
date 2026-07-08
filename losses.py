"""Loss terms for the sharpness-improved VAE (steps 1-4).

Provides the pieces that ``vae.py`` composes into a training objective:

* ``reconstruction_loss`` -- pixel likelihood: ``bce`` (Bernoulli, the original),
  ``mse`` (plain squared error), or ``gaussian`` (Gaussian NLL with a *learned*
  global observation noise sigma -- step 4).
* ``kl_with_free_bits`` -- the standard VAE KL, optionally with a free-bits floor
  so low-KL latent dims are not driven to zero (step 2, revives dead units).
* ``LPIPSLoss`` -- perceptual reconstruction distance via a frozen VGG/AlexNet
  backbone, adapted for 1-channel [0,1] images (step 1).

All reconstruction terms return a per-image mean (a scalar) so they combine
additively with the (per-image-mean) KL term.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mode: str = "bce",
    log_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-image-mean pixel reconstruction loss.

    ``bce``      Bernoulli NLL summed over pixels (original behaviour).
    ``mse``      Sum of squared errors over pixels (unnormalised Gaussian).
    ``gaussian`` Gaussian NLL with a learned scalar ``log_sigma`` -- the observation
                 noise is fit to the data, which balances recon vs KL and sharpens.
    """
    if mode == "bce":
        return F.binary_cross_entropy(recon, x, reduction="none").sum((1, 2, 3)).mean()

    se = (x - recon).pow(2).sum((1, 2, 3))  # per-image sum of squared errors
    if mode == "mse":
        return se.mean()
    if mode == "gaussian":
        if log_sigma is None:
            raise ValueError("gaussian reconstruction requires a log_sigma parameter.")
        n_pixels = x[0].numel()
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (se / var + n_pixels * (math.log(2 * math.pi) + 2.0 * log_sigma))
        return nll.mean()
    raise ValueError(f"Unknown reconstruction mode {mode!r}.")


def kl_with_free_bits(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) as a per-image mean, with an optional free-bits floor.

    With ``free_bits == 0`` this equals the usual ``sum_dim(KL).mean_batch()``. With
    ``free_bits > 0`` each latent dimension's batch-mean KL is clamped up to that
    many nats before summing, so the optimiser stops paying to silence a dimension
    once it is already near the floor -- keeping more units active.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
    kl_dim_mean = kl_per_dim.mean(0)  # (latent,)
    if free_bits > 0.0:
        kl_dim_mean = torch.clamp(kl_dim_mean, min=free_bits)
    return kl_dim_mean.sum()


class LPIPSLoss(nn.Module):
    """Perceptual reconstruction distance for 1-channel [0,1] images.

    Wraps the frozen ``lpips`` backbone: grayscale images are replicated to 3
    channels and rescaled to [-1, 1] (what LPIPS expects). The backbone is not
    trained; only its gradient w.r.t. the decoder output is used.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips  # imported lazily so the dependency is only needed when used
        self.lpips = lpips.LPIPS(net=net, verbose=False)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        recon3 = recon.repeat(1, 3, 1, 1) * 2 - 1
        x3 = x.repeat(1, 3, 1, 1) * 2 - 1
        return self.lpips(recon3, x3).mean()
