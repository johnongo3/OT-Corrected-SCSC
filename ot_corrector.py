"""Optimal-transport collapse correction (POT-based).

Single home for every OT function in the repo. A frozen reference model plus a
fixed set of real "anchor" embeddings define a drift-free target space; each
generation's synthetic latents are transported back toward that anchor manifold
via batch OT before the next generation trains. The anchor space never moves, so
it acts as an external reference that counteracts the diversity collapse of a
self-consuming loop (Gillman et al. 2024, "Self-Correcting Self-Consuming
Training Loops for Generative Models").

    *_lambda    blend *every* latent a proportion
                `lambda` of the way toward its OT target:
                ``z' = (1 - lambda) * z + lambda * OT_target(z)``. Reference model
                is the VAE whose ``encoder`` returns ``(mean, log_var)``; the
                posterior mean is used as the embedding.
"""

from __future__ import annotations

import numpy as np
import ot  # POT: Python Optimal Transport
import torch


# ----------------------------------------------------------------------------
# Shared OT core
# ----------------------------------------------------------------------------
def _stratified_indices(labels, n_anchors, generator):
    """Pick `n_anchors` indices spread as evenly as possible across the classes in
    `labels`. Each class contributes floor(n/C) samples; the remainder is handed to
    a random subset of classes so the total is exactly n_anchors (capped by supply).
    """
    labels = labels.view(-1)
    classes = torch.unique(labels)
    n_classes = len(classes)
    base = n_anchors // n_classes
    remainder = n_anchors - base * n_classes
    # Randomize which classes receive the +1 remainder slot.
    extra = set(classes[torch.randperm(n_classes, generator=generator)[:remainder]].tolist())

    chosen = []
    for cls in classes.tolist():
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=generator)]
        take = min(base + (1 if cls in extra else 0), len(cls_idx))
        chosen.append(cls_idx[:take])
    return torch.cat(chosen)


def _barycentric_targets(source, anchors, reg):
    """OT barycentric projection of `source` points onto `anchors` (numpy, float64).

    Solves an OT plan between the uniform empirical measures on `source` and
    `anchors` (exact EMD if reg == 0, entropic Sinkhorn otherwise), then maps each
    source point to the anchor-weighted barycenter given by its row of the plan.
    """
    n_source, n_anchor = source.shape[0], anchors.shape[0]
    a = np.full(n_source, 1.0 / n_source)
    b = np.full(n_anchor, 1.0 / n_anchor)
    cost = ot.dist(source, anchors, metric="sqeuclidean")
    cost = cost / (cost.max() + 1e-12)  # scale-normalize (EMD plan is scale-invariant)
    if reg and reg > 0:
        plan = ot.sinkhorn(a, b, cost, reg)
    else:
        plan = ot.emd(a, b, cost)
    row_mass = plan.sum(axis=1, keepdims=True)
    row_mass = np.clip(row_mass, 1e-12, None)
    return (plan @ anchors) / row_mass


# ----------------------------------------------------------------------------
# Anchor spaces
# ----------------------------------------------------------------------------
@torch.no_grad()
def build_anchor_latents(frozen, images, labels, n_anchors, device, seed=0):
    """Encode a fixed, class-stratified set of real images through the frozen VAE
    encoder; return their posterior means as (n_anchors, latent_dim) anchors.

    For models whose ``encoder`` returns ``(mean, log_var)`` (the VAE).
    """
    generator = torch.Generator().manual_seed(seed)
    idx = _stratified_indices(labels, n_anchors, generator)
    anchor_imgs = images[idx].to(device).view(len(idx), -1)
    mean, _ = frozen.encoder(anchor_imgs)
    return mean.detach()


# ----------------------------------------------------------------------------
# Lambda-blend correction  (VAE / vae_self_consuming.py)
# ----------------------------------------------------------------------------
def ot_correct_latents_lambda(latents, anchors, lam, reg=0.0, batch_size=512):
    """Blend each latent a proportion `lam` toward its batch-OT barycentric image
    among `anchors`. Chunks of `batch_size` are transported independently (batch
    OT). `lam <= 0` is a no-op; `reg > 0` uses Sinkhorn instead of exact EMD.

        z' = (1 - lam) * z + lam * OT_target(z ; anchors)
    """
    if lam <= 0:
        return latents
    src = latents.detach().cpu().double().numpy()
    anc = anchors.detach().cpu().double().numpy()
    targets = np.empty_like(src)
    for i in range(0, len(src), batch_size):
        chunk = src[i:i + batch_size]
        targets[i:i + len(chunk)] = _barycentric_targets(chunk, anc, reg)
    t = torch.from_numpy(targets).to(device=latents.device, dtype=latents.dtype)
    return (1.0 - lam) * latents + lam * t


@torch.no_grad()
def ot_correct_images_lambda(frozen, images, anchors, lam, device, reg=0.0, batch_size=512):
    """Encode `images` with the frozen VAE, OT-correct the latents toward the
    anchors by `lam`, and decode back to [0,1] images (N,1,28,28). For VAE-style
    models (``encoder`` returns ``(mean, log_var)``; the mean is the embedding).
    """
    mean, _ = frozen.encoder(images.to(device).view(len(images), -1))
    corrected = ot_correct_latents_lambda(mean, anchors, lam, reg, batch_size)
    decoded = frozen.decoder(corrected).view(-1, 1, 28, 28).clamp(0, 1)
    return decoded.cpu()
