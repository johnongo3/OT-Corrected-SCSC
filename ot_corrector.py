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
    anchor_imgs = images[idx].to(device)
    mean, _ = frozen.encoder(anchor_imgs)
    return mean.detach()


# Lambda-blend correction 
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
    mean, _ = frozen.encoder(images.to(device))
    corrected = ot_correct_latents_lambda(mean, anchors, lam, reg, batch_size)
    decoded = frozen.decoder(corrected).clamp(0, 1)
    return decoded.cpu()


# ----------------------------------------------------------------------------
# Full-dataset nearest-neighbour correction
#
# The |anchors| -> N (entire dataset) limit. Exact OT against ~60k anchors is
# intractable (a 512x60000 LP per chunk), so we drop OT's uniform-marginal
# constraint and project each latent onto its NEAREST real-anchor latent in the
# frozen gen-0 space. At lam=1 this snaps every sample to the reconstruction of
# its nearest real image; the marginal constraint matters little when the anchor
# bank vastly outnumbers the sources.
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_all_latents(frozen, images, device, bs=1024):
    """Encode every image to its posterior-mean latent (batched). Kept on `device`
    as the anchor bank for nearest-neighbour projection."""
    out = []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        mean, _ = frozen.encoder(x)
        out.append(mean)
    return torch.cat(out)


def nn_correct_latents(latents, anchor_bank, lam, chunk=1024):
    """Blend each latent a proportion `lam` toward its NEAREST anchor (L2 in latent
    space). `anchor_bank` is (N_anchor, latent_dim) on the same device as `latents`."""
    if lam <= 0:
        return latents
    out = torch.empty_like(latents)
    for i in range(0, len(latents), chunk):
        z = latents[i:i + chunk]
        nn = anchor_bank[torch.cdist(z, anchor_bank).argmin(dim=1)]
        out[i:i + len(z)] = (1.0 - lam) * z + lam * nn
    return out


@torch.no_grad()
def nn_correct_images(frozen, images, anchor_bank, lam, device, chunk=1024):
    """Encode `images` with the frozen VAE, project the latents toward their nearest
    entry in `anchor_bank` by `lam`, and decode back to [0,1] images (N,1,28,28)."""
    mean, _ = frozen.encoder(images.to(device))
    corrected = nn_correct_latents(mean, anchor_bank, lam, chunk)
    decoded = frozen.decoder(corrected).clamp(0, 1)
    return decoded.cpu()


# ----------------------------------------------------------------------------
# Minibatch OT  (scale exact EMD / Sinkhorn to a large anchor POOL)
#
# Exact EMD is intractable against 10k+ anchors. Minibatch OT keeps the exact
# solver but, for each source chunk, transports onto a fresh RANDOM `minibatch`
# subset of the pool (barycentric). Resampling the subset per chunk (and per
# generation via the seed) means the whole pool is covered over the loop while
# each solve stays in the tractable regime -- and the marginal/coverage
# constraint still holds within each subset (Fatras et al. 2020, "Learning with
# minibatch Wasserstein"). Works with EMD (reg=0) or Sinkhorn (reg>0).
# ----------------------------------------------------------------------------
def minibatch_ot_correct_latents(latents, anchor_pool, lam, minibatch, reg=0.0,
                                 chunk=512, seed=0):
    """Blend each latent a proportion `lam` toward its barycentric image among a
    fresh random `minibatch`-subset of `anchor_pool`, resampled per chunk."""
    if lam <= 0:
        return latents
    n_pool = anchor_pool.shape[0]
    k = min(minibatch, n_pool)
    src = latents.detach().cpu().double().numpy()
    pool = anchor_pool.detach().cpu().double().numpy()
    rng = np.random.default_rng(seed)
    out = np.empty_like(src)
    for i in range(0, len(src), chunk):
        s = src[i:i + chunk]
        idx = rng.choice(n_pool, size=k, replace=False)
        out[i:i + len(s)] = _barycentric_targets(s, pool[idx], reg)
    t = torch.from_numpy(out).to(device=latents.device, dtype=latents.dtype)
    return (1.0 - lam) * latents + lam * t


@torch.no_grad()
def minibatch_ot_correct_images(frozen, images, anchor_pool, lam, minibatch, device,
                                reg=0.0, chunk=512, seed=0):
    """Encode `images` with the frozen VAE, minibatch-OT-correct the latents toward
    `anchor_pool` by `lam`, and decode back to [0,1] images (N,1,28,28)."""
    mean, _ = frozen.encoder(images.to(device))
    corrected = minibatch_ot_correct_latents(mean, anchor_pool, lam, minibatch, reg, chunk, seed)
    decoded = frozen.decoder(corrected).clamp(0, 1)
    return decoded.cpu()
