"""Self-consuming (recursive) training loop for the MNIST VAE in ``vae.py``.

A VAE is generative out of the box: it has a prior ``z ~ N(0, I)`` we can decode
directly, so the self-consuming chain is the canonical one --

    gen 0  <- real MNIST images
    gen 1  <- images decoded from gen 0's prior samples
    gen 2  <- images decoded from gen 1's prior samples
    ...

Each generation trains a *fresh* VAE (same architecture as ``vae.py``) from
scratch on the previous generation's synthetic output, then samples a new set.
As the loop proceeds the model forgets the tails of the data distribution and
the samples grow blurry and homogeneous -- "model collapse" (Shumailov et al.,
"The Curse of Recursion"; Alemohammad et al., "Self-Consuming Generative Models
Go MAD"). We track it against a *fixed* real test set with:

    fid               Frechet distance in a small MNIST-CNN's feature space
                      (fidelity + diversity; drifts UP as the model collapses)
    test_bce          reconstruction BCE on real test images (drifts UP)
    sample_pixel_std  per-pixel std across a synthetic batch (diversity, DOWN)
    mean_pairwise_l2  mean pairwise L2 between samples (diversity, DOWN)
    post_std_mean     mean std of the aggregated posterior means (DOWN)

Artifacts per generation land in ``<output-root>/gen<k>/`` (sample grid +
synthetic tensor). Aggregated metrics land in ``summary.{json,csv}`` and a
``collapse.png`` plot.

Usage (you run this):

    python vae_self_consuming.py --generations 50 --epochs 60 \
        --output-root runs/vae_mnist

Set ``--real-fraction`` above 0 to mix fresh real data back into each generation
(the "data accumulation" mitigation) and watch collapse slow down.

Set ``--correction-lambda`` above 0 to enable the batch-OT corrector (Gillman et
al. 2024): a frozen gen-0 VAE + fixed real anchors define a drift-free latent
reference, and each generation's synthetic latents are transported a proportion
``lambda`` toward their OT-matched anchor before the next generation trains --

    python vae_self_consuming.py --generations 50 --epochs 60 \
        --correction-lambda 0.5 --output-root runs/vae_mnist_ot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob
from PIL import Image
import torchvision.transforms as T

from vae import (Encoder, Decoder, ConvEncoder, ConvDecoder, Model, loss_function,
                 x_dim, hidden_dim, latent_dim)
from ot_corrector import (build_anchor_latents, ot_correct_images_lambda, _stratified_indices,
                          encode_all_latents, nn_correct_images, minibatch_ot_correct_images)
from ot import sliced_wasserstein_distance

DATA_ROOT = "data"
CELEBA_DIR = "dataset/img_align_celeba"

# Dataset-dependent config, set by run() from --dataset. Defaults = MNIST (MLP VAE).
DATASET = "mnist"
LATENT = latent_dim          # 96 for MNIST MLP; 128 for the CelebA conv VAE


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# 28x28 single-channel datasets: identical shape, so they share the MLP VAE, the
# FIDNet architecture and all metrics -- only the pixels and the 10 class meanings
# differ. Each still needs its OWN trained FID net (see get_fid_net).
GRAY_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}


def load_gray(name):
    """Return (train_images, test_images) as float [0,1] tensors, shape (N,1,28,28)."""
    cls = GRAY_DATASETS[name]
    train = cls(root=DATA_ROOT, train=True, download=True)
    test = cls(root=DATA_ROOT, train=False, download=True)
    train_images = train.data.float().div(255.0).unsqueeze(1)
    test_images = test.data.float().div(255.0).unsqueeze(1)
    return train_images, test_images


def load_mnist():
    """MNIST (train_images, test_images); kept for vae_digit_generations.py."""
    return load_gray("mnist")


def load_train_labels(name="mnist"):
    """Train targets (for class-stratified anchor selection)."""
    return GRAY_DATASETS[name](root=DATA_ROOT, train=True, download=True).targets


def load_celeba(n_train, n_test=2000, image_size=64, data_dir=CELEBA_DIR):
    """Load n_train + n_test aligned CelebA faces as float [0,1] (N,3,image_size^2).
    CenterCrop(148)->Resize->ToTensor; returns (train_images, test_images)."""
    tf = T.Compose([T.CenterCrop(148), T.Resize(image_size), T.ToTensor()])
    paths = sorted(glob.glob(f"{data_dir}/*.jpg"))
    need = n_train + n_test
    if len(paths) < need:
        raise RuntimeError(f"CelebA: need {need} images, found {len(paths)} in {data_dir}")
    print(f"Loading {need} CelebA faces from {data_dir} (this can take a minute)...")
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[:need]])
    return imgs[:n_train], imgs[n_train:need]


def as_loader(images, batch_size, shuffle, device):
    labels = torch.zeros(len(images), dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))


# ----------------------------------------------------------------------------
# Train / sample / encode  (reuses vae.py's Encoder/Decoder/Model/loss)
# ----------------------------------------------------------------------------
def build_vae(device):
    if DATASET == "celeba":
        enc = ConvEncoder(num_channels=3, image_size=64, latent_dim=LATENT, base=64)
        dec = ConvDecoder(num_channels=3, image_size=64, latent_dim=LATENT, base=64)
    else:
        enc = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        dec = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    return Model(encoder=enc, decoder=dec).to(device)


def train_vae(images, epochs, lr, batch_size, device, seed):
    """Train a fresh VAE from scratch on `images`; return the eval-mode model."""
    set_seed(seed)
    model = build_vae(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    loader = as_loader(images, batch_size, shuffle=True, device=device)
    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def sample_images(model, num, device, bs=512):
    """Decode `num` prior samples z ~ N(0, I) into [0,1] images (N,C,H,W)."""
    out = []
    for i in range(0, num, bs):
        n = min(bs, num - i)
        z = torch.randn(n, LATENT, device=device)
        out.append(model.decoder(z).clamp(0, 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def posterior_std_mean(model, images, device, bs=512):
    """Mean std (across the dataset) of the encoder's posterior means -- a proxy
    for how much of the latent space the aggregated posterior still spans."""
    means = []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        mu, _ = model.encoder(x)
        means.append(mu.cpu())
    means = torch.cat(means)
    return means.std(dim=0).mean().item()


@torch.no_grad()
def test_bce(model, test_loader, device):
    """Mean per-image reconstruction BCE on the real test set (drift proxy)."""
    total, n = 0.0, 0
    for x, _ in test_loader:
        x = x.to(device)
        x_hat, _, _ = model(x)
        bce = F.binary_cross_entropy(x_hat.reshape(x.size(0), -1),
                                     x.reshape(x.size(0), -1), reduction="sum")
        total += bce.item()
        n += x.size(0)
    return total / n


# ----------------------------------------------------------------------------
# Diversity metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def diversity_metrics(images, k=512):
    batch = images[:k]
    pixel_std = batch.std(dim=0).mean().item()
    flat = batch.flatten(1)
    n = flat.shape[0]
    dists = torch.cdist(flat, flat)
    mean_pairwise = (dists.sum() / (n * (n - 1))).item()
    return pixel_std, mean_pairwise


# ----------------------------------------------------------------------------
# FID via a small MNIST CNN (trained once, cached)
# ----------------------------------------------------------------------------
class FIDNet(nn.Module):
    """Tiny LeNet-ish classifier; its 128-d penultimate layer is the FID feature."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.features(x))


def get_fid_net(train_images, device, name="mnist", cache=None, epochs=3):
    """Load or train the FID feature extractor on the real 28x28 dataset.

    Each dataset gets its OWN net and cache file: a CNN trained to separate digits
    produces meaningless features for clothing, which would make the FID and the
    Wasserstein-from-gen0 columns unreadable.
    """
    cache = Path(cache) if cache else Path("outputs") / f"{name}_fid_cnn.pt"
    net = FIDNet().to(device)
    if cache.exists():
        net.load_state_dict(torch.load(cache, map_location=device))
        net.eval()
        return net
    print(f"Training FID feature CNN on real {name} (one-time)...")
    train = GRAY_DATASETS[name](root=DATA_ROOT, train=True, download=True)
    x = train.data.float().div(255.0).unsqueeze(1)
    y = train.targets
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True)
    opt = Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), cache)
    return net


class PixelFeatureNet(nn.Module):
    """Self-contained pixel feature space for FID/Wasserstein on RGB images: average-
    pool to 16x16 and flatten (768-d). No training or downloads -- it uses the same
    Frechet / sliced-Wasserstein machinery as the MNIST CNN features, just on coarse
    pixels, so the Wasserstein-from-gen0 and FID columns line up with MNIST in kind.
    Captures distributional drift/collapse (it is not a perceptual/Inception FID)."""

    def __init__(self, size=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(size)

    def features(self, x):
        return self.pool(x).flatten(1)


def get_feature_net(dataset, train_images, device):
    """FID/Wasserstein feature extractor: a CNN trained on the 28x28 dataset itself,
    or a self-contained downsampled-pixel space for CelebA faces."""
    if dataset == "celeba":
        return PixelFeatureNet().to(device).eval()
    return get_fid_net(train_images, device, name=dataset)


@torch.no_grad()
def fid_features(net, images, device, bs=512):
    feats = []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        feats.append(net.features(x).cpu().numpy())
    return np.concatenate(feats)


def frechet_distance(feat1, feat2, eps=1e-6):
    """Standard FID formula between two feature sets."""
    mu1, mu2 = feat1.mean(0), feat2.mean(0)
    sigma1 = np.cov(feat1, rowvar=False)
    sigma2 = np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    # Jitter the diagonals so the matrix-sqrt of the product stays well-defined
    # even when a collapsed generation makes a covariance near-singular.
    offset = eps * np.eye(sigma1.shape[0])
    covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if isinstance(covmean, tuple):  # older scipy returned (sqrt, errest)
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ----------------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------------
def run(args):
    global DATASET, LATENT
    DATASET = args.dataset
    LATENT = 128 if args.dataset == "celeba" else latent_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "celeba":
        # Real pool must be large enough for both the gen-0 subsample and the anchors.
        pool_size = max(args.num_synth, args.n_anchors)
        # >768 test faces so the 768-d pixel-feature FID covariance is full-rank.
        train_images, test_images = load_celeba(pool_size, n_test=2000, data_dir=args.data_dir)
        train_labels = torch.zeros(len(train_images), dtype=torch.long)  # no class labels
    else:
        train_images, test_images = load_gray(args.dataset)
        train_labels = load_train_labels(args.dataset)
    test_loader = as_loader(test_images, args.batch_size, shuffle=False, device=device)

    # Fixed real reference for FID: the model's feature stats never move.
    fid_net = get_feature_net(args.dataset, train_images, device)
    real_feats = fid_features(fid_net, test_images, device)

    # Constant training-set size across generations so N doesn't confound collapse.
    set_seed(args.seed)
    n = args.num_synth
    perm = torch.randperm(len(train_images))[:n]
    real_pool = train_images[perm]

    # Anchor-augment baseline: keep the SAME class-stratified real anchor images the
    # OT corrector uses (identical seed/selection) inside each generation's training
    # set, so the two methods differ only in HOW those anchors are used -- retained
    # as raw images vs. used as OT transport targets. N is held constant (n_anchors
    # of the num_synth training images are anchors, the rest synthetic).
    anchor_images = None
    if args.anchor_augment:
        gen_anchor = torch.Generator().manual_seed(args.seed)
        anchor_idx = _stratified_indices(train_labels, args.n_anchors, gen_anchor)
        anchor_images = train_images[anchor_idx]

    aug_mode = ("accumulate-all" if args.augment_real and args.accumulate
                else "real+synth (1:1)" if args.augment_real else "off")
    print(f"Self-consuming VAE loop | {args.generations} generations | "
          f"{args.epochs} epochs each | latent={latent_dim} | N={n} | device={device}")
    print(f"real_fraction={args.real_fraction} (0 = pure synthetic replacement) | "
          f"correction_lambda={args.correction_lambda} (0 = off) | "
          f"anchor_augment={args.anchor_augment} | augment_real={aug_mode} | "
          f"full_anchors={args.full_anchors} | anchor_minibatch={args.anchor_minibatch}"
          + (f" ({len(anchor_images)} fixed real anchors retained)" if args.anchor_augment else "")
          + "\n")

    current_images = real_pool
    rows: list[dict] = []
    synthetic_history: list[torch.Tensor] = []  # for --accumulate

    # OT correction: a frozen gen-0 VAE + fixed real anchors define a drift-free
    # reference. Frozen once at gen 0 (below) and reused every later generation.
    # `anchor_bank` holds ALL real latents for --full-anchors (NN projection).
    frozen = None
    anchors = None
    anchor_bank = None
    gen0_feats = None  # FID features of gen-0 raw samples (Wasserstein-from-gen0 reference)

    for gen in range(args.generations):
        gen_dir = output_root / f"gen{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        source = "real" if gen == 0 else f"gen{gen - 1} samples"
        print(f"{'=' * 64}\nGeneration {gen}/{args.generations - 1}  (train on {source})")

        model = train_vae(current_images, args.epochs, args.lr,
                           args.batch_size, device, seed=args.seed + gen)
        torch.save(model.state_dict(), gen_dir / "weights.pt")

        raw_synthetic = sample_images(model, args.num_synth, device)

        # Freeze the gen-0 VAE + real anchors as the OT reference space (once).
        # --full-anchors uses the ENTIRE real set as the anchor bank (NN projection);
        # otherwise a fixed class-stratified set of n_anchors (batch-OT barycentric).
        if args.correction_lambda > 0 and frozen is None:
            frozen = build_vae(device)
            frozen.load_state_dict(model.state_dict())
            frozen.eval()
            if args.anchor_minibatch > 0:
                # Large anchor POOL; minibatch OT draws a random subset per chunk.
                anchors = build_anchor_latents(frozen, train_images, train_labels,
                                               args.n_anchors, device, seed=args.seed)
                print(f"  Froze gen-{gen} VAE + anchor POOL ({len(anchors)}) | "
                      f"minibatch-OT K={args.anchor_minibatch} "
                      f"(lambda={args.correction_lambda}, reg={args.ot_reg}).")
            elif args.full_anchors:
                # NN anchor bank sized by n_anchors (class-stratified); all real if >= dataset.
                if args.n_anchors >= len(train_images):
                    bank_imgs = train_images
                else:
                    gen_a = torch.Generator().manual_seed(args.seed)
                    bank_imgs = train_images[_stratified_indices(train_labels, args.n_anchors, gen_a)]
                anchor_bank = encode_all_latents(frozen, bank_imgs, device)
                print(f"  Froze gen-{gen} VAE + NN anchor bank "
                      f"({len(anchor_bank)} real latents, lambda={args.correction_lambda}).")
            else:
                anchors = build_anchor_latents(frozen, train_images, train_labels,
                                               args.n_anchors, device, seed=args.seed)
                print(f"  Froze gen-{gen} VAE + {len(anchors)} anchors "
                      f"(lambda={args.correction_lambda}, reg={args.ot_reg}).")

        # Correction BEFORE the next generation trains: pull the synthetic latents a
        # proportion `lambda` toward the frozen reference, decode, propagate forward.
        if args.correction_lambda > 0 and frozen is not None:
            if args.anchor_minibatch > 0:
                synthetic = minibatch_ot_correct_images(
                    frozen, raw_synthetic, anchors, args.correction_lambda,
                    args.anchor_minibatch, device, reg=args.ot_reg, seed=args.seed + gen)
            elif args.full_anchors:
                synthetic = nn_correct_images(frozen, raw_synthetic, anchor_bank,
                                              args.correction_lambda, device)
            else:
                synthetic = ot_correct_images_lambda(frozen, raw_synthetic, anchors,
                                                     args.correction_lambda, device, reg=args.ot_reg)
            save_image(raw_synthetic[:64], gen_dir / "samples_raw.png", nrow=8)
        else:
            synthetic = raw_synthetic

        # Metrics against the fixed real reference (on the propagated/corrected set).
        synth_feats = fid_features(fid_net, synthetic, device)
        fid = frechet_distance(real_feats, synth_feats)
        bce = test_bce(model, test_loader, device)
        pixel_std, mean_pairwise = diversity_metrics(synthetic)
        post_std = posterior_std_mean(model, current_images, device)
        # Raw (pre-correction) diversity + FID, for the corrected-vs-raw comparison.
        raw_pixel_std, raw_pairwise = diversity_metrics(raw_synthetic)
        if args.correction_lambda > 0:
            raw_feats = fid_features(fid_net, raw_synthetic, device)
            raw_fid = frechet_distance(real_feats, raw_feats)
        else:
            raw_feats, raw_fid = synth_feats, fid
        # Wasserstein-from-gen0: sliced W2 of the model's RAW samples vs gen-0's, in
        # FID-feature space (drift metric; 0 at gen 0, grows with collapse).
        if gen0_feats is None:
            gen0_feats = raw_feats
        w_from_gen0 = float(sliced_wasserstein_distance(raw_feats, gen0_feats,
                                                        n_projections=100, seed=0))

        rows.append({
            "generation": gen,
            "source": source,
            "fid": round(fid, 4),
            "test_bce": round(bce, 4),
            "sample_pixel_std": round(pixel_std, 6),
            "mean_pairwise_l2": round(mean_pairwise, 6),
            "post_std_mean": round(post_std, 6),
            "raw_fid": round(raw_fid, 4),
            "raw_pixel_std": round(raw_pixel_std, 6),
            "raw_pairwise_l2": round(raw_pairwise, 6),
            "w_from_gen0": round(w_from_gen0, 4),
        })
        print(f"  fid={fid:.3f}  test_bce={bce:.3f}  pixel_std={pixel_std:.4f}  "
              f"pairwise_l2={mean_pairwise:.4f}  post_std={post_std:.4f}  w0={w_from_gen0:.3f}"
              + (f"  (raw fid={raw_fid:.3f} pixel_std={raw_pixel_std:.4f})"
                 if args.correction_lambda > 0 else ""))

        save_image(synthetic[:64], gen_dir / "samples.png", nrow=8)
        torch.save(synthetic, gen_dir / "synthetic.pt")

        # Next generation's training set.
        if args.augment_real:
            # Add the synthetic ON TOP of the real pool (accumulation, set grows).
            # --accumulate keeps every generation's synthetic; else only this gen's.
            if args.accumulate:
                synthetic_history.append(synthetic)
                current_images = torch.cat([real_pool, *synthetic_history])
            else:
                current_images = torch.cat([real_pool, synthetic])
        elif args.anchor_augment:
            n_anc = len(anchor_images)
            current_images = torch.cat([synthetic[: args.num_synth - n_anc], anchor_images])
        elif args.real_fraction > 0:
            n_real = int(round(args.real_fraction * args.num_synth))
            idx = torch.randperm(len(train_images))[:n_real]
            current_images = torch.cat([synthetic[: args.num_synth - n_real], train_images[idx]])
        else:
            current_images = synthetic

        write_summary(output_root, rows, vars(args))
        plot_collapse(output_root, rows)

    print(f"\nDone. summary.csv / summary.json / collapse.png -> {output_root}")


def write_summary(output_root: Path, rows: list[dict], config: dict) -> None:
    with open(output_root / "summary.json", "w") as f:
        json.dump({"config": config, "rows": rows}, f, indent=2)
    with open(output_root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_collapse(output_root: Path, rows: list[dict]) -> None:
    gens = [r["generation"] for r in rows]
    # "raw" columns only differ from the propagated ones when correction is on.
    has_raw = all("raw_fid" in r for r in rows) and any(
        r["raw_fid"] != r["fid"] for r in rows)
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(gens, [r["fid"] for r in rows], marker="o", label="corrected")
    if has_raw:
        axes[0, 0].plot(gens, [r["raw_fid"] for r in rows], marker="s", alpha=0.6,
                        label="raw synthetic")
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("FID vs real test (up = collapse)", fontsize=11)

    axes[0, 1].plot(gens, [r["test_bce"] for r in rows], marker="o")
    axes[0, 1].set_title("Real-test recon BCE (up = drift)", fontsize=11)

    axes[1, 0].plot(gens, [r["sample_pixel_std"] for r in rows], marker="o", label="pixel std")
    axes[1, 0].plot(gens, [r["mean_pairwise_l2"] for r in rows], marker="s", alpha=0.6,
                    label="pairwise L2")
    if has_raw:
        axes[1, 0].plot(gens, [r["raw_pixel_std"] for r in rows], marker="^", alpha=0.5,
                        linestyle="--", label="pixel std (raw)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Sample diversity (down = collapse)", fontsize=11)

    axes[1, 1].plot(gens, [r["post_std_mean"] for r in rows], marker="o")
    axes[1, 1].set_title("Posterior-mean std (down = collapse)", fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("Generation")
    plt.tight_layout()
    plt.savefig(output_root / "collapse.png", dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist", "celeba"], default="mnist",
                   help="mnist / fashion_mnist (28x28 MLP VAE) or celeba (64x64 conv VAE).")
    p.add_argument("--data-dir", default=CELEBA_DIR,
                   help="CelebA image directory (used when --dataset celeba).")
    p.add_argument("--generations", type=int, default=50,
                   help="Number of generations (gen 0 is the real-data model).")
    p.add_argument("--epochs", type=int, default=60, help="Epochs per generation.")
    p.add_argument("--num-synth", type=int, default=10000,
                   help="Images per generation (also the real subsample size for gen 0).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--real-fraction", type=float, default=0.0,
                   help="Fraction of fresh real data mixed into each post-gen-0 set.")
    p.add_argument("--correction-lambda", type=float, default=0.0,
                   help="Batch-OT correction strength lambda in "
                        "z'=(1-lambda)z+lambda*OT_target (0 = off, 1 = full snap).")
    p.add_argument("--n-anchors", type=int, default=512,
                   help="Number of frozen, class-stratified real anchor latents (the POOL "
                        "size when --anchor-minibatch is used).")
    p.add_argument("--anchor-minibatch", type=int, default=0,
                   help="Minibatch-OT: EMD/Sinkhorn-transport each source chunk onto a fresh "
                        "random K-subset of the n_anchors pool (0 = off). Keeps exact OT "
                        "tractable while n_anchors scales to 10k+; coverage builds via "
                        "per-chunk resampling.")
    p.add_argument("--anchor-augment", action="store_true",
                   help="Baseline: retain the same n_anchors real anchor IMAGES in every "
                        "generation's training set (no OT), instead of transporting toward them.")
    p.add_argument("--augment-real", action="store_true",
                   help="Add the synthetic set ON TOP of the full real pool each generation "
                        "(accumulation; training set = real_pool + synthetic, so it grows).")
    p.add_argument("--accumulate", action="store_true",
                   help="With --augment-real, keep EVERY generation's synthetic (real + all "
                        "synth so far), not just the current one. Set grows unbounded.")
    p.add_argument("--full-anchors", action="store_true",
                   help="With --correction-lambda>0, use the ENTIRE real dataset as anchors via "
                        "nearest-neighbour projection in the frozen gen-0 latent space "
                        "(instead of n_anchors batch-OT).")
    p.add_argument("--ot-reg", type=float, default=0.0,
                   help="Entropic (Sinkhorn) OT regularization; 0 = exact EMD.")
    p.add_argument("--output-root", default="runs/vae_mnist")
    p.add_argument("--seed", type=int, default=42, help="Base seed; gen k uses seed + k.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
