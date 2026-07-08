"""Vanilla convolutional VAE for Simpsons-MNIST (grayscale, 28x28).

This is generation 0 of a model-collapse study: train a VAE on the real data,
then generate a synthetic dataset that a *next* generation VAE can be trained on
(``--data-source path/to/synthetic_data.npz``). Each run writes a rich
``metrics.json`` so the degradation across generations can be plotted later.

Run (you drive training, not me):

    # Generation 0 -- train on the real images
    python vae.py --generation 0 --data-source folder --output-dir outputs/gen0

    # Generation 1 -- train on gen0's synthetic output
    python vae.py --generation 1 \
        --data-source outputs/gen0/synthetic_data.npz \
        --output-dir outputs/gen1

Outputs per run (under ``--output-dir``):
    vae_best.pt          best checkpoint (lowest test neg-ELBO)
    metrics.json         config + per-epoch curves + collapse metrics
    reconstructions.png  test images (top) vs their reconstructions (bottom)
    prior_samples.png    grid of samples decoded from the N(0, I) prior
    synthetic_data.npz   generated images (uint8) to feed the next generation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import losses as L
import metrics as M
from data import IMAGE_SIZE, build_loader
from feature_extractor import extract_features, load_feature_extractor


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """A small pre-activation residual block that preserves channels and size."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class VAE(nn.Module):
    """Convolutional VAE with residual blocks and an upsample+conv decoder.

    Improvements over the original (step 3): wider channels, residual blocks, and
    a decoder that upsamples with nearest/bilinear interpolation followed by a
    conv instead of strided ``ConvTranspose2d`` (which produces checkerboard
    artifacts). Also holds a learned scalar ``log_sigma`` for the optional
    Gaussian observation-noise likelihood (step 4).
    """

    def __init__(self, latent_dim: int = 32, width: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        w = width

        # Encoder: (1,28,28) -> (w,14,14) -> (2w,7,7)
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            ResBlock(w),
            nn.Conv2d(w, 2 * w, 4, stride=2, padding=1),  # 7x7
            nn.ReLU(inplace=True),
            ResBlock(2 * w),
        )
        self.enc_out = 2 * w * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out, latent_dim)

        # Decoder: latent -> (2w,7,7) -> (w,14,14) -> (w,28,28) -> (1,28,28)
        self.width = w
        self.fc_dec = nn.Linear(latent_dim, 2 * w * 7 * 7)
        self.dec = nn.Sequential(
            ResBlock(2 * w),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14
            nn.Conv2d(2 * w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(w),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, 1, 3, padding=1),
            nn.Sigmoid(),  # mean image in [0, 1]
        )

        # Learned global observation noise (used only by the Gaussian likelihood).
        self.log_sigma = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 2 * self.width, 7, 7)
        return self.dec(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Discriminator(nn.Module):
    """PatchGAN-ish discriminator for the optional adversarial term (step 1)."""

    def __init__(self, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(w, 2 * w, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(2 * w),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * w, 1, 7),  # -> 1x1 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # (B, 1) real/fake logit


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Bundles the optional sharpness features (steps 1-4)."""

    recon_loss: str = "bce"        # bce | mse | gaussian (learned sigma)  [step 4]
    lpips_weight: float = 0.0      # perceptual reconstruction weight       [step 1]
    lpips_net: str = "vgg"         # LPIPS backbone: vgg | alex
    gan_weight: float = 0.0        # adversarial (VAE-GAN) weight           [step 1]
    disc_lr: float = 2e-4          # discriminator learning rate
    free_bits: float = 0.0         # nats/dim floor on KL                   [step 2]
    beta: float = 1.0              # target KL weight
    kl_warmup_epochs: int = 0      # linear 0 -> beta ramp                  [step 2]

    def beta_at(self, epoch: int, total_epochs: int) -> float:
        """Warmed-up KL weight at a (1-indexed) epoch."""
        if self.kl_warmup_epochs > 0:
            return self.beta * min(1.0, epoch / self.kl_warmup_epochs)
        return self.beta


def add_sharpness_args(p: argparse.ArgumentParser) -> None:
    """Register the step 1-4 flags shared by vae.py and self_consuming_loop.py."""
    p.add_argument("--width", type=int, default=64,
                   help="Base conv channel width (decoder capacity, step 3).")
    p.add_argument("--recon-loss", choices=["bce", "mse", "gaussian"], default="bce",
                   help="Pixel likelihood; 'gaussian' learns observation sigma (step 4).")
    p.add_argument("--lpips-weight", type=float, default=0.0,
                   help="Perceptual reconstruction weight (step 1). 0 disables.")
    p.add_argument("--lpips-net", choices=["vgg", "alex"], default="vgg")
    p.add_argument("--gan-weight", type=float, default=0.0,
                   help="Adversarial (VAE-GAN) weight (step 1). 0 disables.")
    p.add_argument("--disc-lr", type=float, default=2e-4)
    p.add_argument("--free-bits", type=float, default=0.0,
                   help="Nats/dim KL floor to keep latent units active (step 2).")
    p.add_argument("--kl-warmup-epochs", type=int, default=0,
                   help="Linearly ramp KL weight 0->beta over N epochs (step 2).")


def cfg_from_args(args) -> "TrainConfig":
    return TrainConfig(
        recon_loss=args.recon_loss, lpips_weight=args.lpips_weight,
        lpips_net=args.lpips_net, gan_weight=args.gan_weight,
        disc_lr=args.disc_lr, free_bits=args.free_bits,
        beta=args.beta, kl_warmup_epochs=args.kl_warmup_epochs)


def _adv_targets(logits, value):
    return torch.full_like(logits, value)


def train_epoch(model, disc, loader, device, cfg, opt_vae, opt_disc,
                beta_eff, lpips_fn, desc=None):
    """One training pass with the configured reconstruction/KL/GAN objective."""
    model.train()
    if disc is not None:
        disc.train()
    bce = F.binary_cross_entropy_with_logits
    agg = defaultdict(float)
    n = 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, _ in bar:
        images = images.to(device)
        bs = images.size(0)

        recon, mu, logvar = model(images)
        rec = L.reconstruction_loss(recon, images, cfg.recon_loss, model.log_sigma)
        kl = L.kl_with_free_bits(mu, logvar, cfg.free_bits)
        lp = (lpips_fn(recon, images) if lpips_fn is not None
              else torch.zeros((), device=device))
        vae_obj = rec + cfg.lpips_weight * lp + beta_eff * kl

        d_loss = torch.zeros((), device=device)
        g_adv = torch.zeros((), device=device)
        if disc is not None:
            # Fakes = reconstructions AND prior samples, so both the recon path
            # and the (synthetic-data-producing) prior path are pushed realistic.
            z = torch.randn(bs, model.latent_dim, device=device)
            fake_prior = model.decode(z)

            opt_disc.zero_grad()
            d_real = disc(images)
            d_fake_r = disc(recon.detach())
            d_fake_p = disc(fake_prior.detach())
            d_loss = (bce(d_real, _adv_targets(d_real, 0.9))
                      + 0.5 * bce(d_fake_r, _adv_targets(d_fake_r, 0.0))
                      + 0.5 * bce(d_fake_p, _adv_targets(d_fake_p, 0.0)))
            d_loss.backward()
            opt_disc.step()

            g_r = disc(recon)
            g_p = disc(fake_prior)
            g_adv = 0.5 * bce(g_r, _adv_targets(g_r, 1.0)) \
                + 0.5 * bce(g_p, _adv_targets(g_p, 1.0))
            vae_obj = vae_obj + cfg.gan_weight * g_adv

        opt_vae.zero_grad()
        vae_obj.backward()
        opt_vae.step()

        n += bs
        for k, v in {"recon": rec, "kl": kl, "lpips": lp,
                     "g_adv": g_adv, "d_loss": d_loss}.items():
            agg[k] += float(v) * bs
        bar.set_postfix(recon=f"{agg['recon'] / n:.1f}", kl=f"{agg['kl'] / n:.2f}",
                        d=f"{agg['d_loss'] / n:.2f}")
    return {k: v / n for k, v in agg.items()}


@torch.no_grad()
def eval_recon_kl(model, x, cfg, device, batch_size=1024):
    """Mean test reconstruction and (floor-free) KL for logging/checkpointing."""
    model.eval()
    tot_rec = tot_kl = 0.0
    n = 0
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        recon, mu, logvar = model(xb)
        rec = L.reconstruction_loss(recon, xb, cfg.recon_loss, model.log_sigma)
        kl = L.kl_with_free_bits(mu, logvar, 0.0)
        bs = xb.size(0)
        tot_rec += float(rec) * bs
        tot_kl += float(kl) * bs
        n += bs
    return tot_rec / n, tot_kl / n


@torch.no_grad()
def load_split_tensor(source: str, split: str, device: torch.device) -> torch.Tensor:
    """Load every image of a split into one ``(N, 1, 28, 28)`` CPU tensor."""
    loader = build_loader(source, split=split, batch_size=256, shuffle=False)
    return torch.cat([imgs for imgs, _ in loader], dim=0)


# --------------------------------------------------------------------------- #
# Generation of synthetic images
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_prior(model, n: int, device: torch.device, batch_size: int = 512):
    """Decode ``n`` samples from the N(0, I) prior -> ``(n, 1, 28, 28)`` in [0,1]."""
    model.eval()
    out = []
    for start in range(0, n, batch_size):
        k = min(batch_size, n - start)
        z = torch.randn(k, model.latent_dim, device=device)
        out.append(model.decode(z).cpu())
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_collapse_metrics(model, real_test, gen_images, device, kl_active_thresh):
    """Assemble the model-collapse metric dictionary."""
    extractor = load_feature_extractor(device)

    # --- FID + diversity in the fixed feature space ------------------------- #
    # FID uses torchmetrics' vetted computation, but with our domain-trained
    # classifier as the feature module instead of InceptionV3.
    fid = M.frechet_distance(extractor, gen_images, real_test, device)
    real_feat = extract_features(extractor, real_test, device).numpy()
    gen_feat = extract_features(extractor, gen_images, device).numpy()
    feature_variance = float(np.trace(np.cov(gen_feat, rowvar=False)))
    real_feature_variance = float(np.trace(np.cov(real_feat, rowvar=False)))
    mean_pair = M.mean_pairwise_distance(gen_feat)
    real_mean_pair = M.mean_pairwise_distance(real_feat)

    # --- Class coverage (mode collapse) ------------------------------------- #
    real_dist = M.class_distribution(extractor, real_test, device)
    gen_dist = M.class_distribution(extractor, gen_images, device)
    class_entropy = M.entropy(gen_dist)
    real_class_entropy = M.entropy(real_dist)
    class_kl = M.kl_divergence(gen_dist, real_dist)

    # --- Pixel-space distribution drift ------------------------------------- #
    gen_np = gen_images.numpy()
    real_np = real_test.numpy()
    pixel_std_mean = float(gen_np.std(0).mean())
    real_pixel_std_mean = float(real_np.std(0).mean())
    pixel_mean_l1 = float(np.abs(gen_np.mean(0) - real_np.mean(0)).mean())

    # --- Latent activity (posterior collapse) ------------------------------- #
    mus, logvars = [], []
    for start in range(0, len(real_test), 256):
        batch = real_test[start:start + 256].to(device)
        mu, logvar = model.encode(batch)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    mu_all, logvar_all = torch.cat(mus), torch.cat(logvars)
    dim_kl = M.per_dim_kl(mu_all, logvar_all)
    active_units = int((dim_kl > kl_active_thresh).sum().item())

    return {
        "fid": fid,
        "active_units": active_units,
        "latent_dim": model.latent_dim,
        "feature_variance": feature_variance,
        "real_feature_variance": real_feature_variance,
        "mean_pairwise_feat_dist": mean_pair,
        "real_mean_pairwise_feat_dist": real_mean_pair,
        "class_entropy": class_entropy,
        "real_class_entropy": real_class_entropy,
        "class_kl": class_kl,
        "gen_class_distribution": gen_dist.tolist(),
        "real_class_distribution": real_dist.tolist(),
        "pixel_std_mean": pixel_std_mean,
        "real_pixel_std_mean": real_pixel_std_mean,
        "pixel_mean_l1": pixel_mean_l1,
        "per_dim_kl": dim_kl.tolist(),
    }


# --------------------------------------------------------------------------- #
# Visual samples
# --------------------------------------------------------------------------- #
@torch.no_grad()
def save_reconstructions(model, real_test, device, path, n=8):
    model.eval()
    x = real_test[:n].to(device)
    recon, _, _ = model(x)
    grid = make_grid(torch.cat([x.cpu(), recon.cpu()], dim=0), nrow=n)
    save_image(grid, path)


def save_samples(samples, path, n=64):
    grid = make_grid(samples[:n], nrow=8)
    save_image(grid, path)


# --------------------------------------------------------------------------- #
# One generation (reused by both the CLI and the self-consuming loop)
# --------------------------------------------------------------------------- #
def train_one_generation(
    generation: int,
    data_source: str,
    output_dir: str,
    device: torch.device,
    epochs: int = 30,
    latent_dim: int = 32,
    beta: float = 1.0,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_synth: int = 8000,
    kl_active_thresh: float = 1e-2,
    seed: int = 0,
    width: int = 64,
    train_cfg: "TrainConfig | None" = None,
) -> dict:
    """Train one VAE generation, write all artifacts, and return its report dict.

    ``data_source`` is ``"folder"`` (real data) or a path to a previous
    generation's ``synthetic_data.npz``. ``train_cfg`` selects the sharpness
    features (reconstruction likelihood, LPIPS, GAN, free-bits, KL warm-up); if
    omitted, a plain BCE ``TrainConfig(beta=beta)`` reproduces the original VAE.
    Writes ``vae_best.pt``, ``metrics.json``, ``reconstructions.png``,
    ``prior_samples.png`` and ``synthetic_data.npz`` into ``output_dir``.
    """
    cfg = train_cfg or TrainConfig(beta=beta)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generation {generation} | source={data_source} | device={device}")
    print(f"  recon={cfg.recon_loss} lpips={cfg.lpips_weight} gan={cfg.gan_weight} "
          f"free_bits={cfg.free_bits} beta={cfg.beta} warmup={cfg.kl_warmup_epochs}")

    # Data: train from the chosen source; the *real* test set is always the
    # yardstick for the metrics (that is the distribution we care about).
    train_loader = build_loader(data_source, split="train",
                                batch_size=batch_size, shuffle=True)
    real_test = load_split_tensor("folder", "test", device)

    model = VAE(latent_dim=latent_dim, width=width).to(device)
    opt_vae = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional adversarial (step 1) and perceptual (step 1) machinery.
    disc = opt_disc = None
    if cfg.gan_weight > 0:
        disc = Discriminator(width=width).to(device)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=cfg.disc_lr,
                                    betas=(0.5, 0.999))
    lpips_fn = None
    if cfg.lpips_weight > 0:
        lpips_fn = L.LPIPSLoss(cfg.lpips_net).to(device)

    history = []
    best_obj = float("inf")  # checkpoint by test (recon + beta*KL)
    ckpt_path = os.path.join(output_dir, "vae_best.pt")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        beta_eff = cfg.beta_at(epoch, epochs)
        tr = train_epoch(model, disc, train_loader, device, cfg,
                         opt_vae, opt_disc, beta_eff, lpips_fn,
                         desc=f"gen {generation} epoch {epoch}/{epochs}")
        te_rec, te_kl = eval_recon_kl(model, real_test, cfg, device)
        te_obj = te_rec + cfg.beta * te_kl
        history.append({
            "epoch": epoch, "beta_eff": beta_eff,
            "train_recon": tr["recon"], "train_kl": tr["kl"],
            "train_lpips": tr["lpips"], "train_g_adv": tr["g_adv"],
            "train_d_loss": tr["d_loss"],
            "test_loss": te_obj, "test_recon": te_rec, "test_kl": te_kl,
        })
        print(f"gen {generation} epoch {epoch:3d}/{epochs} | "
              f"train recon {tr['recon']:8.2f} | test obj {te_obj:8.2f} "
              f"(recon {te_rec:7.2f}, kl {te_kl:6.2f}, "
              f"lpips {tr['lpips']:.3f}, d {tr['d_loss']:.3f})")

        if te_obj < best_obj:
            best_obj = te_obj
            torch.save({"model_state": model.state_dict(),
                        "latent_dim": latent_dim, "width": width,
                        "generation": generation}, ckpt_path)

    # Reload best checkpoint for all downstream artifacts.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    # Synthetic dataset for the NEXT generation.
    synth = sample_prior(model, num_synth, device)
    synth_uint8 = (synth.squeeze(1).clamp(0, 1).numpy() * 255).astype(np.uint8)
    np.savez_compressed(os.path.join(output_dir, "synthetic_data.npz"),
                        images=synth_uint8)

    # Visual artifacts.
    save_reconstructions(model, real_test, device,
                         os.path.join(output_dir, "reconstructions.png"))
    save_samples(synth, os.path.join(output_dir, "prior_samples.png"))

    # Collapse metrics: compare a fresh batch of prior samples against real test.
    eval_samples = sample_prior(model, len(real_test), device)
    collapse = compute_collapse_metrics(model, real_test, eval_samples, device,
                                        kl_active_thresh)

    report = {
        "generation": generation,
        "data_source": data_source,
        "config": {
            "epochs": epochs, "latent_dim": latent_dim, "width": width,
            "beta": cfg.beta, "batch_size": batch_size,
            "lr": lr, "seed": seed,
            "train_images": len(train_loader.dataset),
            "recon_loss": cfg.recon_loss, "lpips_weight": cfg.lpips_weight,
            "lpips_net": cfg.lpips_net, "gan_weight": cfg.gan_weight,
            "free_bits": cfg.free_bits, "kl_warmup_epochs": cfg.kl_warmup_epochs,
            "learned_log_sigma": float(model.log_sigma.detach().cpu()),
        },
        "train_seconds": round(time.time() - t0, 1),
        "best_test_neg_elbo": best_obj,
        "final_test": history[-1],
        "collapse_metrics": collapse,
        "history": history,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== collapse metrics (generation {generation}) ===")
    for k in ["fid", "active_units", "feature_variance",
              "mean_pairwise_feat_dist", "class_entropy", "class_kl",
              "pixel_std_mean", "pixel_mean_l1"]:
        print(f"  {k:26s}: {collapse[k]}")
    print(f"\nSaved metrics -> {metrics_path}")
    print("Saved synthetic data for next generation -> "
          f"{os.path.join(output_dir, 'synthetic_data.npz')}")
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generation", type=int, default=0,
                   help="Index of this generation in the collapse chain.")
    p.add_argument("--data-source", default="folder",
                   help="'folder' for real data, or path to a synthetic_data.npz.")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-synth", type=int, default=8000,
                   help="How many synthetic images to write for the next generation.")
    p.add_argument("--kl-active-thresh", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    add_sharpness_args(p)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_one_generation(
        generation=args.generation,
        data_source=args.data_source,
        output_dir=args.output_dir,
        device=device,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        beta=args.beta,
        batch_size=args.batch_size,
        lr=args.lr,
        num_synth=args.num_synth,
        kl_active_thresh=args.kl_active_thresh,
        seed=args.seed,
        width=args.width,
        train_cfg=cfg_from_args(args),
    )


if __name__ == "__main__":
    main()
