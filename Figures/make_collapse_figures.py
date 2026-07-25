"""Figures for the self-consuming model-collapse run.

Produces two images under ``--output-root``:

1. ``reconstruction_generations.png`` -- a stacked grid where the top row is the
   real test images and each subsequent row is that generation's reconstruction
   of the *same* images (row 0 = gen 0, row 1 = gen 1, ...). Reconstructions use
   the posterior mean (decode(mu)) so they are deterministic and clean.

2. ``metrics_fid_entropy_featvar.png`` -- FID, class-coverage entropy and
   feature variance vs generation, the three headline collapse signals.

Usage:
    python make_collapse_figures.py --output-root runs/experiment1
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display/Tk
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from data import CLASSES, FolderDataset
from vae import VAE


def pick_one_per_class(test_ds: FolderDataset):
    """Return (images, titles) with the first test image of each of the 10 classes."""
    chosen = {}
    for idx in range(len(test_ds)):
        label = test_ds.labels[idx]
        if label not in chosen:
            chosen[label] = idx
        if len(chosen) == len(CLASSES):
            break
    order = sorted(chosen)  # class 0..9
    images = torch.stack([test_ds[chosen[c]][0] for c in order])  # (10,1,28,28)
    titles = [CLASSES[c].split("_")[0] for c in order]
    return images, titles


@torch.no_grad()
def reconstruct(ckpt_path: str, images: torch.Tensor, device: torch.device):
    """Load a generation's checkpoint and return decode(mu) for the given images."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VAE(latent_dim=ckpt["latent_dim"], width=ckpt.get("width", 64)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu, _ = model.encode(images.to(device))
    return model.decode(mu).cpu()


def figure_reconstructions(output_root, num_gens, images, titles, device):
    n_cols = images.shape[0]
    n_rows = 1 + num_gens  # test row + one row per generation

    # Build the stack of rows: real images first, then each generation.
    rows = [images]
    row_labels = ["test"]
    for gen in range(num_gens):
        ckpt = os.path.join(output_root, f"gen{gen}", "vae_best.pt")
        rows.append(reconstruct(ckpt, images, device))
        row_labels.append(f"gen {gen}")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 0.9, n_rows * 0.9))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(rows[r][c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], rotation=0, ha="right", va="center",
                              fontsize=9,
                              fontweight="bold" if r == 0 else "normal")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    # Separate the real row from the generations with a little breathing room.
    fig.suptitle("Reconstructions collapsing across self-consuming generations",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(output_root, "reconstruction_generations.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def figure_metrics(output_root):
    with open(os.path.join(output_root, "summary.json")) as f:
        series = json.load(f)["series"]
    gens = series["generation"]

    panels = [
        ("fid", "FID vs real (lower = better)", None),
        ("class_entropy", "Class-coverage entropy", "real_class_entropy"),
        ("feature_variance", "Feature variance (diversity)", "real_feature_variance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title, ref) in zip(axes, panels):
        ax.plot(gens, series[key], marker="o", color="#c1121f", label=key)
        if ref and ref in series:
            ax.axhline(series[ref][0], ls="--", color="#457b9d", label="real data")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("generation")
        # Let matplotlib thin the ticks to integers instead of one per generation,
        # which overlaps badly once there are dozens of generations.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
        ax.grid(alpha=0.3)
    fig.suptitle("Model-collapse metrics across generations", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(output_root, "metrics_fid_entropy_featvar.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="runs/experiment1")
    p.add_argument("--num-gens", type=int, default=10,
                   help="How many generations to stack in the reconstruction grid.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = FolderDataset(split="test")
    images, titles = pick_one_per_class(test_ds)

    figure_reconstructions(args.output_root, args.num_gens, images, titles, device)
    figure_metrics(args.output_root)


if __name__ == "__main__":
    main()
