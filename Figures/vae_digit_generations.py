"""Digit x generation collapse grid for the self-consuming VAE run.

Builds a 10x10 labelled grid: row k = generation k's model, column d = that
model's most convincing prior sample of digit d (chosen by the cached MNIST FID
CNN's class probability). Early generations render all ten distinct digits;
collapsed generations produce the same blob across every column -- the class
structure disappears, visualised per digit.

    python vae_digit_generations.py            # gens 0-9, writes Imgs/vae_digit_generations.png
"""

from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae import Encoder, Decoder, Model, x_dim, hidden_dim, latent_dim, device
from vae_self_consuming import FIDNet, get_fid_net, load_mnist

RUN_ROOT = Path("runs/vae_mnist_ot_1.0")
GEN_START = 40
GEN_END = 49
OUT = Path("Imgs") / f"vae_digit_generations_ot1.0_{GEN_START}_{GEN_END}.png"
N_GENS = GEN_END - GEN_START + 1
N_DIGITS = 10
POOL = 4000  # prior samples per generation to search for each digit


def build_vae():
    enc = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    dec = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    return Model(encoder=enc, decoder=dec).to(device)


@torch.no_grad()
def best_per_digit(gen, fid_net):
    """Return a (10,1,28,28) tensor: the gen's most-digit-d-like sample for each d."""
    torch.manual_seed(1000 + gen)
    model = build_vae()
    model.load_state_dict(torch.load(RUN_ROOT / f"gen{gen}" / "weights.pt", map_location=device))
    model.eval()

    z = torch.randn(POOL, latent_dim, device=device)
    imgs = model.decoder(z).clamp(0, 1)                 # (POOL,1,28,28)
    probs = fid_net(imgs).softmax(dim=1)                # (POOL,10)

    picks = []
    for d in range(N_DIGITS):
        best = probs[:, d].argmax().item()              # most digit-d-looking sample
        picks.append(imgs[best].cpu())
    return torch.stack(picks)


def main():
    train_images, _ = load_mnist()
    fid_net = get_fid_net(train_images, device)         # reuses cached CNN

    grid = [best_per_digit(g, fid_net) for g in range(GEN_START, GEN_END + 1)]  # list of (10,1,28,28)

    fig, axes = plt.subplots(N_GENS, N_DIGITS, figsize=(N_DIGITS, N_GENS))
    for r in range(N_GENS):
        for c in range(N_DIGITS):
            ax = axes[r, c]
            ax.imshow(grid[r][c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(str(c), fontsize=13, pad=6)
            if c == 0:
                ax.set_ylabel(f"gen {GEN_START + r}", fontsize=11, rotation=0, ha="right", va="center", labelpad=12)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
