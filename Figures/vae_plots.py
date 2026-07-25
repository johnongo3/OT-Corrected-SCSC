"""Sample and latent-interpolation plots for the trained VAE checkpoint.

Loads outputs/vae_trained.pt and writes to Imgs/:
    vae_samples.png        8x8 grid decoded from z ~ N(0, I)
    vae_interpolation.png  rows interpolating between pairs of test digits
"""

import torch
from pathlib import Path
from torchvision import datasets
from torchvision.utils import save_image

from vae import load_checkpoint, checkpoint_path, latent_dim, device

out_dir = Path("Imgs")
n_steps = 12       # columns per interpolation row
n_pairs = 8        # interpolation rows


@torch.no_grad()
def sample_grid(model, path, n=64, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n, latent_dim, device=device)
    imgs = model.decoder(z)
    save_image(imgs, path, nrow=8)


@torch.no_grad()
def interpolation_grid(model, path, seed=0):
    test = datasets.MNIST(root="data", train=False, download=True)
    images = test.data.float().div(255.0)
    labels = test.targets

    # One endpoint image per digit class, then interpolate d -> d+1 (…9 -> 0)
    # so each row morphs between two different digits.
    torch.manual_seed(seed)
    endpoints = []
    for d in range(10):
        idx = (labels == d).nonzero(as_tuple=True)[0]
        endpoints.append(images[idx[torch.randint(len(idx), (1,))]].squeeze(0))

    rows = []
    for r in range(n_pairs):
        a, b = endpoints[r], endpoints[(r + 1) % 10]
        x = torch.stack([a, b]).view(2, -1).to(device)
        mu, _ = model.encoder(x)  # interpolate between posterior means
        t = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
        z = (1 - t) * mu[0] + t * mu[1]
        rows.append(model.decoder(z).cpu())
    save_image(torch.cat(rows), path, nrow=n_steps)


def main():
    model = load_checkpoint(checkpoint_path, device)
    out_dir.mkdir(exist_ok=True)
    sample_grid(model, out_dir / "vae_samples.png")
    interpolation_grid(model, out_dir / "vae_interpolation.png")
    print(f"Wrote {out_dir / 'vae_samples.png'} and {out_dir / 'vae_interpolation.png'}")


if __name__ == "__main__":
    main()
