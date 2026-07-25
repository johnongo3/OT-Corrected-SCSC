"""Animate how each generation's VAE reconstructs a FIXED set of real test digits.

Complements ``vae_collapse_gif.py`` (which animates prior samples). Because the
input here is a fixed row of real images -- one exemplar per digit class 0-9 --
the frames are directly comparable: the top row (the real originals) never
changes, and the bottom row is that generation's reconstruction of them. So you
watch the SAME digits degrade, rather than unrelated samples flickering past.

Reconstructions decode the posterior MEAN (not a sampled z), so the only thing
changing between frames is the model itself -- no sampling noise.

Reconstruction PNGs are not saved by the loop, so each frame is rendered from
``<run-root>/gen<k>/weights.pt``.

    python vae_recon_gif.py --run-root runs/vae_mnist --start 0 --end 49
    python vae_recon_gif.py --run-root runs/vae_mnist_ot_1.0 --ms 400 --scale 4
"""

from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import datasets
from torchvision.utils import make_grid

from vae import device
from vae_self_consuming import build_vae
from vae_collapse_gif import _font, LABEL_H

DATA_ROOT = "data"


def class_exemplars():
    """One fixed real test image per digit class 0-9 -> (10,1,28,28) in [0,1]."""
    test = datasets.MNIST(root=DATA_ROOT, train=False, download=True)
    images = test.data.float().div(255.0).unsqueeze(1)
    labels = test.targets
    idx = [(labels == d).nonzero(as_tuple=True)[0][0].item() for d in range(10)]
    return images[idx]


@torch.no_grad()
def reconstruct(run_root: Path, gen: int, originals: torch.Tensor) -> torch.Tensor:
    """Load gen k's VAE and reconstruct `originals` from the posterior mean."""
    weights = run_root / f"gen{gen}" / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    model = build_vae(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    x = originals.to(device).view(len(originals), -1)
    mean, _ = model.encoder(x)                       # posterior mean -> deterministic
    return model.decoder(mean).view(-1, 1, 28, 28).clamp(0, 1).cpu()


def to_frame(originals, recons, gen, scale, font) -> Image.Image:
    """2x10 grid (real on top, reconstruction below) + a 'gen k' label bar."""
    grid = make_grid(torch.cat([originals, recons]), nrow=10, padding=2)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    canvas = Image.new("RGB", (img.width, img.height + LABEL_H), "white")
    canvas.paste(img, (0, LABEL_H))
    ImageDraw.Draw(canvas).text(
        (6, 3), f"gen {gen}   (top: real   bottom: reconstruction)", fill="black", font=font)
    return canvas


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", default="runs/vae_mnist", help="Run directory containing gen*/ dirs.")
    p.add_argument("--start", type=int, default=0, help="First generation (inclusive).")
    p.add_argument("--end", type=int, default=49, help="Last generation (inclusive).")
    p.add_argument("--out", default=None, help="Output .gif (default: Imgs/<run>_recon_<a>_<b>.gif).")
    p.add_argument("--ms", type=int, default=250, help="Milliseconds per frame.")
    p.add_argument("--scale", type=int, default=3, help="Nearest-neighbour upscale factor.")
    p.add_argument("--loop", type=int, default=0, help="Loop count; 0 = forever.")
    args = p.parse_args()

    run_root = Path(args.run_root)
    out = Path(args.out) if args.out else (
        Path("Imgs") / f"{run_root.name}_recon_{args.start}_{args.end}.gif")

    originals = class_exemplars()
    font = _font()
    frames = [to_frame(originals, reconstruct(run_root, g, originals), g, args.scale, font)
              for g in range(args.start, args.end + 1)]

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.ms, loop=args.loop)
    print(f"Wrote {out}  ({len(frames)} frames, {args.ms}ms each, "
          f"{frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == "__main__":
    main()
