"""Animate a self-consuming run's per-generation sample grids into a GIF.

Every generation writes an 8x8 sample grid to ``<run-root>/gen<k>/samples.png``
(the propagated set), and OT runs additionally write ``samples_raw.png`` (the
model's own pre-correction prior samples). This stitches a generation range of
those grids into an animated GIF, labelled by generation, so you can watch the
collapse (or the lack of it) unfold.

    # baseline collapse, all 50 generations
    python vae_collapse_gif.py --run-root runs/vae_mnist --start 0 --end 49

    # an OT run's RAW samples (what the drifting model actually generates)
    python vae_collapse_gif.py --run-root runs/vae_mnist_ot_1.0 --which samples_raw.png

    # a short, slow clip
    python vae_collapse_gif.py --start 0 --end 20 --ms 500 --scale 3

Note: each generation is an independently trained VAE, so the latent spaces are
NOT aligned across frames -- the samples change identity frame to frame. Watch
the aggregate sharpness/diversity, not individual digits.
"""

from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont

LABEL_H = 22


def _font(size=16):
    """Scalable font via matplotlib's bundled DejaVuSans; fall back to PIL's default."""
    try:
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def load_frame(run_root: Path, gen: int, which: str, scale: int, font) -> Image.Image:
    """Open one generation's grid, upscale it, and add a 'gen k' label bar on top."""
    path = run_root / f"gen{gen}" / which
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    canvas = Image.new("RGB", (img.width, img.height + LABEL_H), "white")
    canvas.paste(img, (0, LABEL_H))
    ImageDraw.Draw(canvas).text((6, 3), f"gen {gen}", fill="black", font=font)
    return canvas


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", default="runs/vae_mnist", help="Run directory containing gen*/ dirs.")
    p.add_argument("--start", type=int, default=0, help="First generation (inclusive).")
    p.add_argument("--end", type=int, default=49, help="Last generation (inclusive).")
    p.add_argument("--which", default="samples.png",
                   help="Grid to animate: samples.png (propagated) or samples_raw.png (pre-correction).")
    p.add_argument("--out", default=None, help="Output .gif path (default: Imgs/<run>_<which>_<a>_<b>.gif).")
    p.add_argument("--ms", type=int, default=250, help="Milliseconds per frame.")
    p.add_argument("--scale", type=int, default=2, help="Nearest-neighbour upscale factor.")
    p.add_argument("--loop", type=int, default=0, help="Loop count; 0 = forever.")
    args = p.parse_args()

    run_root = Path(args.run_root)
    if args.out:
        out = Path(args.out)
    else:
        stem = args.which.replace(".png", "")
        out = Path("Imgs") / f"{run_root.name}_{stem}_{args.start}_{args.end}.gif"

    font = _font()
    frames = [load_frame(run_root, g, args.which, args.scale, font)
              for g in range(args.start, args.end + 1)]

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.ms, loop=args.loop)
    print(f"Wrote {out}  ({len(frames)} frames, {args.ms}ms each, {frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == "__main__":
    main()
