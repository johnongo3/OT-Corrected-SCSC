"""Compare collapse-mitigation strategies for the self-consuming VAE.

Reads the summary.csv of each run and plots the honest model-quality metric --
the FID of each generation's model's OWN prior samples vs the real test set
(`raw_fid` where a run has it, else `fid`; for uncorrected runs these are equal).
This is what you actually get when you sample from the trained VAE, so it is the
fair yardstick across methods that use their synthetic set differently.

Compared:
    baseline (lambda=0)     pure synthetic replacement (collapses)
    OT lambda=1.0           every synthetic latent snapped onto the anchor manifold
    anchor-augment (512)    the same 512 real anchor images retained in the
                            training set each generation (no OT)

    python vae_compare_mitigations.py   ->  Imgs/vae_mitigation_comparison.png
"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 can't print the λ labels

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("baseline (λ=0)", "runs/vae_mnist", "#8c8c8c", "o"),
    ("OT λ=1.0", "runs/vae_mnist_ot_1.0", "#1f77b4", "s"),
    ("anchor-augment (512)", "runs/vae_mnist_anchor512", "#d62728", "^"),
]
OUT = Path("Imgs") / "vae_mitigation_comparison.png"
TAIL = 10  # generations averaged for the summary verdict


def load(path):
    """Return (gens, model_fid, model_pixel_std, test_bce) using each model's own
    prior-sample metrics (raw_* where present, else the plain columns)."""
    gens, fids, pstds, bces = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            fids.append(float(row.get("raw_fid") or row["fid"]))
            pstds.append(float(row.get("raw_pixel_std") or row["sample_pixel_std"]))
            bces.append(float(row["test_bce"]))
    return gens, fids, pstds, bces


def main():
    loaded = []
    for label, path, color, marker in RUNS:
        csv_path = Path(path) / "summary.csv"
        if not csv_path.exists():
            print(f"[skip] {label}: {csv_path} not found")
            continue
        loaded.append((label, color, marker, *load(csv_path)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for label, color, marker, gens, fids, pstds, bces in loaded:
        axes[0].plot(gens, fids, marker=marker, ms=4, color=color, label=label)
        axes[1].plot(gens, pstds, marker=marker, ms=4, color=color, label=label)
        axes[2].plot(gens, bces, marker=marker, ms=4, color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("Model sample FID (↓ better)")
    axes[1].set_title("Sample pixel std / diversity (↑ better)")
    axes[2].set_title("Real-test recon BCE (↓ better)")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=8)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"Wrote {OUT}\n")

    # Verdict: mean over the last TAIL generations (steady-state behavior).
    print(f"{'method':<24}{'FID (last %d)' % TAIL:>16}{'pixel_std':>14}{'BCE':>12}")
    print("-" * 66)
    for label, _c, _m, gens, fids, pstds, bces in loaded:
        f = sum(fids[-TAIL:]) / len(fids[-TAIL:])
        p = sum(pstds[-TAIL:]) / len(pstds[-TAIL:])
        b = sum(bces[-TAIL:]) / len(bces[-TAIL:])
        print(f"{label:<24}{f:>16.2f}{p:>14.4f}{b:>12.2f}")


if __name__ == "__main__":
    main()
