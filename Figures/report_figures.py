"""Result figures for the technical report (Report/Images/results/).

Wasserstein-from-generation-0 is the headline metric throughout; FID is not
plotted. Each experiment family gets one figure, with every run in that family
overlaid:

    lambda_sweep.png    correction strength lambda, 5000 anchors, exact EMD
    anchor_sweep.png    anchor-pool size at lambda = 0.8, minibatch-OT (K = 1000)
    augmentation.png    real-data augmentation, with and without correction
    cross_dataset.png   MNIST / Fashion-MNIST / CelebA at lambda = 0.8

    python Figures/report_figures.py
"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "Report" / "Images" / "results"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "figure.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

W = "w_from_gen0"
DIV = "sample_pixel_std"
BCE = "test_bce"


def load(run, cols=(W, DIV, BCE)):
    """summary.csv for `run`, falling back to w_backfill.csv for runs whose
    Wasserstein column was recomputed after the fact."""
    rows = list(csv.DictReader(open(RUNS / run / "summary.csv")))
    out = {"generation": [int(r["generation"]) for r in rows]}
    for c in cols:
        if c in rows[0]:
            out[c] = [float(r[c]) for r in rows]
    if W not in out:
        bf = RUNS / run / "w_backfill.csv"
        if bf.exists():
            b = {int(r["generation"]): float(r[W]) for r in csv.DictReader(open(bf))}
            out[W] = [b.get(g, float("nan")) for g in out["generation"]]
    return out


def panels(fig_title, series, out_name, legend_title=None, wide=False):
    """Three panels -- Wasserstein drift, diversity, reconstruction -- with every
    run in `series` = [(label, run, color, style), ...] overlaid on each."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0 if wide else 12.0, 3.5))
    for label, run, color, ls in series:
        d = load(run)
        g = d["generation"]
        for ax, key in zip(axes, (W, DIV, BCE)):
            if key in d:
                ax.plot(g, d[key], color=color, ls=ls, lw=1.5, label=label)

    axes[0].set_ylabel(r"$W_2$ from gen 0")
    axes[0].set_title("Distributional drift  (down = better)")
    axes[1].set_ylabel("sample pixel std")
    axes[1].set_title("Sample diversity  (up = better)")
    axes[2].set_ylabel("test BCE")
    axes[2].set_title("Real-test reconstruction  (down = better)")
    for ax in axes:
        ax.set_xlabel("Generation")

    # Shared legend below the panels -- keeps it clear of every curve.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, title=legend_title, frameon=False,
                     loc="upper center", bbox_to_anchor=(0.5, 0.02),
                     ncol=min(len(labels), 6))
    if legend_title:
        leg.get_title().set_fontsize(9)

    fig.suptitle(fig_title, fontsize=12, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / out_name}")


def tail_table(title, series, tail=10):
    """Steady-state summary: mean of the last `tail` generations."""
    print(f"\n{title}")
    print(f"  {'run':<34}{'W2 (tail)':>12}{'div (tail)':>12}{'BCE (tail)':>12}")
    print("  " + "-" * 70)
    for label, run, *_ in series:
        d = load(run)
        def m(k):
            if k not in d:
                return float("nan")
            v = [x for x in d[k][-tail:] if x == x]
            return sum(v) / len(v) if v else float("nan")
        print(f"  {label:<34}{m(W):>12.4f}{m(DIV):>12.4f}{m(BCE):>12.1f}")


def main():
    # ---- 1. correction strength ------------------------------------------
    cmap = plt.get_cmap("viridis")
    lambdas = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    lam_series = [(rf"$\lambda$ = {l}", f"EXP_corr_sink_{l}",
                   cmap(i / (len(lambdas) - 1)), "-")
                  for i, l in enumerate(lambdas)]
    panels("Correction strength: 5000 anchors, exact EMD, MNIST",
           lam_series, "lambda_sweep.png", legend_title="correction strength")
    tail_table("Correction-strength sweep (mean of last 10 generations)", lam_series)

    # ---- 2. anchor-pool size ---------------------------------------------
    anchors = ["10", "50", "100", "1000", "10000", "60000"]
    anch_series = [(f"{int(a):,} anchors", f"EXP_anch_{a}",
                    cmap(i / (len(anchors) - 1)), "-")
                   for i, a in enumerate(anchors)]
    panels(r"Anchor-pool size: $\lambda$ = 0.8, minibatch-OT (K = 1000), MNIST",
           anch_series, "anchor_sweep.png", legend_title="anchor pool")
    tail_table("Anchor-pool sweep (mean of last 10 generations)", anch_series)

    # ---- 3. augmentation --------------------------------------------------
    aug_series = [
        ("no augmentation (pure synthetic)", "vae_mnist",              "#4c4c4c", "-"),
        ("512 real anchors retained",        "vae_mnist_anchor512",    "#c1662f", "-"),
        ("full real pool + synthetic",       "vae_mnist_aug_raw",      "#2f6fb1", "-"),
        ("full real pool + corrected",       "vae_mnist_aug_ot_full",  "#2f8f4e", "--"),
    ]
    panels("Real-data augmentation, with and without correction (MNIST)",
           aug_series, "augmentation.png", wide=True)
    tail_table("Augmentation family (mean of last 10 generations)", aug_series, tail=5)

    # ---- 4. cross-dataset -------------------------------------------------
    ds = [
        ("MNIST",         "EXP_corr_sink_0.8", "EXP_corr_sink_0.0", "#2f6fb1"),
        ("Fashion-MNIST", "fashion_ot_08",     "fashion_baseline",  "#c1662f"),
        ("CelebA",        "celeba_ot_08",      "celeba_baseline",   "#2f8f4e"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for label, corr, base, color in ds:
        c = load(corr, cols=(W,))
        b = load(base, cols=(W,))
        axes[0].plot(c["generation"], c[W], color=color, lw=1.6, label=f"{label}")
        axes[0].plot(b["generation"], b[W], color=color, lw=1.2, ls=":", alpha=0.75)
        # normalized: each dataset against its own uncorrected end-state
        scale = max(b[W]) or 1.0
        axes[1].plot(c["generation"], [v / scale for v in c[W]], color=color, lw=1.6,
                     label=f"{label} (corrected)")
        axes[1].plot(b["generation"], [v / scale for v in b[W]], color=color, lw=1.2,
                     ls=":", alpha=0.75)

    axes[0].set_ylabel(r"$W_2$ from gen 0")
    axes[0].set_title("Absolute drift")
    axes[1].set_ylabel(r"$W_2$ / max uncorrected drift")
    axes[1].set_title("Normalised within dataset")
    for ax in axes:
        ax.set_xlabel("Generation")
    axes[0].plot([], [], color="k", lw=1.6, label="corrected ($\\lambda$ = 0.8)")
    axes[0].plot([], [], color="k", lw=1.2, ls=":", label="uncorrected")
    axes[0].legend(frameon=False, ncol=1)
    fig.suptitle(r"Distributional drift across datasets, $\lambda$ = 0.8", fontsize=12, y=1.03)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cross_dataset.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'cross_dataset.png'}")

    print("\nCross-dataset final W2 (corrected vs uncorrected):")
    for label, corr, base, _ in ds:
        c, b = load(corr, cols=(W,)), load(base, cols=(W,))
        print(f"  {label:<16} corrected {c[W][-1]:.4f}   uncorrected {b[W][-1]:.4f}"
              f"   ratio {c[W][-1] / b[W][-1]:.3f}")


if __name__ == "__main__":
    main()
