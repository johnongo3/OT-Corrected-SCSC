"""Plot model-collapse metrics across generations from a summary.json.

Reads the ``summary.json`` written by ``self_consuming_loop.py`` and saves a
grid of metric-vs-generation charts to ``<output-root>/collapse_curves.png``.
Where a real-data reference exists (``real_*``), it is drawn as a dashed
horizontal line so the drift away from the real distribution is obvious.

Usage:
    python plot_metrics.py --output-root runs/experiment1
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs without needing a display/Tk
import matplotlib.pyplot as plt

# (metric_key, human title, matching real-reference key or None).
PANELS = [
    ("fid", "FID vs real (lower = better)", None),
    ("best_test_neg_elbo", "Best test neg-ELBO", None),
    ("feature_variance", "Feature variance (diversity)", "real_feature_variance"),
    ("mean_pairwise_feat_dist", "Mean pairwise feat. dist.",
     "real_mean_pairwise_feat_dist"),
    ("class_entropy", "Class-coverage entropy", "real_class_entropy"),
    ("class_kl", "Class-dist. KL vs real", None),
    ("pixel_std_mean", "Mean per-pixel std", "real_pixel_std_mean"),
    ("pixel_mean_l1", "Pixel-mean L1 vs real", None),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="runs/experiment1")
    args = p.parse_args()

    with open(os.path.join(args.output_root, "summary.json")) as f:
        summary = json.load(f)
    series = summary["series"]
    gens = series["generation"]

    n = len(PANELS)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = axes.ravel()

    for ax, (key, title, ref_key) in zip(axes, PANELS):
        if key not in series:
            ax.set_visible(False)
            continue
        ax.plot(gens, series[key], marker="o", color="#c1121f", label=key)
        if ref_key and ref_key in series:
            # Reference is constant across generations; one value suffices.
            ax.axhline(series[ref_key][0], ls="--", color="#457b9d",
                       label="real data")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("generation")
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Model collapse across self-consuming generations", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.output_root, "collapse_curves.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
