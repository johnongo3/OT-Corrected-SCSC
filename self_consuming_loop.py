"""Self-consuming (recursive) training loop for the model-collapse study.

Trains a chain of VAE generations where each generation is trained on the
*synthetic* output of the previous one:

    gen 0  <- real Simpsons-MNIST images        (data_source="folder")
    gen 1  <- gen0/synthetic_data.npz
    gen 2  <- gen1/synthetic_data.npz
    ...

Every generation writes its own artifacts (checkpoint, sample grids, metrics)
into ``<output-root>/gen<k>/``. After the chain finishes, all per-generation
collapse metrics are aggregated into ``<output-root>/summary.json`` and
``<output-root>/summary.csv`` so the degradation can be plotted against the
generation index later (see ``plot_metrics.py``).

Usage (you run this; I only wrote it):

    python self_consuming_loop.py --generations 10 --epochs 50 \
        --output-root runs/experiment1

Resuming: pass ``--resume`` to reuse any generation whose ``metrics.json``
already exists instead of retraining it (handy for long chains).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from vae import add_sharpness_args, cfg_from_args, train_one_generation

# Scalar metrics pulled into summary.csv / the plottable "series" dict. Each real_*
# entry is the real-data reference value for its metric, constant across gens.
SCALAR_METRICS = [
    "fid",
    "active_units",
    "feature_variance",
    "real_feature_variance",
    "mean_pairwise_feat_dist",
    "real_mean_pairwise_feat_dist",
    "class_entropy",
    "real_class_entropy",
    "class_kl",
    "pixel_std_mean",
    "real_pixel_std_mean",
    "pixel_mean_l1",
]


def flatten_report(report: dict) -> dict:
    """Pull the plottable scalars out of one generation's full report."""
    row = {
        "generation": report["generation"],
        "data_source": report["data_source"],
        "train_seconds": report["train_seconds"],
        "best_test_neg_elbo": report["best_test_neg_elbo"],
        "test_recon": report["final_test"]["test_recon"],
        "test_kl": report["final_test"]["test_kl"],
    }
    for key in SCALAR_METRICS:
        row[key] = report["collapse_metrics"][key]
    return row


def write_summary(output_root: str, rows: list[dict], config: dict) -> None:
    """Write summary.json (series keyed by metric) and summary.csv (one row/gen)."""
    rows = sorted(rows, key=lambda r: r["generation"])
    columns = list(rows[0].keys())

    # summary.json: a "series" dict maps each column to a per-generation list,
    # which is the shape plotting code wants (y-values indexed by generation).
    series = {col: [r[col] for r in rows] for col in columns}
    summary = {"config": config, "num_generations": len(rows),
               "series": series, "rows": rows}
    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_root, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generations", type=int, default=10,
                   help="Number of generations to train (gen 0 is the real-data model).")
    p.add_argument("--output-root", default="runs/experiment1")
    p.add_argument("--epochs", type=int, default=50,
                   help="Epochs per generation (kept identical across generations).")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-synth", type=int, default=8000,
                   help="Synthetic images each generation produces for the next.")
    p.add_argument("--kl-active-thresh", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; generation k uses seed + k.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse existing generations whose metrics.json is present.")
    add_sharpness_args(p)  # --width/--recon-loss/--lpips-*/--gan-*/--free-bits/...
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg_from_args(args)
    os.makedirs(args.output_root, exist_ok=True)
    print(f"Self-consuming loop: {args.generations} generations "
          f"| {args.epochs} epochs each | device={device}")
    print(f"  recon={train_cfg.recon_loss} lpips={train_cfg.lpips_weight} "
          f"gan={train_cfg.gan_weight} free_bits={train_cfg.free_bits} "
          f"beta={train_cfg.beta} warmup={train_cfg.kl_warmup_epochs}")
    print(f"Output root: {args.output_root}\n")

    config = {
        "generations": args.generations, "epochs": args.epochs,
        "latent_dim": args.latent_dim, "width": args.width, "beta": args.beta,
        "batch_size": args.batch_size, "lr": args.lr,
        "num_synth": args.num_synth, "kl_active_thresh": args.kl_active_thresh,
        "base_seed": args.seed,
        "recon_loss": args.recon_loss, "lpips_weight": args.lpips_weight,
        "gan_weight": args.gan_weight, "free_bits": args.free_bits,
        "kl_warmup_epochs": args.kl_warmup_epochs,
    }

    rows: list[dict] = []
    for gen in range(args.generations):
        gen_dir = os.path.join(args.output_root, f"gen{gen}")
        # gen 0 trains on real data; every later gen consumes its predecessor.
        if gen == 0:
            data_source = "folder"
        else:
            prev_npz = os.path.join(args.output_root, f"gen{gen - 1}",
                                    "synthetic_data.npz")
            if not os.path.exists(prev_npz):
                raise FileNotFoundError(
                    f"Missing synthetic data for gen {gen}: {prev_npz}. "
                    "Did the previous generation finish?")
            data_source = prev_npz

        metrics_path = os.path.join(gen_dir, "metrics.json")
        synth_path = os.path.join(gen_dir, "synthetic_data.npz")
        if args.resume and os.path.exists(metrics_path) and os.path.exists(synth_path):
            print(f"[resume] gen {gen}: reusing {metrics_path}")
            with open(metrics_path) as f:
                report = json.load(f)
        else:
            print(f"\n{'=' * 70}\nGeneration {gen}/{args.generations - 1}\n{'=' * 70}")
            report = train_one_generation(
                generation=gen,
                data_source=data_source,
                output_dir=gen_dir,
                device=device,
                epochs=args.epochs,
                latent_dim=args.latent_dim,
                beta=args.beta,
                batch_size=args.batch_size,
                lr=args.lr,
                num_synth=args.num_synth,
                kl_active_thresh=args.kl_active_thresh,
                seed=args.seed + gen,  # distinct but reproducible per generation
                width=args.width,
                train_cfg=train_cfg,
            )

        rows.append(flatten_report(report))
        # Rewrite the aggregate after every generation so a long run that is
        # interrupted still leaves a usable, up-to-date summary.
        write_summary(args.output_root, rows, config)

    print(f"\n{'=' * 70}")
    print("Self-consuming loop complete.")
    print(f"  summary.json -> {os.path.join(args.output_root, 'summary.json')}")
    print(f"  summary.csv  -> {os.path.join(args.output_root, 'summary.csv')}")
    print("\nFID by generation (collapse should make this rise):")
    for r in rows:
        print(f"  gen {r['generation']:2d}: fid={r['fid']:.3f}  "
              f"class_entropy={r['class_entropy']:.3f}  "
              f"feat_var={r['feature_variance']:.2f}")


if __name__ == "__main__":
    main()
