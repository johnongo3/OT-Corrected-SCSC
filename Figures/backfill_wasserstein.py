"""Recompute `w_from_gen0` for runs that predate the metric.

Early runs (the augmentation family) logged FID/diversity/BCE but not the
sliced-Wasserstein drift. Every generation's `weights.pt` was saved, so the metric
can be reconstructed exactly as `vae_self_consuming.py` defines it: draw prior
samples from each generation's model, embed them with the fixed feature CNN, and
take the sliced W2 against generation 0's embeddings.

    python Figures/backfill_wasserstein.py runs/vae_mnist runs/vae_mnist_aug_raw ...

Writes `w_backfill.csv` (generation, w_from_gen0) into each run directory. The
values are recomputed rather than logged, so they carry fresh prior draws; with
10,000 samples the distance is stable to ~1e-3 across seeds.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ot import sliced_wasserstein_distance

import vae_self_consuming as vsc

N_SAMPLES = 10000
N_PROJECTIONS = 100
SEED = 0


def backfill(run_dir, fid_net, device):
    run_dir = Path(run_dir)
    gens = sorted((p for p in run_dir.glob("gen*") if (p / "weights.pt").exists()),
                  key=lambda p: int(p.name[3:]))
    if not gens:
        print(f"  [skip] {run_dir}: no gen*/weights.pt")
        return

    gen0_feats, rows = None, []
    for gen_dir in gens:
        gen = int(gen_dir.name[3:])
        model = vsc.build_vae(device)
        model.load_state_dict(torch.load(gen_dir / "weights.pt", map_location=device))
        model.eval()

        torch.manual_seed(SEED)  # same prior draw for every generation
        samples = vsc.sample_images(model, N_SAMPLES, device)
        feats = vsc.fid_features(fid_net, samples, device)
        if gen0_feats is None:
            gen0_feats = feats
        w = float(sliced_wasserstein_distance(feats, gen0_feats,
                                              n_projections=N_PROJECTIONS, seed=SEED))
        rows.append({"generation": gen, "w_from_gen0": round(w, 4)})
        print(f"    gen {gen:>2}  w_from_gen0 = {w:.4f}")

    out = run_dir / "w_backfill.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "w_from_gen0"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out}")


def main():
    runs = sys.argv[1:] or ["runs/vae_mnist", "runs/vae_mnist_anchor512",
                            "runs/vae_mnist_aug_raw", "runs/vae_mnist_aug_ot_full"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vsc.DATASET, vsc.LATENT = "mnist", vsc.latent_dim
    train_images, _ = vsc.load_gray("mnist")
    fid_net = vsc.get_feature_net("mnist", train_images, device)

    for r in runs:
        print(f"[{r}]")
        backfill(r, fid_net, device)


if __name__ == "__main__":
    main()
