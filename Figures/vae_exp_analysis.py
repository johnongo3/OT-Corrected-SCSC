"""Analyse an experiment sweep: per-generation metrics for a set of runs, coloured
by the swept parameter, plus a steady-state (last-10-generation) table and the
best setting by Wasserstein-from-gen0.

    python vae_exp_analysis.py "runs/EXP_corr_*" "lambda"    Imgs/exp_corr.png
    python vae_exp_analysis.py "runs/EXP_anch_*" "n_anchors" Imgs/exp_anch.png

Metrics use each generation's model's OWN samples (raw_* where present). The
primary panel/verdict is Wasserstein-from-gen0 (drift; 0 at gen 0, up = collapse).
"""

import sys
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAIL = 10


def param_val(path):
    s = Path(path).name.split("_")[-1]
    try:
        return float(s)
    except ValueError:
        return s


def load(path):
    rows = list(csv.DictReader(open(Path(path) / "summary.csv")))
    g = [int(r["generation"]) for r in rows]
    w = [float(r.get("w_from_gen0", "nan")) for r in rows]
    fid = [float(r.get("raw_fid") or r["fid"]) for r in rows]
    pstd = [float(r.get("raw_pixel_std") or r["sample_pixel_std"]) for r in rows]
    bce = [float(r["test_bce"]) for r in rows]
    return g, w, fid, pstd, bce


def main():
    pattern, plabel, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dirs = sorted((d for d in glob.glob(pattern) if (Path(d) / "summary.csv").exists()),
                  key=param_val)
    if not dirs:
        print(f"no runs with summary.csv match {pattern}")
        return
    data = [(param_val(d), *load(d)) for d in dirs]
    colors = plt.cm.viridis(np.linspace(0, 1, len(dirs)))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for (v, g, w, fid, pstd, bce), c in zip(data, colors):
        ax[0, 0].plot(g, w, color=c, marker=".", ms=4, label=f"{plabel}={v}")
        ax[0, 1].plot(g, fid, color=c, marker=".", ms=4)
        ax[1, 0].plot(g, pstd, color=c, marker=".", ms=4)
        ax[1, 1].plot(g, bce, color=c, marker=".", ms=4)
    ax[0, 1].set_yscale("log")
    for a, t in zip(ax.flat, ["Wasserstein from gen-0 (down = less drift)",
                              "Model sample FID (log, down better)",
                              "Sample pixel std / diversity (up better)",
                              "Real-test recon BCE (down better)"]):
        a.set_title(t, fontsize=11)
        a.set_xlabel("Generation")
    ax[0, 0].legend(fontsize=7, title=plabel)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}\n")

    print(f"{plabel:<14}{'W(last%d)' % TAIL:>12}{'FID':>10}{'pixel_std':>12}{'BCE':>10}")
    print("-" * 58)
    best = None
    for v, g, w, fid, pstd, bce in data:
        n = min(TAIL, len(w))
        mw = sum(w[-n:]) / n
        print(f"{str(v):<14}{mw:>12.3f}{sum(fid[-n:])/n:>10.1f}"
              f"{sum(pstd[-n:])/n:>12.4f}{sum(bce[-n:])/n:>10.2f}")
        if best is None or mw < best[1]:
            best = (v, mw)
    print(f"\nbest by Wasserstein-from-gen0: {plabel}={best[0]} (W={best[1]:.3f})")


if __name__ == "__main__":
    main()
