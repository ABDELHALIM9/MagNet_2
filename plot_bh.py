"""
B-H loop comparison and True-H vs Predicted-H plots on the test set.

Example:
    python plot_bh.py \
        --ckpt /content/drive/MyDrive/BTH_MAG/models/A_run1/best.pt \
        --test /content/drive/MyDrive/BTH_MAG/processed/A/test.npz \
        --out-dir /content/drive/MyDrive/BTH_MAG/models/A_run1

Saves bh_loops.png and h_parity.png in --out-dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference import load_model, predict_H
from src.preprocessing import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test", required=True, help="processed test.npz (normalized)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--max-points", type=int, default=20000,
                   help="max points in the parity scatter")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, stats = load_model(args.ckpt)
    ds = load_dataset(args.test)

    # test.npz is normalized -> convert back to physical units
    B = ds.B * stats.B_max          # tesla
    T = ds.T * stats.T_max          # deg C
    H_true = ds.H * stats.H_max     # A/m
    H_pred = predict_H(model, stats, B, T)

    # per-window relative error, used to pick representative windows
    sre = np.linalg.norm(H_pred - H_true, axis=1) / np.maximum(
        np.linalg.norm(H_true, axis=1), 1e-12)
    order = np.argsort(sre)
    print(f"Windows: {len(sre)} | SRE mean {sre.mean():.4f} "
          f"median {np.median(sre):.4f} max {sre.max():.4f}")

    # ---------- 1) B-H loops: best / typical / worst windows ----------
    quantiles = [0.0, 0.25, 0.50, 0.75, 0.95, 1.0]
    names = ["best", "25%", "median", "75%", "95%", "worst"]
    picks = [order[int(round(q * (len(order) - 1)))] for q in quantiles]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, idx, name in zip(axes.ravel(), picks, names):
        ax.plot(H_true[idx], B[idx] * 1e3, "b-", lw=1.8, label="True")
        ax.plot(H_pred[idx], B[idx] * 1e3, "r--", lw=1.5, label="Predicted")
        ax.set_title(f"{name} | window {idx} | T={T[idx]:.0f} C | SRE={sre[idx]:.3f}")
        ax.set_xlabel("H (A/m)")
        ax.set_ylabel("B (mT)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("B-H loop comparison (True vs Predicted)")
    fig.tight_layout()
    fig.savefig(out_dir / "bh_loops.png", dpi=150)

    # ---------- 2) True H vs Predicted H (parity plot) ----------
    rng = np.random.default_rng(0)
    yt, yp = H_true.ravel(), H_pred.ravel()
    sel = rng.choice(len(yt), size=min(args.max_points, len(yt)), replace=False)

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    ax2.scatter(yt[sel], yp[sel], s=3, alpha=0.3)
    ax2.plot(lim, lim, "k--", lw=1, label="Ideal (y = x)")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("True H (A/m)")
    ax2.set_ylabel("Predicted H (A/m)")
    ax2.set_title(f"True vs Predicted H  (R2 = {r2:.4f})")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "h_parity.png", dpi=150)

    print(f"R2 = {r2:.4f}")
    print(f"Saved: {out_dir / 'bh_loops.png'}, {out_dir / 'h_parity.png'}")
    plt.show()


if __name__ == "__main__":
    main()