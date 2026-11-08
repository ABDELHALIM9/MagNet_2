
"""
Scientific evaluation plots for B-to-H prediction.

Generates:
1. figure_1_bh_loops.png
2. figure_2_h_waveforms.png
3. figure_3_h_parity.png
4. evaluation_metrics.json

Usage:
python plot_bh.py \
    --ckpt /content/drive/MyDrive/BTH_MAG/models/A_run1/best.pt \
    --test /content/drive/MyDrive/BTH_MAG/processed/A/test.npz \
    --out-dir /content/drive/MyDrive/BTH_MAG/models/A_run1 \
    --max-points 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference import load_model, predict_H
from src.preprocessing import load_dataset


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def to_numpy(x):
    """Convert arrays/tensors to NumPy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def calculate_metrics(y_true, y_pred):
    """Calculate global regression metrics."""

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    error = yp - yt

    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)

    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "R2": float(r2),
    }


def get_temperature(T, idx):
    """Get one representative temperature for a window."""

    if T.ndim == 1:
        return float(T[idx])

    return float(np.mean(T[idx]))


def prepare_arrays(ds, stats):
    """
    Convert normalized test data back to physical units.

    Assumption:
    B_norm = B / B_max
    T_norm = T / T_max
    H_norm = H / H_max

    Change these equations if your preprocessing uses
    a different normalization method.
    """

    B_norm = to_numpy(ds.B)
    T_norm = to_numpy(ds.T)
    H_norm = to_numpy(ds.H)

    B = B_norm * stats.B_max
    T = T_norm * stats.T_max
    H_true = H_norm * stats.H_max

    # Expected shapes: (N, L)
    B = np.squeeze(B)
    H_true = np.squeeze(H_true)
    T = np.squeeze(T)

    if B.ndim == 1:
        B = B[None, :]

    if H_true.ndim == 1:
        H_true = H_true[None, :]

    if T.ndim == 0:
        T = np.full((len(B),), float(T))

    if B.shape != H_true.shape:
        raise ValueError(
            f"B shape {B.shape} != H shape {H_true.shape}"
        )

    if len(B) != len(T):
        raise ValueError(
            f"Number of B windows ({len(B)}) "
            f"does not match T windows ({len(T)})"
        )

    return B, T, H_true


# ---------------------------------------------------------
# Figure 1: B-H Loop Comparison
# ---------------------------------------------------------

def plot_bh_loops(B, H_true, H_pred, T, sre, out_dir):

    order = np.argsort(sre)

    # Three representative examples
    selected = {
        "Best": order[0],
        "Median": order[len(order) // 2],
        "Worst": order[-1],
    }

    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 5.5),
        sharey=True
    )

    for ax, (name, idx) in zip(axes, selected.items()):

        # True B-H loop
        ax.plot(
            H_true[idx],
            B[idx] * 1000,
            color="blue",
            linewidth=2,
            label="True H"
        )

        # Predicted B-H loop
        ax.plot(
            H_pred[idx],
            B[idx] * 1000,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="Predicted H"
        )

        temp = get_temperature(T, idx)

        ax.set_title(
            f"{name} Window\n"
            f"Index={idx} | T={temp:.1f} °C | "
            f"SRE={sre[idx]:.4f}"
        )

        ax.set_xlabel("H (A/m)")
        ax.set_ylabel("B (mT)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "B-H Hysteresis Loop: True vs Predicted",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()

    path = out_dir / "figure_1_bh_loops.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    return selected


# ---------------------------------------------------------
# Figure 2: H Waveform Comparison
# ---------------------------------------------------------

def plot_h_waveforms(
    B, H_true, H_pred, T, sre, selected, out_dir, dt=None
):

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True
    )

    for ax, (name, idx) in zip(axes, selected.items()):

        length = H_true.shape[1]

        if dt is None:
            x = np.arange(length)
            xlabel = "Time Step"
        else:
            x = np.arange(length) * dt * 1000
            xlabel = "Time (ms)"

        ax.plot(
            x,
            H_true[idx],
            color="blue",
            linewidth=1.8,
            label="True H"
        )

        ax.plot(
            x,
            H_pred[idx],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Predicted H"
        )

        temp = get_temperature(T, idx)

        ax.set_title(
            f"{name} | Window {idx} | "
            f"T={temp:.1f} °C | SRE={sre[idx]:.4f}"
        )

        ax.set_ylabel("H (A/m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(xlabel)

    fig.suptitle(
        "H Waveform: True vs Predicted",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()

    path = out_dir / "figure_2_h_waveforms.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ---------------------------------------------------------
# Figure 3: H Parity Plot
# ---------------------------------------------------------

def plot_h_parity(H_true, H_pred, max_points, out_dir):

    yt = H_true.reshape(-1)
    yp = H_pred.reshape(-1)

    rng = np.random.default_rng(42)

    n_points = min(max_points, len(yt))

    selected = rng.choice(
        len(yt),
        size=n_points,
        replace=False
    )

    metrics = calculate_metrics(H_true, H_pred)

    lim_min = min(yt.min(), yp.min())
    lim_max = max(yt.max(), yp.max())

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        yt[selected],
        yp[selected],
        s=5,
        alpha=0.3,
        label="Predictions"
    )

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        linewidth=1.8,
        label="Ideal Prediction"
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel("True H (A/m)")
    ax.set_ylabel("Predicted H (A/m)")

    ax.set_title(
        f"H Parity Plot\n"
        f"R²={metrics['R2']:.4f} | "
        f"RMSE={metrics['RMSE']:.3f} A/m"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    path = out_dir / "figure_3_h_parity.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    return metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out-dir", default=".")

    parser.add_argument(
        "--max-points",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling interval in seconds. Optional."
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model and test dataset...")

    model, stats = load_model(args.ckpt)
    ds = load_dataset(args.test)

    # Convert normalized data to physical units
    B, T, H_true = prepare_arrays(ds, stats)

    print("Running inference...")

    # Same inference interface as the original script
    H_pred = to_numpy(
        predict_H(model, stats, B, T)
    )

    if H_pred.shape != H_true.shape:
        raise ValueError(
            f"Predicted H shape {H_pred.shape} "
            f"does not match True H shape {H_true.shape}"
        )

    # Per-window SRE
    numerator = np.linalg.norm(
        H_pred - H_true,
        axis=1
    )

    denominator = np.maximum(
        np.linalg.norm(H_true, axis=1),
        1e-12
    )

    sre = numerator / denominator

    print("\n========== TEST RESULTS ==========")
    print(f"Number of windows: {len(B)}")
    print(f"Sequence length: {B.shape[1]}")
    print(f"SRE Mean:   {sre.mean():.6f}")
    print(f"SRE Median: {np.median(sre):.6f}")
    print(f"SRE P95:    {np.percentile(sre, 95):.6f}")
    print(f"SRE Max:    {sre.max():.6f}")

    # Figure 1
    selected = plot_bh_loops(
        B, H_true, H_pred, T, sre, out_dir
    )

    # Figure 2
    plot_h_waveforms(
        B, H_true, H_pred, T, sre,
        selected, out_dir, args.dt
    )

    # Figure 3
    metrics = plot_h_parity(
        H_true,
        H_pred,
        args.max_points,
        out_dir
    )

    metrics.update({
        "num_windows": int(len(B)),
        "sequence_length": int(B.shape[1]),
        "SRE_mean": float(sre.mean()),
        "SRE_median": float(np.median(sre)),
        "SRE_p95": float(np.percentile(sre, 95)),
        "SRE_max": float(sre.max()),
    })

    metrics["selected_windows"] = {
        name: int(idx)
        for name, idx in selected.items()
    }

    metrics_path = out_dir / "evaluation_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n========== FINAL METRICS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nAll results saved successfully.")


if __name__ == "__main__":
    main()
