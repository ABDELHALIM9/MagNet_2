"""
Predict H from raw (physical-unit) B and T using a trained checkpoint.
Works with both model types (gru, mss_gru).

Python API:
    from inference import load_model, predict_H
    model, stats = load_model("models/A/best.pt")
    H = predict_H(model, stats, B, T)        # B: [N, L] or [L] in tesla, T in deg C
                                             # H returned in A/m

CLI:
    python inference.py --ckpt models/A/best.pt --B B.csv --T T.csv --out H_pred.csv
    (B.csv: one sequence per row; T.csv: one temperature per sequence)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models import build_model_from_config
from src.preprocessing import NormalizationStats


def load_model(ckpt_path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model_from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = ckpt["normalization"]
    stats = NormalizationStats(B_max=n["B_max"], T_max=n["T_max"], H_max=n["H_max"])
    return model, stats


@torch.no_grad()
def predict_H(model: nn.Module, stats: NormalizationStats, B, T, batch_size: int = 256):
    """Raw B [N, L] (or [L]) and T [N] (or scalar) -> raw H with the same shape as B."""
    device = next(model.parameters()).device

    B = np.atleast_2d(np.asarray(B, dtype=np.float32))
    T = np.broadcast_to(np.asarray(T, dtype=np.float32).reshape(-1), (B.shape[0],))

    B_n, T_n = stats.normalize(B, np.ascontiguousarray(T))
    B_t = torch.from_numpy(B_n).float()
    T_t = torch.from_numpy(T_n).float()

    out = []
    for i in range(0, len(B_t), batch_size):
        pred = model(B_t[i:i + batch_size].to(device), T_t[i:i + batch_size].to(device))
        out.append(pred.cpu().numpy())

    return stats.denormalize_H(np.concatenate(out, axis=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--B", required=True, help="CSV, one B sequence per row (tesla)")
    p.add_argument("--T", required=True, help="CSV, one temperature per sequence (deg C)")
    p.add_argument("--out", default="H_pred.csv")
    args = p.parse_args()

    model, stats = load_model(args.ckpt)
    B = pd.read_csv(args.B, header=None).values
    T = pd.read_csv(args.T, header=None).values[:, 0]

    H = predict_H(model, stats, B, T)
    pd.DataFrame(H).to_csv(args.out, header=False, index=False)
    print(f"Saved {H.shape} predictions to {args.out}")


if __name__ == "__main__":
    main()