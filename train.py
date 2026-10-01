"""
Train a B+T -> H GRU for one material.

Example:
    python train.py --material A \
        --processed-root /content/drive/MyDrive/BTH_MAG/processed \
        --out-dir /content/drive/MyDrive/BTH_MAG/models/A

Defaults can also come from a YAML file:  --config configs/default.yaml
(command-line flags override the file).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.dataset import make_loaders
from src.metrics import (
    normalized_energy_relative_error,
    nrmse,
    sequence_relative_error,
    summarize,
)
from src.model import BTHGRU, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--material", type=str, default="A")
    p.add_argument("--processed-root", type=str, default="data/processed")
    p.add_argument("--out-dir", type=str, default="models/A")

    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bidirectional", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15,
                   help="early stopping patience (epochs); 0 disables")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # two-pass parse so the YAML file provides defaults, CLI overrides them
    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        p.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum, n = 0.0, 0
    sre_all, nere_all, nrmse_all = [], [], []

    for batch in loader:
        B, T, H = (batch[k].to(device) for k in ("B", "T", "H"))
        pred = model(B, T)
        mse_sum += ((pred - H) ** 2).mean(-1).sum().item()
        n += B.shape[0]
        sre_all.append(sequence_relative_error(pred, H).cpu())
        nere_all.append(normalized_energy_relative_error(pred, H, B).cpu())
        nrmse_all.append(nrmse(pred, H).cpu())

    return {
        "mse": mse_sum / max(n, 1),
        "sre": summarize(torch.cat(sre_all)),
        "nere": summarize(torch.cat(nere_all)),
        "nrmse": summarize(torch.cat(nrmse_all)),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, stats = make_loaders(
        args.processed_root,
        args.material,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = BTHGRU(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    print(f"Parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    best_val, best_epoch, history = float("inf"), -1, []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, count = 0.0, 0

        for batch in train_loader:
            B, T, H = (batch[k].to(device) for k in ("B", "T", "H"))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(B, T), H)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * B.shape[0]
            count += B.shape[0]

        train_loss = running / count
        val = evaluate(model, val_loader, device)
        scheduler.step(val["mse"])

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val["mse"],
            "val_sre_mean": val["sre"]["mean"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch:03d} | train {train_loss:.3e} | val {val['mse']:.3e} "
            f"| val SRE {val['sre']['mean']:.4f} | {time.time() - t0:.1f}s"
        )

        if val["mse"] < best_val:
            best_val, best_epoch = val["mse"], epoch
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model.config,
                "normalization": {
                    "B_max": stats.B_max, "T_max": stats.T_max, "H_max": stats.H_max,
                },
                "material": args.material,
                "epoch": epoch,
            }, ckpt_path)

        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    # ---- final test evaluation with the best checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, device)

    print("\nTest metrics (best checkpoint):")
    print(json.dumps(test, indent=2))

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **test}, f, indent=2)


if __name__ == "__main__":
    main()
