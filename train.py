"""
Train a B+T -> H GRU for one material.

Example:
    python train.py --material A \
        --processed-root /content/drive/MyDrive/BTH_MAG/processed \
        --out-dir /content/drive/MyDrive/BTH_MAG/models/A

Defaults can also come from a YAML file:  --config configs/default.yaml
(command-line flags override the file).

Outputs in --out-dir:
    train.log          timestamped log (appended across runs)
    args.json          arguments of the run
    metrics.csv        per-epoch metrics (appended across runs)
    history.json       per-epoch metrics (rewritten every epoch)
    best.pt            best checkpoint (used by inference.py)
    last.pt            latest checkpoint with optimizer/scheduler state
    test_metrics.json  final test metrics of the best checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
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
from src.model import build_model, count_parameters


def setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / "train.log", mode="a")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


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
    p.add_argument("--model", type=str, default="gru", choices=["gru", "mss_gru"],
                   help="gru = baseline BTHGRU, mss_gru = Magnetic State-Space GRU")
    # MSS-GRU only (ignored by --model gru)
    p.add_argument("--enc-dim", type=int, default=32)
    p.add_argument("--dec-dim", type=int, default=64)
    p.add_argument("--db-scale", type=float, default=100.0)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(out_dir)
    log.info("=" * 60)
    log.info(f"New run | material={args.material}")
    log.info(f"Python {platform.python_version()} | torch {torch.__version__}")
    log.info(f"Device: {device}"
             + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info(f"Args: {json.dumps(vars(args))}")

    train_loader, val_loader, test_loader, stats = make_loaders(
        args.processed_root,
        args.material,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    log.info(f"Windows | train={len(train_loader.dataset)} "
             f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    log.info(f"Normalization | B_max={stats.B_max:.4g} "
             f"T_max={stats.T_max:.4g} H_max={stats.H_max:.4g}")

    model_kwargs = dict(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    if args.model == "mss_gru":
        model_kwargs.update(enc_dim=args.enc_dim, dec_dim=args.dec_dim,
                            db_scale=args.db_scale)
    model = build_model(args.model, **model_kwargs).to(device)
    log.info(f"Model: {args.model} | config: {model.config}")
    log.info(f"Parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    best_val, best_epoch, history = float("inf"), -1, []
    ckpt_path = out_dir / "best.pt"
    csv_path = out_dir / "metrics.csv"

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
            "val_nrmse_mean": val["nrmse"]["mean"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        log.info(
            f"Epoch {epoch:03d} | train {train_loss:.3e} | val {val['mse']:.3e} "
            f"| val SRE {val['sre']['mean']:.4f} "
            f"| val NRMSE {val['nrmse']['mean']:.4f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e} "
            f"| {time.time() - t0:.1f}s"
        )

        # CSV (header written once)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[-1].keys())
            if write_header:
                w.writeheader()
            w.writerow(history[-1])

        # history rewritten every epoch, so a crash doesn't lose it
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # best checkpoint
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
            log.info(f"  -> new best val MSE {best_val:.3e}, saved {ckpt_path}")

        # latest checkpoint (saved after the best update so values are current)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
        }, out_dir / "last.pt")

        if args.patience and epoch - best_epoch >= args.patience:
            log.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    # ---- final test evaluation with the best checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, device)

    log.info(f"Test metrics (best checkpoint, epoch {best_epoch}):\n"
             f"{json.dumps(test, indent=2)}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **test}, f, indent=2)
    log.info("Done.")


if __name__ == "__main__":
    main()