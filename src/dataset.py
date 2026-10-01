"""PyTorch wrappers around the processed .npz datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import NormalizationStats, SequenceDataset, load_dataset


class BTHDataset(Dataset):
    """Each item: B [L], T scalar, H [L] (all normalized float32)."""

    def __init__(self, dataset: SequenceDataset):
        self.B = torch.from_numpy(dataset.B).float()
        self.T = torch.from_numpy(dataset.T).float()
        self.H = torch.from_numpy(dataset.H).float()

    def __len__(self):
        return len(self.B)

    def __getitem__(self, idx):
        return {"B": self.B[idx], "T": self.T[idx], "H": self.H[idx]}


def make_loaders(
    processed_root: str | Path,
    material: str,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, NormalizationStats]:
    root = Path(processed_root) / material

    train = BTHDataset(load_dataset(root / "train.npz"))
    val = BTHDataset(load_dataset(root / "val.npz"))
    test = BTHDataset(load_dataset(root / "test.npz"))
    stats = NormalizationStats.load(root / "normalization.json")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train, shuffle=True, **kw),
        DataLoader(val, shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
        stats,
    )
