"""
Preprocessing for the BTH-MAG project.

Task:   (B_{0:T}, T) -> H_{0:T}      (no H history at inference)

Pipeline:
    raw CSV (B/H/T)  ->  sequence-level split (stratified by temperature)
                     ->  normalization (train statistics only)
                     ->  optional windowing
                     ->  .npz + normalization.json + metadata.json
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class PreprocessConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    seed: int = 42

    # Keep every temperature represented in train/val/test
    stratify_by_temperature: bool = True

    # Optional windowing (applied AFTER the split)
    use_windows: bool = True
    window_size: int = 512
    stride: int = 256

    dtype: str = "float32"

    def validate(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"train + val + test must equal 1.0, got {total}")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")


# ============================================================
# Dataset container
# ============================================================

@dataclass
class SequenceDataset:
    B: np.ndarray   # [N, L]
    T: np.ndarray   # [N]
    H: np.ndarray   # [N, L]

    def __len__(self):
        return len(self.B)

    def __getitem__(self, idx):
        return {"B": self.B[idx], "T": self.T[idx], "H": self.H[idx]}

    def subset(self, idx: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(B=self.B[idx], T=self.T[idx], H=self.H[idx])


# ============================================================
# Normalization
# ============================================================

@dataclass
class NormalizationStats:
    B_max: float
    T_max: float
    H_max: float

    def normalize(self, B: np.ndarray, T: np.ndarray, H: Optional[np.ndarray] = None):
        B_norm = B / self.B_max
        T_norm = T / self.T_max
        if H is None:
            return B_norm, T_norm
        return B_norm, T_norm, H / self.H_max

    def denormalize_H(self, H_norm: np.ndarray):
        return H_norm * self.H_max

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "B_max": float(self.B_max),
            "T_max": float(self.T_max),
            "H_max": float(self.H_max),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(B_max=data["B_max"], T_max=data["T_max"], H_max=data["H_max"])


# ============================================================
# ZIP extraction
# ============================================================

def extract_zip(zip_path: str | Path, output_dir: str | Path, overwrite: bool = False):
    """Extract the raw MagNet CSV dataset. Skips extraction if output_dir
    is already non-empty, unless overwrite=True."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()) and not overwrite:
        print(f"{output_dir} is not empty - skipping extraction (overwrite=False).")
        return

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
    print(f"Dataset extracted to: {output_dir}")


# ============================================================
# Find material files
# ============================================================

def find_materials(raw_root: str | Path) -> List[str]:
    """Find material folders (A, B, C, ...). If the CSVs sit directly in
    raw_root (no sub-folders), material prefixes are inferred from file names."""
    raw_root = Path(raw_root)

    dirs = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if dirs:
        return dirs

    prefixes = set()
    for f in raw_root.glob("*_*_B.csv"):
        prefixes.add(f.stem.split("_")[0])
    return sorted(prefixes)


def _material_dir(raw_root: Path, material: str) -> Path:
    """Folder that contains the material's CSVs: raw_root/<material> if it
    exists, otherwise raw_root itself (flat layout)."""
    sub = raw_root / material
    if sub.exists():
        return sub
    return raw_root


def find_sequence_numbers(material_dir: str | Path, material: str) -> List[int]:
    """Find indices such as A_1_B.csv, A_2_B.csv, ... -> [1, 2, ...]"""
    material_dir = Path(material_dir)
    numbers = []
    for file in material_dir.glob(f"{material}_*_B.csv"):
        parts = file.stem.split("_")
        if len(parts) != 3:
            continue
        try:
            numbers.append(int(parts[1]))
        except ValueError:
            continue
    return sorted(numbers)


# ============================================================
# Load one sequence group
# ============================================================

def load_sequence(
    material_dir: str | Path,
    material: str,
    number: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load B, H, T for one measurement group:
    <material>_<number>_B.csv / _H.csv / _T.csv"""
    material_dir = Path(material_dir)

    B_path = material_dir / f"{material}_{number}_B.csv"
    H_path = material_dir / f"{material}_{number}_H.csv"
    T_path = material_dir / f"{material}_{number}_T.csv"

    for path in (B_path, H_path, T_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    B = pd.read_csv(B_path, header=None).values.astype(np.float32)
    H = pd.read_csv(H_path, header=None).values.astype(np.float32)
    T = pd.read_csv(T_path, header=None).values.astype(np.float32)

    # Temperature: one value per sequence -> flatten to [N]
    if T.ndim == 2:
        T = T[:, 0]

    if B.shape != H.shape:
        raise ValueError(
            f"B/H shape mismatch for {material}_{number}: B={B.shape}, H={H.shape}"
        )
    if B.shape[0] != len(T):
        raise ValueError(
            f"B/T sequence mismatch for {material}_{number}: B={B.shape}, T={T.shape}"
        )

    return B, H, T


# ============================================================
# Load complete material
# ============================================================

def load_material(raw_root: str | Path, material: str) -> SequenceDataset:
    """Load and concatenate all B/H/T groups for one material."""
    raw_root = Path(raw_root)
    material_dir = _material_dir(raw_root, material)

    numbers = find_sequence_numbers(material_dir, material)
    if not numbers:
        raise RuntimeError(
            f"No sequences found for material {material} in {material_dir}"
        )

    all_B, all_H, all_T = [], [], []
    print(f"Loading material {material} ({len(numbers)} measurement groups)")

    for number in numbers:
        B, H, T = load_sequence(material_dir, material, number)
        all_B.append(B)
        all_H.append(H)
        all_T.append(T)
        print(f"  {material}_{number}: B={B.shape}, H={H.shape}, T={T.shape}")

    # Groups may have different sequence lengths; fail loudly if so.
    lengths = {b.shape[1] for b in all_B}
    if len(lengths) > 1:
        raise ValueError(
            f"Groups of material {material} have different sequence lengths: "
            f"{sorted(lengths)}. Resample or process groups separately."
        )

    return SequenceDataset(
        B=np.concatenate(all_B, axis=0),
        T=np.concatenate(all_T, axis=0),
        H=np.concatenate(all_H, axis=0),
    )


# ============================================================
# Train / validation / test split
# ============================================================

def _split_group(idx: np.ndarray, config: PreprocessConfig, rng):
    idx = idx.copy()
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * config.train_ratio))
    n_val = int(round(n * config.val_ratio))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def split_dataset(dataset: SequenceDataset, config: PreprocessConfig):
    """Split COMPLETE sequences (before windowing) so windows of the same
    sequence never end up in different splits. Optionally stratified by
    temperature so every temperature appears in train/val/test."""
    rng = np.random.default_rng(config.seed)
    n = len(dataset)

    temps = np.round(dataset.T, 3)
    unique_temps = np.unique(temps)

    # Stratification only makes sense with a small set of discrete temperatures.
    stratify = config.stratify_by_temperature and 1 < len(unique_temps) <= 20

    if stratify:
        tr, va, te = [], [], []
        for t in unique_temps:
            a, b, c = _split_group(np.where(temps == t)[0], config, rng)
            tr.append(a); va.append(b); te.append(c)
        train_idx = np.concatenate(tr)
        val_idx = np.concatenate(va)
        test_idx = np.concatenate(te)
        print(f"Stratified split over {len(unique_temps)} temperatures: "
              f"{unique_temps.tolist()}")
    else:
        if config.stratify_by_temperature:
            print("Temperature stratification skipped "
                  f"({len(unique_temps)} unique temperatures).")
        train_idx, val_idx, test_idx = _split_group(np.arange(n), config, rng)

    return (
        dataset.subset(train_idx),
        dataset.subset(val_idx),
        dataset.subset(test_idx),
    )


# ============================================================
# Normalization
# ============================================================

def compute_normalization(train: SequenceDataset) -> NormalizationStats:
    """Max-abs statistics from the TRAINING split only (no leakage)."""
    B_max = max(float(np.max(np.abs(train.B))), 1e-12)
    T_max = max(float(np.max(np.abs(train.T))), 1e-12)
    H_max = max(float(np.max(np.abs(train.H))), 1e-12)
    return NormalizationStats(B_max=B_max, T_max=T_max, H_max=H_max)


def normalize_dataset(dataset: SequenceDataset, stats: NormalizationStats) -> SequenceDataset:
    B, T, H = stats.normalize(dataset.B, dataset.T, dataset.H)
    return SequenceDataset(
        B=B.astype(np.float32), T=T.astype(np.float32), H=H.astype(np.float32)
    )


# ============================================================
# Windowing
# ============================================================

def create_windows(dataset: SequenceDataset, window_size: int, stride: int) -> SequenceDataset:
    """[N, L] sequences -> [num_windows, window_size] windows."""
    B_windows, H_windows, T_windows = [], [], []

    for i in range(len(dataset)):
        B_seq, H_seq, T_val = dataset.B[i], dataset.H[i], dataset.T[i]
        seq_len = len(B_seq)

        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            B_windows.append(B_seq[start:end])
            H_windows.append(H_seq[start:end])
            T_windows.append(T_val)

    if not B_windows:
        raise RuntimeError(
            "No windows were generated. Check window_size against sequence length."
        )

    return SequenceDataset(
        B=np.asarray(B_windows, dtype=np.float32),
        T=np.asarray(T_windows, dtype=np.float32),
        H=np.asarray(H_windows, dtype=np.float32),
    )


# ============================================================
# Save / load processed dataset
# ============================================================

def save_dataset(dataset: SequenceDataset, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, B=dataset.B, T=dataset.T, H=dataset.H)
    print(f"Saved dataset: {path}")
    print(f"  B: {dataset.B.shape}")
    print(f"  T: {dataset.T.shape}")
    print(f"  H: {dataset.H.shape}")


def load_dataset(path: str | Path) -> SequenceDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return SequenceDataset(B=data["B"], T=data["T"], H=data["H"])


# ============================================================
# Complete pipeline
# ============================================================

def preprocess_material(
    raw_root: str | Path,
    processed_root: str | Path,
    material: str,
    config: Optional[PreprocessConfig] = None,
):
    """raw CSV -> split -> normalize -> (window) -> save."""
    if config is None:
        config = PreprocessConfig()
    config.validate()

    print("=" * 70)
    print(f"Processing material: {material}")
    print("=" * 70)

    # 1. Load
    dataset = load_material(raw_root=raw_root, material=material)
    print("\nFull dataset:")
    print("B:", dataset.B.shape)
    print("H:", dataset.H.shape)
    print("T:", dataset.T.shape)

    # 2. Split sequences
    train, val, test = split_dataset(dataset, config)
    print("\nSequence split:")
    print("Train:", len(train))
    print("Val:  ", len(val))
    print("Test: ", len(test))

    # 3. Normalization from TRAIN only
    stats = compute_normalization(train)
    print("\nNormalization:")
    print("B_max:", stats.B_max)
    print("T_max:", stats.T_max)
    print("H_max:", stats.H_max)

    # 4. Normalize
    train = normalize_dataset(train, stats)
    val = normalize_dataset(val, stats)
    test = normalize_dataset(test, stats)

    # 5. Window
    if config.use_windows:
        train = create_windows(train, config.window_size, config.stride)
        val = create_windows(val, config.window_size, config.stride)
        test = create_windows(test, config.window_size, config.stride)

    # 6. Save
    material_root = Path(processed_root) / material
    material_root.mkdir(parents=True, exist_ok=True)

    save_dataset(train, material_root / "train.npz")
    save_dataset(val, material_root / "val.npz")
    save_dataset(test, material_root / "test.npz")
    stats.save(material_root / "normalization.json")

    # 7. Metadata
    metadata = {
        "material": material,
        "input_features": ["B", "T"],
        "target": "H",
        "use_windows": config.use_windows,
        "window_size": config.window_size,
        "stride": config.stride,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "stratify_by_temperature": config.stratify_by_temperature,
        "seed": config.seed,
        "B_max": stats.B_max,
        "T_max": stats.T_max,
        "H_max": stats.H_max,
    }
    with open(material_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nFinished material {material}")
    return train, val, test, stats
