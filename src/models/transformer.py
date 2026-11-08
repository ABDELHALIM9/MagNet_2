"""Transformer baseline: (B sequence, T) -> H sequence. No H history is used."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import compute_dB


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.max_len = max_len
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch, L, d_model]
        if x.size(1) > self.max_len:
            raise ValueError(f"Sequence length {x.size(1)} > max_len {self.max_len}")
        return x + self.pe[: x.size(1)].unsqueeze(0)


class BTHTransformer(nn.Module):
    """
    Per time step input:  [B(t), dB(t), T]   (dB optional via use_db)
    Output:               H(t)

    Pre-norm Transformer encoder with sinusoidal positions. With causal=True
    each step only attends to the past and present B samples (comparable to
    a unidirectional GRU); causal=False sees the whole window (comparable to
    a bidirectional GRU, offline use only).

    Args:
        d_model:    embedding width (must be even and divisible by n_heads)
        n_heads:    attention heads
        num_layers: encoder layers
        ff_dim:     feed-forward width
        dropout:    dropout inside the encoder layers
        causal:     use a causal attention mask
        use_db:     add dB_t as an input feature
        db_scale:   multiplier that brings dB_t to a range similar to B_t
        max_len:    longest sequence supported by the positional encoding
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.0,
        causal: bool = True,
        use_db: bool = True,
        db_scale: float = 100.0,
        max_len: int = 4096,
    ):
        super().__init__()
        if d_model % 2 != 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be even and divisible by n_heads")

        self.config = dict(
            model_type="transformer",
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            causal=causal,
            use_db=use_db,
            db_scale=db_scale,
            max_len=max_len,
        )
        self.causal = causal
        self.use_db = use_db
        self.db_scale = db_scale

        in_features = 3 if use_db else 2
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    @classmethod
    def from_args(cls, args):
        return cls(
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            causal=not args.non_causal,
            db_scale=args.db_scale,
        )

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        batch, length = B.shape
        T_ = T.view(batch, 1).expand(batch, length).unsqueeze(-1)
        feats = [B.unsqueeze(-1)]
        if self.use_db:
            feats.append(compute_dB(B, self.db_scale).unsqueeze(-1))
        feats.append(T_)

        x = self.input_proj(torch.cat(feats, dim=-1))   # [batch, L, d_model]
        x = self.pos_enc(x)

        mask = None
        if self.causal:  # True = position may NOT be attended to
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=B.device), diagonal=1
            )
        x = self.encoder(x, mask=mask)
        return self.head(x).squeeze(-1)                 # [batch, L]