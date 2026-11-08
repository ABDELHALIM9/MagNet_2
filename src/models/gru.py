"""Baseline GRU: (B sequence, T) -> H sequence. No H history is used."""
from __future__ import annotations

import torch
import torch.nn as nn


class BTHGRU(nn.Module):
    """
    Per time step input:  [B(t), T]
    Output:               H(t)

    Args:
        hidden_size:   GRU hidden units
        num_layers:    stacked GRU layers
        dropout:       dropout between GRU layers (only if num_layers > 1)
        bidirectional: if True the model also sees FUTURE B samples.
                       Fine for offline B->H mapping; keep False for a
                       strictly causal model.
        num_materials: if > 0, adds a learned material embedding as an extra
                       input (universal-model experiment). 0 = per-material model.
        material_emb_dim: embedding size when num_materials > 0
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        num_materials: int = 0,
        material_emb_dim: int = 4,
    ):
        super().__init__()
        self.config = dict(
            model_type="gru",
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            num_materials=num_materials,
            material_emb_dim=material_emb_dim,
        )

        in_features = 2  # B, T
        self.material_emb = None
        if num_materials > 0:
            self.material_emb = nn.Embedding(num_materials, material_emb_dim)
            in_features += material_emb_dim

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    @classmethod
    def from_args(cls, args):
        return cls(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )

    def forward(
        self,
        B: torch.Tensor,                          # [batch, L]
        T: torch.Tensor,                          # [batch]
        material_id: torch.Tensor | None = None,  # [batch] long
    ) -> torch.Tensor:                            # [batch, L]
        batch, length = B.shape
        T_seq = T.view(batch, 1).expand(batch, length)
        feats = [B.unsqueeze(-1), T_seq.unsqueeze(-1)]

        if self.material_emb is not None:
            if material_id is None:
                raise ValueError("material_id required when num_materials > 0")
            emb = self.material_emb(material_id)                  # [batch, E]
            feats.append(emb.unsqueeze(1).expand(batch, length, -1))

        x = torch.cat(feats, dim=-1)       # [batch, L, in_features]
        out, _ = self.gru(x)               # [batch, L, hidden]
        return self.head(out).squeeze(-1)  # [batch, L]