"""Sequence models: (B sequence, T) -> H sequence. No H history is used.

    BTHGRU  - baseline: plain GRU + linear head
    MSSGRU  - Magnetic State-Space GRU (RHINO-MAG-inspired, NOT an exact
              reproduction of the published model)
    build_model(...) / build_model_from_config(...) - factories used by
              train.py and inference.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Baseline
# ============================================================

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

    def forward(
        self,
        B: torch.Tensor,                       # [batch, L]
        T: torch.Tensor,                       # [batch]
        material_id: torch.Tensor | None = None,  # [batch] long
    ) -> torch.Tensor:                         # [batch, L]
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


# ============================================================
# Magnetic State-Space GRU
# ============================================================

class MSSGRU(nn.Module):
    """
    Magnetic State-Space GRU:  (B_t, dB_t, T) -> H_t

        e_t = Encoder(B_t, dB_t, T)                 feature encoder
        s_t = F(s_{t-1}, e_t)                       magnetic state transition
              (gated: update gate z_t, reset gate r_t, candidate state s~_t
               s_t = (1 - z_t) * s_{t-1} + z_t * s~_t)
        m_t = StateRepr(s_t)                        magnetic state representation
        H_t = Decoder(m_t, B_t, T)                  H prediction decoder

    The state is learned from the B waveform only (no H history, no previous
    H predictions). The gated transition F is implemented with nn.GRU so it
    runs on the fast cuDNN kernel.

    Args:
        hidden_size:   dimension of the magnetic state s_t
        enc_dim:       width of the feature encoder
        dec_dim:       width of the H decoder
        num_layers:    stacked state-transition layers
        dropout:       dropout between state layers (only if num_layers > 1)
        bidirectional: if True the state also sees FUTURE B samples
                       (offline mapping only; keep False for a causal model)
        db_scale:      dB_t is tiny compared with B_t (about 1/1000 of it per
                       step), so it is multiplied by this constant to bring
                       it to a similar range. Tune if training is slow.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        enc_dim: int = 32,
        dec_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        db_scale: float = 100.0,
    ):
        super().__init__()
        self.config = dict(
            model_type="mss_gru",
            hidden_size=hidden_size,
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            db_scale=db_scale,
        )
        self.db_scale = db_scale

        # 1. feature encoder: [B, dB, T] -> e_t
        self.encoder = nn.Sequential(
            nn.Linear(3, enc_dim), nn.GELU(),
            nn.Linear(enc_dim, enc_dim), nn.GELU(),
        )

        # 2. magnetic state transition: s_t = F(s_{t-1}, e_t)
        self.state_rnn = nn.GRU(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 3. magnetic state representation
        state_out = hidden_size * (2 if bidirectional else 1)
        self.state_repr = nn.Sequential(
            nn.Linear(state_out, hidden_size), nn.GELU(),
        )

        # 4. H decoder: G(m_t, B_t, T)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 2, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, 1),
        )

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # B: [batch, L], T: [batch] -> H: [batch, L]
        batch, length = B.shape

        # dB_t = B_t - B_{t-1}; first sample has no predecessor -> 0
        dB = torch.cat([torch.zeros_like(B[:, :1]), B[:, 1:] - B[:, :-1]], dim=1)
        dB = dB * self.db_scale

        T_seq = T.view(batch, 1).expand(batch, length)
        B_ = B.unsqueeze(-1)
        T_ = T_seq.unsqueeze(-1)

        e = self.encoder(torch.cat([B_, dB.unsqueeze(-1), T_], dim=-1))  # [b, L, enc]
        s, _ = self.state_rnn(e)                                         # [b, L, hid*dirs]
        m = self.state_repr(s)                                           # [b, L, hid]
        h = self.decoder(torch.cat([m, B_, T_], dim=-1))                 # [b, L, 1]
        return h.squeeze(-1)


# ============================================================
# Factories
# ============================================================

MODEL_TYPES = {"gru": BTHGRU, "mss_gru": MSSGRU}


def build_model(model_type: str = "gru", **kwargs) -> nn.Module:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {list(MODEL_TYPES)}")
    return MODEL_TYPES[model_type](**kwargs)


def build_model_from_config(config: dict) -> nn.Module:
    """Rebuild a model from checkpoint['model_config'].
    Old checkpoints (BTHGRU, no 'model_type' key) still load."""
    cfg = dict(config)
    model_type = cfg.pop("model_type", "gru")
    return build_model(model_type, **cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)