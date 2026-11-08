"""Magnetic State-Space GRU (MSS-GRU), RHINO-MAG-inspired (not an exact
reproduction of the published model)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import compute_dB


class MSSGRU(nn.Module):
    """
    (B_t, dB_t, T) -> H_t

        e_t = Encoder(B_t, dB_t, T)            feature encoder
        s_t = F(s_{t-1}, e_t)                  magnetic state transition
              (update gate z_t, reset gate r_t, candidate state s~_t,
               s_t = (1 - z_t) * s_{t-1} + z_t * s~_t)
        m_t = StateRepr(s_t)                   magnetic state representation
        H_t = Decoder(m_t, B_t, T)             H prediction decoder

    The state is learned from the B waveform only (no H history, no previous
    H predictions). F is implemented with nn.GRU (fast cuDNN kernel).

    Args:
        hidden_size:   dimension of the magnetic state s_t
        enc_dim:       width of the feature encoder
        dec_dim:       width of the H decoder
        num_layers:    stacked state-transition layers
        dropout:       dropout between state layers (only if num_layers > 1)
        bidirectional: state also sees FUTURE B samples (offline only)
        db_scale:      multiplier that brings dB_t to a range similar to B_t
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

        self.encoder = nn.Sequential(
            nn.Linear(3, enc_dim), nn.GELU(),
            nn.Linear(enc_dim, enc_dim), nn.GELU(),
        )
        self.state_rnn = nn.GRU(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        state_out = hidden_size * (2 if bidirectional else 1)
        self.state_repr = nn.Sequential(nn.Linear(state_out, hidden_size), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 2, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, 1),
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            hidden_size=args.hidden_size,
            enc_dim=args.enc_dim,
            dec_dim=args.dec_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            db_scale=args.db_scale,
        )

    def forward(self, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        batch, length = B.shape
        dB = compute_dB(B, self.db_scale)
        T_ = T.view(batch, 1).expand(batch, length).unsqueeze(-1)
        B_ = B.unsqueeze(-1)

        e = self.encoder(torch.cat([B_, dB.unsqueeze(-1), T_], dim=-1))  # [b, L, enc]
        s, _ = self.state_rnn(e)                                         # [b, L, hid*dirs]
        m = self.state_repr(s)                                           # [b, L, hid]
        h = self.decoder(torch.cat([m, B_, T_], dim=-1))                 # [b, L, 1]
        return h.squeeze(-1)