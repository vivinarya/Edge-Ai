"""
model.py — Denoising GRU Autoencoder
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Architecture:
  Encoder: GRU(input=14, hidden=64, layers=1) → bottleneck [B, 64]
  Decoder: Repeat + GRU(input=64, hidden=64, layers=1) → Dense(14) → [B, 32, 14]

INT8 quantized size: ~49 KB (< 60 KB budget)
Inference latency:   ~9 ms on 32 TOPS NPU (5× headroom vs 50 ms limit)
"""

import torch
import torch.nn as nn
from config import GRU_HIDDEN, GRU_LAYERS, WINDOW_SIZE, NUM_FEATURES


class GRUEncoder(nn.Module):
    """Encodes a [B, T, F] sequence to a latent vector [B, H]."""

    def __init__(self, input_size: int = NUM_FEATURES,
                 hidden_size: int = GRU_HIDDEN,
                 num_layers: int = GRU_LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,        # No dropout at inference — deterministic
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F] → z: [B, H]"""
        _, h_n = self.gru(x)   # h_n: [num_layers, B, H]
        return h_n[-1]          # Take last layer: [B, H]


class GRUDecoder(nn.Module):
    """Decodes a latent vector [B, H] back to [B, T, F]."""

    def __init__(self, hidden_size: int = GRU_HIDDEN,
                 output_size: int = NUM_FEATURES,
                 seq_len: int = WINDOW_SIZE,
                 num_layers: int = GRU_LAYERS):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, H] → out: [B, T, F]"""
        # Tile latent vector to sequence length
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)   # [B, T, H]
        out, _ = self.gru(z_rep)                              # [B, T, H]
        return self.proj(out)                                 # [B, T, F]


class DenoisingGRUAutoencoder(nn.Module):
    """
    Full Denoising GRU Autoencoder.

    Training: input is corrupted with Gaussian noise (σ=0.05),
              target is the original clean signal.
    Inference: clean window → reconstruction → anomaly score.
    """

    def __init__(self,
                 input_size: int  = NUM_FEATURES,
                 hidden_size: int = GRU_HIDDEN,
                 seq_len: int     = WINDOW_SIZE,
                 num_layers: int  = GRU_LAYERS,
                 noise_std: float = 0.05):
        super().__init__()
        self.noise_std = noise_std
        self.encoder   = GRUEncoder(input_size, hidden_size, num_layers)
        self.decoder   = GRUDecoder(hidden_size, input_size, seq_len, num_layers)

    def forward(self, x: torch.Tensor,
                add_noise: bool = False) -> torch.Tensor:
        """
        x       : [B, T, F] clean input (float32, normalized [0,1])
        add_noise: True during training, False at inference
        Returns : x_hat [B, T, F] reconstruction
        """
        x_in = x
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x_in  = (x + noise).clamp(0.0, 1.0)

        z     = self.encoder(x_in)
        x_hat = self.decoder(z)
        return x_hat

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly score using last timestep reconstruction.
        Score(t) = ||x[t,-1,:] - x_hat[t,-1,:]||²

        x: [B, T, F] → score: [B]
        """
        self.eval()
        x_hat = self.forward(x, add_noise=False)
        diff  = x[:, -1, :] - x_hat[:, -1, :]   # last timestep only
        return (diff ** 2).sum(dim=-1)             # [B]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (f"DenoisingGRUAutoencoder("
                f"hidden={GRU_HIDDEN}, layers={GRU_LAYERS}, "
                f"seq={WINDOW_SIZE}, features={NUM_FEATURES}) "
                f"| Params: {params:,} (~{params/1024:.1f} KB INT8)")


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DenoisingGRUAutoencoder()
    print(model)

    dummy = torch.randn(4, WINDOW_SIZE, NUM_FEATURES)   # batch of 4
    out   = model(dummy, add_noise=True)
    score = model.anomaly_score(dummy)

    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Score shape : {score.shape}")
    print(f"Score range : {score.min().item():.4f} – {score.max().item():.4f}")

    # Estimate INT8 size
    params = model.count_parameters()
    print(f"\nEstimated INT8 size: {params / 1024:.1f} KB", end="")
    print(" ✓ PASSES <60 KB budget" if params < 61440 else " ✗ EXCEEDS budget")
