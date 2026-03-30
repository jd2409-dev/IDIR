"""Knowledge-Dense Factorized Linear Layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FactorizedLinear(nn.Module):
    """
    Factorized linear transformation: W ≈ A × B
    Reduces redundancy and increases parameter efficiency.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Calculate rank if not specified (knowledge-dense factorization)
        if rank is None:
            rank = max(min(in_features, out_features) // 4, 64)
        self.rank = rank

        # Factorized matrices: W ≈ A × B where A ∈ R^(out × rank), B ∈ R^(rank × in)
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.02)

        # Optional: Add residual path for better gradient flow
        self.use_residual = in_features == out_features
        if self.use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):
        """Initialize with careful scaling for stability"""
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = (A × B) × x + bias
        """
        # Efficient computation: first multiply by B, then by A
        # This avoids materializing the full matrix
        out = F.linear(x, self.B)  # x @ B^T
        out = F.linear(out, self.A)  # (x @ B^T) @ A^T = x @ (A @ B)^T

        if self.use_residual:
            out = out + self.residual_scale * x

        return out

    def get_full_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix for analysis"""
        return self.A @ self.B

    def effective_params(self) -> int:
        """Return effective parameter count"""
        return self.out_features * self.rank + self.rank * self.in_features

    def compression_ratio(self) -> float:
        """Return compression ratio vs full dense matrix"""
        full_params = self.in_features * self.out_features
        return full_params / self.effective_params()


class FactorizedMLP(nn.Module):
    """MLP with factorized transformations"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        rank: int = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.fc1 = FactorizedLinear(dim, hidden_dim, rank)
        self.fc2 = FactorizedLinear(hidden_dim, dim, rank)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual
