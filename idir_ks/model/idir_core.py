"""IDIR Core - Iterative Fixed-Point Solver"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
import math


class PhiLayer(nn.Module):
    """
    Phi(h, x): Knowledge-dense transformation combining latent and input.
    This is the main transformation layer in the fixed-point operator.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Input projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Gate for input integration
        self.input_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent representation (batch, seq, dim)
            x: Input embedding (batch, seq, dim)

        Returns:
            Transformed output (batch, seq, dim)
        """
        batch_size, seq_len, dim = h.shape

        # Combine h and x with gating
        gate_input = torch.cat([h, x], dim=-1)
        gate = self.input_gate(gate_input)

        # Blend latent and input
        blended = gate * h + (1 - gate) * x

        # Multi-head self-attention on blended representation
        q = self.q_proj(blended).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(blended).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(blended).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class GatingMechanism(nn.Module):
    """
    Adaptive gating for combining different components.
    G1, G2, G3 in the paper.
    """

    def __init__(self, dim: int, num_gates: int = 3):
        super().__init__()
        self.dim = dim
        self.num_gates = num_gates

        # Separate gate for each component
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim // 4),
                    nn.GELU(),
                    nn.Linear(dim // 4, dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_gates)
            ]
        )

        # Gate importance weights (learnable)
        self.gate_weights = nn.Parameter(torch.ones(num_gates))

    def forward(self, h: torch.Tensor) -> list:
        """
        Compute gates for each component.

        Args:
            h: Input tensor (batch, seq, dim)

        Returns:
            List of gate tensors (batch, seq, dim)
        """
        gates = []
        weights = F.softmax(self.gate_weights, dim=0)

        for i, gate_fn in enumerate(self.gates):
            gate = gate_fn(h) * weights[i]
            gates.append(gate)

        return gates


class FixedPointSolver(nn.Module):
    """
    Iterative fixed-point solver:
    h_{k+1} = F_theta(h_k, x, M)

    Supports adaptive compute based on convergence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        convergence_threshold: float = 1e-4,
        max_steps: int = 12,
        min_steps: int = 4,
        enable_adaptive: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.enable_adaptive = enable_adaptive

        # Core transformation Phi(h, x)
        self.phi = PhiLayer(dim, num_heads, dropout)

        # Gating mechanisms
        self.gating = GatingMechanism(dim, num_gates=3)

        # Normalization
        self.norm = nn.LayerNorm(dim)

        # Adaptive compute predictor (learns to predict needed steps)
        if enable_adaptive:
            self.step_predictor = nn.Sequential(
                nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1)
            )
            self.base_steps = min_steps
            self.extra_steps = max_steps - min_steps

    def compute_adaptive_steps(self, h: torch.Tensor) -> int:
        """
        Predict number of steps needed: steps = base + sigmoid(W h) × extra

        Args:
            h: Current latent representation (batch, seq, dim)

        Returns:
            Number of steps to run
        """
        if not self.enable_adaptive:
            return self.max_steps

        # Average pooling over sequence
        pooled = h.mean(dim=1)  # (batch, dim)

        # Predict extra steps
        extra = torch.sigmoid(self.step_predictor(pooled))  # (batch, 1)
        extra = extra.mean().item()  # Average across batch

        steps = self.base_steps + int(extra * self.extra_steps)
        return min(steps, self.max_steps)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        memory_fn: Optional[Callable] = None,
        experts_fn: Optional[Callable] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Iterate to find fixed point.

        Args:
            h: Initial latent state (batch, seq, dim)
            x: Input embedding (batch, seq, dim)
            memory_fn: Function to retrieve from memory
            experts_fn: Function to call experts
            return_trajectory: Whether to return intermediate states

        Returns:
            h_star: Fixed point representation
            info: Dictionary with solver information
        """
        trajectory = [h.clone()] if return_trajectory else None

        # Determine number of steps
        num_steps = self.compute_adaptive_steps(h)

        info = {"num_steps": num_steps, "converged": False, "residuals": []}

        for step in range(num_steps):
            h_prev = h.clone()

            # Compute gates
            gates = self.gating(h)
            G1, G2, G3 = gates

            # Compute components
            phi_out = self.phi(h, x)  # Phi(h, x)

            memory_out = torch.zeros_like(h)
            if memory_fn is not None:
                memory_out = memory_fn(h)  # Memory(h, M)

            experts_out = torch.zeros_like(h)
            aux_loss = torch.tensor(0.0, device=h.device)
            if experts_fn is not None:
                experts_out, aux_loss = experts_fn(h)  # Experts(h)

            # Combine: h + G1*Phi + G2*Memory + G3*Experts
            delta = G1 * phi_out + G2 * memory_out + G3 * experts_out
            h = h + delta

            # Normalize
            h = self.norm(h)

            # Check convergence
            residual = torch.norm(h - h_prev) / (torch.norm(h_prev) + 1e-8)
            info["residuals"].append(residual.item())

            if trajectory is not None:
                trajectory.append(h.clone())

            # Early stopping if converged (after min_steps)
            if step >= self.min_steps and residual < self.convergence_threshold:
                info["converged"] = True
                info["num_steps"] = step + 1
                break

        if trajectory is not None:
            info["trajectory"] = trajectory

        info["aux_loss"] = aux_loss

        return h, info


class IDIRCore(nn.Module):
    """
    Implicit Dense Iterative Reasoning Core.
    Main building block that encapsulates the fixed-point solver.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        convergence_threshold: float = 1e-4,
        max_steps: int = 12,
        min_steps: int = 4,
        enable_adaptive: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # Fixed-point solver
        self.solver = FixedPointSolver(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            convergence_threshold=convergence_threshold,
            max_steps=max_steps,
            min_steps=min_steps,
            enable_adaptive=enable_adaptive,
        )

        # Skip connection for single-pass variant (ablation)
        self.single_pass_mode = False

    def set_single_pass(self, enabled: bool):
        """Enable single-pass mode (ablation)"""
        self.single_pass_mode = enabled

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        memory_fn: Optional[Callable] = None,
        experts_fn: Optional[Callable] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through IDIR core.

        Args:
            h: Latent state
            x: Input
            memory_fn: Memory retrieval function
            experts_fn: Expert function
            return_info: Whether to return solver information

        Returns:
            h_star or (h_star, info)
        """
        if self.single_pass_mode:
            # Ablation: single iteration only
            gates = self.solver.gating(h)
            G1, G2, G3 = gates

            phi_out = self.solver.phi(h, x)
            memory_out = memory_fn(h) if memory_fn else torch.zeros_like(h)
            experts_out, aux_loss = (
                experts_fn(h) if experts_fn else (torch.zeros_like(h), 0.0)
            )

            h_star = h + G1 * phi_out + G2 * memory_out + G3 * experts_out
            h_star = self.solver.norm(h_star)

            if return_info:
                return h_star, {"num_steps": 1, "converged": False}
            return h_star

        # Full iterative solving
        h_star, info = self.solver(h, x, memory_fn, experts_fn)

        if return_info:
            return h_star, info
        return h_star
