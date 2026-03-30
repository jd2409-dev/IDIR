"""Mixture-of-Experts Layer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Expert(nn.Module):
    """Individual expert network"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TopKRouter(nn.Module):
    """Top-k gating network for routing to experts"""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_eps: float = 1e-2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_eps = noise_eps
        self.aux_loss_coef = aux_loss_coef

        # Routing network
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Noise for load balancing during training
        self.training_noise = True

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Route inputs to experts.

        Returns:
            top_k_indices: (batch, seq, top_k)
            top_k_gates: (batch, seq, top_k)
            aux_loss: load balancing loss
        """
        # Compute routing scores
        router_logits = self.router(x)  # (batch, seq, num_experts)

        # Add noise during training for load balancing
        if self.training and self.training_noise:
            noise = torch.randn_like(router_logits) * self.noise_eps
            router_logits = router_logits + noise

        # Softmax over experts
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize gates
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)

        return top_k_indices, top_k_gates, aux_loss

    def _compute_aux_loss(
        self, router_probs: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        Encourages uniform distribution across experts.
        """
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(
            top_k_indices, num_classes=self.num_experts
        ).float()  # (batch, seq, top_k, num_experts)

        # Average across tokens and top_k
        avg_tokens_per_expert = expert_mask.mean(dim=[0, 1, 2])

        # Average probability of selecting each expert
        avg_router_prob = router_probs.mean(dim=[0, 1])

        # Load balancing loss: encourage uniform distribution
        aux_loss = self.num_experts * (avg_tokens_per_expert * avg_router_prob).sum()

        return self.aux_loss_coef * aux_loss

    def set_training_noise(self, enabled: bool):
        """Enable/disable training noise"""
        self.training_noise = enabled


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture-of-Experts layer.
    Experts(h) = Σ_i g_i E_i(h), top-k routing

    Experts specialize in different domains: code, math, logic, language.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        expert_specialization: Optional[list] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * dim

        # Create experts
        self.experts = nn.ModuleList(
            [Expert(dim, expert_hidden_dim, dropout) for _ in range(num_experts)]
        )

        # Router
        self.router = TopKRouter(dim, num_experts, top_k)

        # Expert specialization labels (for monitoring/analysis)
        if expert_specialization is None:
            # Default: code, math, logic, language, general x4
            expert_specialization = ["code", "math", "logic", "language"] + [
                "general"
            ] * (num_experts - 4)
        self.expert_specialization = expert_specialization

        # Statistics tracking
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through MoE.

        Args:
            x: Input tensor (batch, seq, dim)

        Returns:
            output: Mixed expert outputs (batch, seq, dim)
            aux_loss: Load balancing auxiliary loss
        """
        batch_size, seq_len, dim = x.shape

        # Route to experts
        top_k_indices, top_k_gates, aux_loss = self.router(x)

        # Initialize output
        output = torch.zeros_like(x)

        # Process through selected experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # (batch, seq)
            gate = top_k_gates[..., i : i + 1]  # (batch, seq, 1)

            # For each expert, process tokens routed to it
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id  # (batch, seq)
                if mask.any():
                    expert_input = x[mask]  # (num_tokens, dim)
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += gate[mask] * expert_output

        # Update statistics
        if self.training:
            with torch.no_grad():
                for i in range(self.top_k):
                    for expert_id in range(self.num_experts):
                        count = (top_k_indices[..., i] == expert_id).sum()
                        self.expert_usage[expert_id] += count
                self.total_tokens += batch_size * seq_len * self.top_k

        return output, aux_loss

    def get_expert_stats(self) -> dict:
        """Get statistics about expert usage"""
        if self.total_tokens == 0:
            return {"expert_usage": [0.0] * self.num_experts}

        usage_ratio = self.expert_usage / self.total_tokens

        return {
            "expert_usage": usage_ratio.tolist(),
            "expert_specialization": self.expert_specialization,
            "usage_entropy": -(usage_ratio * torch.log(usage_ratio + 1e-10))
            .sum()
            .item(),
        }

    def reset_stats(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_()
        self.total_tokens.zero_()


class DenseMLP(nn.Module):
    """Dense MLP for ablation (replaces MoE)"""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, torch.tensor(0.0, device=x.device)  # Return dummy aux_loss
