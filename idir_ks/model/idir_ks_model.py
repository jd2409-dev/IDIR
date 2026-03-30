"""IDIR-KS: Complete Model Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .idir_core import IDIRCore
from .memory_module import MemoryModule
from .moe_layer import MixtureOfExperts, DenseMLP
from .factorized_linear import FactorizedLinear


class TokenEmbedding(nn.Module):
    """Token and positional embeddings"""

    def __init__(
        self, vocab_size: int, dim: int, max_seq_len: int = 8192, dropout: float = 0.0
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_emb = self.token_embed(input_ids)

        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)

        # Combine
        x = token_emb + pos_emb
        x = self.dropout(x)

        return x


class IDIRKSModel(nn.Module):
    """
    IDIR-KS: Knowledge-Dense Implicit Reasoning with Inference-Time Scaling

    Architecture combining:
    - Implicit fixed-point reasoning
    - Knowledge-dense parameterization
    - Internal memory retrieval
    - Sparse mixture-of-experts
    - Inference-time compute scaling
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_experts: int = 8,
        expert_top_k: int = 2,
        num_memories: int = 4096,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        # IDIR specific
        convergence_threshold: float = 1e-4,
        max_solver_steps: int = 12,
        min_solver_steps: int = 4,
        enable_adaptive: bool = True,
        # Inference scaling
        num_trajectories: int = 1,
        trajectory_noise: float = 0.0,
        # Ablation flags
        use_implicit_solver: bool = True,
        use_memory: bool = True,
        use_moe: bool = True,
        use_factorization: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_trajectories = num_trajectories
        self.trajectory_noise = trajectory_noise

        # Embeddings
        self.embedding = TokenEmbedding(vocab_size, dim, max_seq_len, dropout)

        # IDIR layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = self._create_layer(
                dim=dim,
                num_heads=num_heads,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                num_memories=num_memories,
                dropout=dropout,
                convergence_threshold=convergence_threshold,
                max_solver_steps=max_solver_steps,
                min_solver_steps=min_solver_steps,
                enable_adaptive=enable_adaptive,
                use_implicit_solver=use_implicit_solver,
                use_memory=use_memory,
                use_moe=use_moe,
                use_factorization=use_factorization,
            )
            self.layers.append(layer)

        # Output layers
        self.final_norm = nn.LayerNorm(dim)

        # Factorized output projection
        if use_factorization:
            self.output_proj = FactorizedLinear(dim, vocab_size, rank=dim // 2)
        else:
            self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings with output (optional, saves parameters)
        self.tie_weights = True
        if self.tie_weights:
            self.output_proj.weight = self.embedding.token_embed.weight

        self._init_weights()

    def _create_layer(self, **kwargs) -> nn.ModuleDict:
        """Create a single IDIR-KS layer with all components"""
        dim = kwargs["dim"]

        layer = nn.ModuleDict()

        # IDIR Core (implicit solver)
        layer["idir"] = IDIRCore(
            dim=dim,
            num_heads=kwargs["num_heads"],
            dropout=kwargs["dropout"],
            convergence_threshold=kwargs["convergence_threshold"],
            max_steps=kwargs["max_solver_steps"],
            min_steps=kwargs["min_solver_steps"],
            enable_adaptive=kwargs["enable_adaptive"],
        )

        # Memory module
        if kwargs["use_memory"]:
            layer["memory"] = MemoryModule(dim, kwargs["num_memories"])
        else:
            layer["memory"] = None

        # MoE or Dense MLP
        if kwargs["use_moe"]:
            layer["experts"] = MixtureOfExperts(
                dim=dim,
                num_experts=kwargs["num_experts"],
                top_k=kwargs["expert_top_k"],
                dropout=kwargs["dropout"],
            )
        else:
            layer["experts"] = DenseMLP(dim, dropout=kwargs["dropout"])

        # Factorized transformation for input (optional)
        if kwargs["use_factorization"]:
            layer["input_transform"] = FactorizedLinear(dim, dim, rank=dim // 4)
        else:
            layer["input_transform"] = nn.Identity()

        # Store ablation flags
        layer.use_implicit = kwargs["use_implicit_solver"]
        layer.use_memory = kwargs["use_memory"]
        layer.use_moe = kwargs["use_moe"]

        return layer

    def _init_weights(self):
        """Initialize remaining weights"""
        if not self.tie_weights:
            nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward_layer(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        layer: nn.ModuleDict,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Forward through a single IDIR-KS layer"""

        # Get memory function if enabled
        memory_fn = None
        if layer.use_memory and layer["memory"] is not None:
            memory_fn = layer["memory"]

        # Get experts function
        experts_fn = layer["experts"]

        # Run IDIR core
        h_out, info = layer["idir"](h, x, memory_fn, experts_fn, return_info=True)

        if return_info:
            return h_out, info
        return h_out, None

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            return_hidden: Whether to return hidden states
            return_info: Whether to return solver information

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            Optionally: hidden states and/or solver info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get input embeddings
        x = self.embedding(input_ids)

        # Initialize latent state
        h = x.clone()

        # Storage for info
        layer_infos = []

        # Pass through layers
        for layer in self.layers:
            h, info = self.forward_layer(h, x, layer, return_info=return_info)
            if return_info:
                layer_infos.append(info)

        # Final normalization
        h = self.final_norm(h)

        # Output projection
        logits = self.output_proj(h)

        if return_hidden and return_info:
            return logits, h, layer_infos
        elif return_hidden:
            return logits, h
        elif return_info:
            return logits, layer_infos
        return logits

    def forward_multi_trajectory(
        self,
        input_ids: torch.Tensor,
        num_trajectories: int = None,
        noise_scale: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-trajectory reasoning with self-consistency.

        Args:
            input_ids: Input token IDs
            num_trajectories: Number of trajectories (default: self.num_trajectories)
            noise_scale: Scale of trajectory noise (default: self.trajectory_noise)

        Returns:
            logits: Self-consistent logits (batch, seq, vocab_size)
            trajectory_logits: All trajectory logits (num_traj, batch, seq, vocab_size)
        """
        if num_trajectories is None:
            num_trajectories = self.num_trajectories
        if noise_scale is None:
            noise_scale = self.trajectory_noise

        # Generate multiple trajectories with different noise
        trajectory_logits = []

        for i in range(num_trajectories):
            # Add trajectory-specific noise to input
            if noise_scale > 0:
                x = self.embedding(input_ids)
                noise = torch.randn_like(x) * noise_scale
                # Temporarily replace embedding output
                with torch.no_grad():
                    h = x + noise
            else:
                h = self.embedding(input_ids)

            # Forward through layers with this initialization
            batch_size, seq_len = input_ids.shape
            x_base = self.embedding(input_ids)

            for layer in self.layers:
                h, _ = self.forward_layer(h, x_base, layer)

            h = self.final_norm(h)
            logits = self.output_proj(h)
            trajectory_logits.append(logits)

        # Stack trajectories
        trajectory_logits = torch.stack(
            trajectory_logits, dim=0
        )  # (num_traj, batch, seq, vocab)

        # Self-consistency: aggregate probabilities
        probs = F.softmax(trajectory_logits, dim=-1)
        avg_probs = probs.mean(dim=0)  # (batch, seq, vocab)

        # Convert back to logits
        consistent_logits = torch.log(avg_probs + 1e-10)

        return consistent_logits, trajectory_logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        use_multi_trajectory: bool = False,
        num_trajectories: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_multi_trajectory: Whether to use multi-trajectory reasoning
            num_trajectories: Number of trajectories if using multi-trajectory

        Returns:
            Generated token IDs
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                if use_multi_trajectory:
                    logits, _ = self.forward_multi_trajectory(
                        input_ids, num_trajectories or self.num_trajectories
                    )
                else:
                    logits = self.forward(input_ids)

                # Get logits for last token
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_model_stats(self) -> dict:
        """Get model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        component_params = {}
        for name, module in self.named_modules():
            if any(skip in name for skip in ["idir", "memory", "experts", "embedding"]):
                if not any(child in name for child in ["idir.", "memory.", "experts."]):
                    component_params[name] = sum(p.numel() for p in module.parameters())

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_layers": self.num_layers,
            "dim": self.dim,
            "vocab_size": self.vocab_size,
        }


def create_idir_ks_base(**kwargs) -> IDIRKSModel:
    """Factory function for base IDIR-KS model"""
    config = {
        "vocab_size": 50000,
        "dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "num_experts": 8,
        "expert_top_k": 2,
        "num_memories": 4096,
        "max_seq_len": 4096,
        "dropout": 0.1,
        "convergence_threshold": 1e-4,
        "max_solver_steps": 12,
        "min_solver_steps": 4,
        "enable_adaptive": True,
        "num_trajectories": 1,
        "trajectory_noise": 0.0,
        "use_implicit_solver": True,
        "use_memory": True,
        "use_moe": True,
        "use_factorization": True,
    }
    config.update(kwargs)
    return IDIRKSModel(**config)


def create_idir_ks_large(**kwargs) -> IDIRKSModel:
    """Factory function for large IDIR-KS model"""
    config = {
        "vocab_size": 50000,
        "dim": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "num_experts": 16,
        "expert_top_k": 2,
        "num_memories": 8192,
        "max_seq_len": 8192,
        "dropout": 0.1,
        "convergence_threshold": 1e-4,
        "max_solver_steps": 16,
        "min_solver_steps": 6,
        "enable_adaptive": True,
        "num_trajectories": 1,
        "trajectory_noise": 0.0,
        "use_implicit_solver": True,
        "use_memory": True,
        "use_moe": True,
        "use_factorization": True,
    }
    config.update(kwargs)
    return IDIRKSModel(**config)
