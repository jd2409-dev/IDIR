"""Internal Memory Retrieval Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryModule(nn.Module):
    """
    Internal memory retrieval system.
    Memory(h) = softmax(h M^T) M

    This implements differentiable memory access for the implicit solver.
    """

    def __init__(self, dim: int, num_memories: int = 4096, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_memories = num_memories
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Memory matrix M ∈ R^(N × d)
        self.memory = nn.Parameter(torch.randn(num_memories, dim) * 0.02)

        # Query projection for memory retrieval
        self.query_proj = nn.Linear(dim, dim)

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Gating mechanism for memory integration
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, dim), nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize memory and projections"""
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory: Memory(h) = softmax(h M^T) M

        Args:
            h: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Memory-augmented tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = h.shape

        # Compute queries
        queries = self.query_proj(h)  # (batch, seq_len, dim)

        # Multi-head attention over memory
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        # Memory is shared across all heads
        memory_reshaped = self.memory.view(
            self.num_memories, self.num_heads, self.head_dim
        )
        memory_reshaped = memory_reshaped.transpose(0, 1)  # (heads, N, head_dim)

        # Compute attention scores: h @ M^T
        scores = torch.matmul(queries, memory_reshaped.transpose(-2, -1))
        scores = scores / self.temperature  # Scale by temperature

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, seq_len, N)

        # Retrieve from memory: attn_weights @ M
        retrieved = torch.matmul(
            attn_weights, memory_reshaped
        )  # (batch, heads, seq_len, head_dim)

        # Reshape back
        retrieved = retrieved.transpose(1, 2).contiguous()
        retrieved = retrieved.view(batch_size, seq_len, dim)

        # Output projection
        retrieved = self.out_proj(retrieved)

        # Gating
        gate = self.gate(h)
        retrieved = gate * retrieved

        return retrieved

    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage"""
        with torch.no_grad():
            # Compute average attention entropy (measure of specialization)
            # Use a random sample to avoid computing on full batch
            sample_queries = self.memory[:128]  # Sample memories as queries
            sample_queries = sample_queries.view(128, self.num_heads, self.head_dim)
            sample_queries = sample_queries.transpose(0, 1)  # (heads, 128, head_dim)

            memory_reshaped = self.memory.view(
                self.num_memories, self.num_heads, self.head_dim
            )
            memory_reshaped = memory_reshaped.transpose(0, 1)  # (heads, N, head_dim)

            scores = torch.matmul(sample_queries, memory_reshaped.transpose(-2, -1))
            scores = scores / self.temperature
            attn_weights = F.softmax(scores, dim=-1)

            # Compute entropy (lower entropy = more concentrated attention)
            entropy = (
                -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
            )

            return {
                "memory_entropy": entropy.item(),
                "max_attention": attn_weights.max().item(),
                "mean_attention": attn_weights.mean().item(),
            }


class StructuredMemory(nn.Module):
    """
    Hierarchical structured memory with different timescales.
    Inspired by Compressive Transformer and related work.
    """

    def __init__(self, dim: int, num_short_term: int = 1024, num_long_term: int = 4096):
        super().__init__()
        self.dim = dim
        self.num_short_term = num_short_term
        self.num_long_term = num_long_term

        # Short-term memory (fast access, limited capacity)
        self.short_term = MemoryModule(dim, num_short_term)

        # Long-term memory (slower but larger)
        self.long_term = MemoryModule(dim, num_long_term, num_heads=4)

        # Compression network (short-term -> long-term)
        self.compressor = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # Balance parameter
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Retrieve from both short and long-term memory"""
        short_out = self.short_term(h)
        long_out = self.long_term(h)

        # Weighted combination
        alpha = torch.sigmoid(self.alpha)
        return alpha * short_out + (1 - alpha) * long_out

    def compress(self):
        """Compress short-term memories into long-term storage"""
        # This would be called periodically during training
        # to prevent short-term memory from filling up
        pass
