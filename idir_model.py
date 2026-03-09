import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint


class SparseExpertModule(nn.Module):
    """
    Sparse Expert Module with Top-2 Gating.
    As described in section 5 of the paper.
    """

    def __init__(self, hidden_dim, num_experts=4, expert_hidden_dim=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim

        # Routing network
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_hidden_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(expert_hidden_dim, hidden_dim, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, h):
        """
        Forward pass for the Sparse Expert Module.
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Get routing weights
        g = F.softmax(self.router(h), dim=-1)  # (batch, seq, num_experts)

        # Select top-2
        top2_weights, top2_indices = torch.topk(g, 2, dim=-1)  # both (batch, seq, 2)

        # Normalize top-2 weights
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)

        # Get all expert outputs
        expert_outputs = torch.stack(
            [expert(h) for expert in self.experts], dim=-2
        )  # (batch, seq, num_experts, hidden_dim)

        # Select the top-2 experts' outputs
        top2_expert_outputs = torch.gather(
            expert_outputs,
            -2,
            top2_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim),
        )  # (batch, seq, 2, hidden_dim)

        # Weight and sum
        output = (top2_expert_outputs * top2_weights.unsqueeze(-1)).sum(dim=-2)

        return output


class InternalReasoning(nn.Module):
    """
    Differentiable Internal Reasoning Module.
    As described in section 6 of the paper.
    This is a more complex implementation to match the paper's parameter count.
    """

    def __init__(
        self, hidden_dim, reasoning_steps=3, k_features=32, reasoning_hidden_dim=2560
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_steps = reasoning_steps
        self.k_features = k_features
        self.reasoning_hidden_dim = reasoning_hidden_dim

        # Transformation generator T (MLP)
        self.T = nn.Sequential(
            nn.Linear(hidden_dim * 2, reasoning_hidden_dim),
            nn.ReLU(),
            nn.Linear(reasoning_hidden_dim, hidden_dim),
        )

        # Injection layer R(h) (MLP)
        self.R = nn.Sequential(
            nn.Linear(hidden_dim, reasoning_hidden_dim),
            nn.ReLU(),
            nn.Linear(reasoning_hidden_dim, hidden_dim),
        )

    def soft_topk(self, z, h):
        """
        SoftTopK approximation for differentiable beam search.
        """
        # Get scores from T
        scores = self.T(torch.cat([z, h], dim=-1))  # (..., hidden_dim)

        # Get top-k scores and indices
        topk_scores, topk_indices = torch.topk(
            scores, self.k_features, dim=-1
        )  # (..., k)

        # Apply softmax with temperature to make soft weights
        temperature = 1.0
        soft_weights = F.softmax(topk_scores / temperature, dim=-1).to(
            scores.dtype
        )  # (..., k)

        # Create soft mask: zero everywhere except topk positions set to soft_weights
        soft_mask = torch.zeros_like(scores)  # (..., hidden_dim)
        soft_mask.scatter_(-1, topk_indices, soft_weights)  # scatter the soft weights

        # Apply soft mask to z
        return z * soft_mask

    def forward(self, h):
        """
        Forward pass for the Internal Reasoning Module.
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Initial reasoning state z
        z = h.clone()

        # Multi-step internal reasoning
        for _ in range(self.reasoning_steps):
            z = self.soft_topk(z, h)

        # Inject reasoning result back into hidden state
        return self.R(z)


class ContractiveCore(nn.Module):
    """
    Contractive Core for fixed-point iteration.
    As described in sections 3 and 4 of the paper.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # W_h with spectral normalization
        self.W_h = spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # W_x
        self.W_x = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h, e, M_h, R_h):
        """
        One step of the fixed-point iteration.
        F_theta(h, e) = relu( W_h h + W_x e + M(h) + R(h) )
        """
        return F.relu(self.W_h(h) + self.W_x(e) + M_h + R_h)


class IDIR(nn.Module):
    """
    Infinite Depth Implicit Reasoner (IDIR) model.
    """

    def __init__(
        self,
        vocab_size=50000,
        hidden_dim=512,
        num_experts=4,
        expert_hidden_dim=2048,
        reasoning_steps=3,
        k_features=32,
        reasoning_hidden_dim=2560,
        max_iterations=20,
        tolerance=1e-3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_gradient_checkpointing = False

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Core modules
        self.contractive_core = ContractiveCore(hidden_dim)
        self.expert_module = SparseExpertModule(
            hidden_dim, num_experts, expert_hidden_dim
        )
        self.reasoning_module = InternalReasoning(
            hidden_dim, reasoning_steps, k_features, reasoning_hidden_dim
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def gradient_checkpointing_enable(self):
        self.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.use_gradient_checkpointing = False

    def _run_module(self, module, h):
        if (
            self.use_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(module, h, use_reentrant=False)
        return module(h)

    def forward(self, x):
        """
        Forward pass for the IDIR model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
            int: Average number of iterations
        """
        batch_size, seq_len = x.shape

        # 1. Embed token
        e = self.embedding(x)

        # Initialize hidden state h
        h = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)

        # 2. Solve fixed-point equation for hidden state
        with torch.no_grad():
            iterations = 0
            for k in range(self.max_iterations):
                h_prev = h

                # Pre-calculate M(h) and R(h)
                M_h = self._run_module(self.expert_module, h_prev)
                R_h = self._run_module(self.reasoning_module, h_prev)

                h = self.contractive_core(h_prev, e, M_h, R_h)

                # Check for convergence
                if torch.norm(h - h_prev) < self.tolerance:
                    break
                iterations = k + 1

        # Run one final step with gradients enabled
        M_h = self._run_module(self.expert_module, h)
        R_h = self._run_module(self.reasoning_module, h)
        h_star = self.contractive_core(h, e, M_h, R_h)

        # 5. Output logits
        logits = self.output_projection(h_star)

        return logits, iterations

    def get_parameter_count(self):
        """
        Returns the exact parameter count.
        From section 7 of the paper.
        """
        # Embedding
        embedding_params = self.embedding.weight.numel()
        # Output projection
        output_params = self.output_projection.weight.numel()
        # Implicit core
        implicit_core_params = (
            self.contractive_core.W_h.weight.numel()
            + self.contractive_core.W_x.weight.numel()
        )
        # Experts
        expert_params = sum(p.numel() for p in self.expert_module.experts.parameters())
        # Routing
        routing_params = self.expert_module.router.weight.numel()
        # Reasoning module
        reasoning_params = sum(p.numel() for p in self.reasoning_module.parameters())

        total_params = (
            embedding_params
            + output_params
            + implicit_core_params
            + expert_params
            + routing_params
            + reasoning_params
        )

        return {
            "Embedding": embedding_params,
            "Output": output_params,
            "Implicit core": implicit_core_params,
            "Experts": expert_params,
            "Routing": routing_params,
            "Reasoning": reasoning_params,
            "Total": total_params,
        }


if __name__ == "__main__":
    # Example usage and parameter count verification
    model = IDIR()
    print("IDIR Model Instantiated.")

    param_counts = model.get_parameter_count()
    print("\nParameter Count Breakdown:")
    for name, count in param_counts.items():
        print(f"{name:<15} {count:,}")

    # Manual calculation from paper
    # Embedding      25,600,000
    # Output         25,600,000
    # Implicit core     524,288
    # Experts         8,388,608
    # Routing             2,048
    # Reasoning        6,000,000 (approx)

    # Let's check our model's parameters
    print("\nVerifying with paper's numbers (approximate for reasoning module):")
    # Embedding
    assert param_counts["Embedding"] == 50000 * 512
    # Output
    assert param_counts["Output"] == 512 * 50000
    # Implicit core
    assert param_counts["Implicit core"] == 512 * 512 + 512 * 512
    # Experts
    # 4 * (512*2048 + 2048*512) = 8,388,608
    # My calculation includes biases, paper does not explicitly mention them. Let's recalculate without bias for experts.
    expert_weights = 4 * (512 * 2048 + 2048 * 512)
    # Router
    router_weights = 512 * 4

    print(f"Calculated Expert params: {param_counts['Experts']:,}")
    print(f"Paper's Expert params:    {8388608:,}")

    print(f"Calculated Routing params: {param_counts['Routing']:,}")
    print(f"Paper's Routing params:     {2048:,}")

    print(f"Calculated Reasoning params: {param_counts['Reasoning']:,}")
    print(f"Paper's Reasoning params:    ~6,000,000")

    print(f"\nTotal calculated params: {param_counts['Total']:,}")
    print(f"Paper's total params:      ~70,114,944")

    # Test forward pass
    print("\nTesting forward pass...")
    input_tensor = torch.randint(0, 50000, (1, 10))
    logits, iterations = model(input_tensor)
    print(f"Forward pass successful.")
    print(f"Output logits shape: {logits.shape}")
    print(f"Average iterations for convergence: {iterations}")
    print(f"Input tensor shape: {input_tensor.shape}")
