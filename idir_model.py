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

    def __init__(self, hidden_dim, num_experts=4, expert_hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim

        # Routing network with layer norm for stability
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.router_norm = nn.LayerNorm(hidden_dim)

        # Experts with dropout for regularization
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_hidden_dim, bias=False),
                    nn.LayerNorm(expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_dim, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        """
        Forward pass for the Sparse Expert Module.
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Normalize input
        h_norm = self.router_norm(h)

        # Get routing weights
        g = F.softmax(self.router(h_norm), dim=-1)  # (batch, seq, num_experts)

        # Select top-2
        top2_weights, top2_indices = torch.topk(g, 2, dim=-1)  # both (batch, seq, 2)

        # Normalize top-2 weights
        top2_weights = top2_weights / (top2_weights.sum(dim=-1, keepdim=True) + 1e-8)

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

        # Apply output normalization
        output = self.output_norm(output)

        return output


class InternalReasoning(nn.Module):
    """
    Differentiable Internal Reasoning Module.
    As described in section 6 of the paper.
    """

    def __init__(
        self, hidden_dim, reasoning_steps=3, k_features=32, reasoning_hidden_dim=2560,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_steps = reasoning_steps
        self.k_features = k_features
        self.reasoning_hidden_dim = reasoning_hidden_dim

        # Transformation generator T (MLP) with layer norm
        self.T = nn.Sequential(
            nn.Linear(hidden_dim * 2, reasoning_hidden_dim),
            nn.LayerNorm(reasoning_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reasoning_hidden_dim, hidden_dim),
        )

        # Injection layer R(h) (MLP) with layer norm
        self.R = nn.Sequential(
            nn.Linear(hidden_dim, reasoning_hidden_dim),
            nn.LayerNorm(reasoning_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reasoning_hidden_dim, hidden_dim),
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

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
        output = self.R(z)
        output = self.output_norm(output)

        return output


class ContractiveCore(nn.Module):
    """
    Contractive Core for fixed-point iteration.
    As described in sections 3 and 4 of the paper.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # W_h with spectral normalization
        self.W_h = spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # W_x
        self.W_x = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, e, M_h, R_h):
        """
        One step of the fixed-point iteration.
        F_theta(h, e) = relu( W_h h + W_x e + M(h) + R(h) )
        """
        # Combine all inputs
        combined = self.W_h(h) + self.W_x(e) + M_h + R_h

        # Apply ReLU activation
        activated = F.relu(combined)

        # Apply normalization and dropout
        output = self.norm(activated)
        output = self.dropout(output)

        return output


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
        tolerance=1e-4,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_gradient_checkpointing = False

        # Embedding layer with proper scaling
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Core modules
        self.contractive_core = ContractiveCore(hidden_dim, dropout)
        self.expert_module = SparseExpertModule(
            hidden_dim, num_experts, expert_hidden_dim, dropout
        )
        self.reasoning_module = InternalReasoning(
            hidden_dim, reasoning_steps, k_features, reasoning_hidden_dim, dropout
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Tie weights between embedding and output projection
        self.output_projection.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper schemes."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False

    def _run_module(self, module, h):
        """Run module with optional gradient checkpointing."""
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

        # 1. Embed token with dropout
        e = self.embed_dropout(self.embedding(x))

        # Initialize hidden state h
        h = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)

        # 2. Solve fixed-point equation for hidden state
        iterations = 0
        with torch.no_grad():
            for k in range(self.max_iterations):
                h_prev = h.detach().clone()

                # Pre-calculate M(h) and R(h)
                M_h = self._run_module(self.expert_module, h_prev)
                R_h = self._run_module(self.reasoning_module, h_prev)

                h = self.contractive_core(h_prev, e, M_h, R_h)

                # Check for convergence
                diff = torch.norm(h - h_prev)
                if diff < self.tolerance:
                    iterations = k + 1
                    break
                iterations = k + 1

        # 3. Run one final step with gradients enabled
        M_h = self._run_module(self.expert_module, h)
        R_h = self._run_module(self.reasoning_module, h)
        h_star = self.contractive_core(h, e, M_h, R_h)

        # 4. Output logits
        logits = self.output_projection(h_star)

        return logits, iterations

    def get_parameter_count(self):
        """
        Returns the exact parameter count.
        From section 7 of the paper.
        """
        # Embedding (tied with output)
        embedding_params = self.embedding.weight.numel()

        # Implicit core
        implicit_core_params = sum(
            p.numel() for p in self.contractive_core.parameters()
        )

        # Experts
        expert_params = sum(p.numel() for p in self.expert_module.experts.parameters())

        # Routing
        routing_params = self.expert_module.router.weight.numel()

        # Reasoning module
        reasoning_params = sum(p.numel() for p in self.reasoning_module.parameters())

        # Don't double count embedding/output
        total_params = (
            embedding_params
            + implicit_core_params
            + expert_params
            + routing_params
            + reasoning_params
        )

        return {
            "Embedding/Output (tied)": embedding_params,
            "Implicit core": implicit_core_params,
            "Experts": expert_params,
            "Routing": routing_params,
            "Reasoning": reasoning_params,
            "Total": total_params,
        }


class IDIRModel(IDIR):
    """
    Convenience wrapper that exposes a simple generation API for interactive usage.
    Includes repetition penalty, top-p (nucleus) sampling, and temperature scaling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _apply_repetition_penalty(self, logits, input_ids, repetition_penalty=1.2):
        """Apply repetition penalty to logits."""
        if repetition_penalty == 1.0:
            return logits

        # Count token frequencies in input
        token_counts = {}
        for token_id in input_ids.flatten():
            token_id = token_id.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        # Apply penalty
        for token_id, count in token_counts.items():
            if token_id < logits.shape[-1]:
                if logits[..., token_id] > 0:
                    logits[..., token_id] /= (repetition_penalty ** min(count, 10))
                else:
                    logits[..., token_id] *= (repetition_penalty ** min(count, 10))

        return logits

    def _top_p_filtering(self, logits, top_p=0.9, min_tokens_to_keep=1):
        """Filter logits using nucleus (top-p) sampling."""
        if top_p < 0 or top_p > 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _sample_next_tokens(
        self, logits, temperature=1.0, top_k=0, top_p=0.9,
        repetition_penalty=1.0, input_ids=None
    ):
        """Sample next tokens with various decoding strategies."""
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0 and input_ids is not None:
            logits = self._apply_repetition_penalty(
                logits, input_ids, repetition_penalty
            )

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Autoregressive generation with nucleus sampling and repetition penalty.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]

        if prompt_length >= max_length:
            return input_ids

        output_ids = input_ids.clone()
        eos_token_id = eos_token_id or pad_token_id or tokenizer.eos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - prompt_length):
            logits, _ = self(output_ids)
            next_logits = logits[:, -1, :]

            next_tokens = self._sample_next_tokens(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                input_ids=output_ids
            )

            output_ids = torch.cat([output_ids, next_tokens], dim=-1)

            if eos_token_id is not None:
                finished |= next_tokens.squeeze(-1) == eos_token_id
                if finished.all():
                    break

        return output_ids

    def beam_search_generate(
        self,
        input_ids,
        max_length=100,
        num_beams=4,
        temperature=1.0,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=1.0,
    ):
        """
        Beam search generation for higher quality outputs.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        vocab_size = self.vocab_size

        eos_token_id = eos_token_id or pad_token_id

        # Expand input for beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(
            batch_size * num_beams, -1
        )

        # Track beam scores
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam has good score

        done = [False for _ in range(batch_size)]

        for step in range(max_length - prompt_length):
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Log softmax for scores
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores.view(-1, 1)

            # Reshape for beam selection
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Select top beams
            topk_scores, topk_indices = torch.topk(
                next_token_scores, num_beams, dim=-1, largest=True, sorted=True
            )

            # Calculate beam indices and token ids
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # Update beams
            next_beams = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    continue

                for beam_idx in range(num_beams):
                    beam_token_idx = token_indices[batch_idx, beam_idx]
                    beam_token_idx = beam_token_idx.unsqueeze(0)

                    source_beam = beam_indices[batch_idx, beam_idx]
                    source_idx = batch_idx * num_beams + source_beam

                    beam_input = input_ids[source_idx].unsqueeze(0)
                    next_beams.append(
                        torch.cat([beam_input, beam_token_idx.unsqueeze(0)], dim=-1)
                    )

                    # Check for EOS
                    if beam_token_idx.item() == eos_token_id:
                        done[batch_idx] = True

            if all(done):
                break

            input_ids = torch.cat(next_beams, dim=0)
            beam_scores = topk_scores

        # Return best beam for each batch
        final_outputs = []
        for batch_idx in range(batch_size):
            start_idx = batch_idx * num_beams
            best_beam_idx = beam_scores[batch_idx].argmax()
            final_outputs.append(input_ids[start_idx + best_beam_idx].unsqueeze(0))

        return torch.cat(final_outputs, dim=0)


if __name__ == "__main__":
    # Example usage and parameter count verification
    model = IDIR()
    print("IDIR Model Instantiated.")

    param_counts = model.get_parameter_count()
    print("\nParameter Count Breakdown:")
    for name, count in param_counts.items():
        print(f"{name:<25} {count:,}")

    print(f"\nTotal parameters: {param_counts['Total']:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    input_tensor = torch.randint(0, 50000, (1, 10))
    logits, iterations = model(input_tensor)
    print(f"Forward pass successful.")
    print(f"Output logits shape: {logits.shape}")
    print(f"Iterations for convergence: {iterations}")
