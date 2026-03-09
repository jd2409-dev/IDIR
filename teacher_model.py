import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_size, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None):
        # Self-attention
        normalized_x = self.norm1(x)
        attn_output, _ = self.attn(
            normalized_x,
            normalized_x,
            normalized_x,
            attn_mask=attn_mask,
            is_causal=False,
        )
        x = x + self.dropout1(attn_output)

        # Feed-forward
        normalized_x = self.norm2(x)
        ffn_output = self.ffn(normalized_x)
        x = x + self.dropout2(ffn_output)
        return x


class TeacherModel(nn.Module):
    """
    A simplified 300M Dense Transformer Decoder model for distillation.
    Approximates the architecture and parameter count of a GPT-style model.
    """

    def __init__(
        self,
        vocab_size=50000,
        max_seq_len=1024,
        hidden_size=768,
        num_layers=32,
        num_heads=12,
        ffn_hidden_size=3072,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hidden_size, num_heads, ffn_hidden_size, dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm and output projection
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(
            hidden_size, vocab_size, bias=False
        )  # Often tied to embedding or no bias

        # Initialize weights (optional, but good practice for Transformers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        Forward pass for the Teacher Model.
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token and positional embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        x = token_embeds + position_embeds

        # Causal mask for decoder
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def get_parameter_count(self):
        """
        Calculates the total number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params


if __name__ == "__main__":
    # Example usage and parameter count verification
    print("Instantiating Teacher Model...")
    model = TeacherModel()

    param_count = model.get_parameter_count()
    print(f"Total parameters in Teacher Model: {param_count:,}")

    # Expected: ~300M parameters.
    # Our calculation was around 304M, which is good enough as an approximation.
    assert param_count > 290_000_000 and param_count < 310_000_000

    print("\nTesting forward pass...")
    input_tensor = torch.randint(
        0, model.vocab_size, (2, 128)
    )  # Batch size 2, seq_len 128
    logits = model(input_tensor)
    print(f"Forward pass successful. Output logits shape: {logits.shape}")
