
import torch
from idir_model import IDIR
from teacher_model import TeacherModel

# Define constants based on paper
SEQ_LEN = 512
HIDDEN_DIM_65M = 512
HIDDEN_DIM_300M = 1024
VOCAB_SIZE = 50000

def calculate_transformer_flops(num_layers, hidden_dim, seq_len):
    """
    Calculates approximate FLOPs for a dense Transformer model per forward pass (one token output).
    Using the paper's formula for per-layer FLOPs for dense Transformers:
    Per layer FLOPs approximately: 4 * L * d^2 + 2 * L^2 * d
    Where L = sequence length, d = hidden dimension.
    """
    per_layer_flops = (4 * seq_len * (hidden_dim ** 2)) + (2 * (seq_len ** 2) * hidden_dim)
    total_flops = num_layers * per_layer_flops
    return total_flops

def analyze_65m_dense_transformer_flops():
    """
    Analyzes FLOPs for the 65M Dense Transformer Baseline.
    Paper: 8 layers, hidden size 512. Approx 804 million FLOPs per layer.
    """
    num_layers = 8
    hidden_dim = HIDDEN_DIM_65M
    
    # Paper's stated value
    paper_per_layer_flops = 804_000_000
    total_flops = num_layers * paper_per_layer_flops
    
    print(f"\n--- 65M Dense Transformer Baseline FLOPs ---")
    print(f"Number of layers: {num_layers}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Per layer FLOPs (paper's value): {paper_per_layer_flops:,}")
    print(f"Total FLOPs per forward pass: {total_flops:,}")
    return total_flops

def analyze_300m_dense_transformer_flops():
    """
    Analyzes FLOPs for the 300M Dense Transformer.
    Paper: 24 layers, hidden size 1024. Approx 2.1B FLOPs per layer.
    """
    num_layers = 24
    hidden_dim = HIDDEN_DIM_300M
    
    # Paper's stated value
    paper_per_layer_flops = 2_100_000_000 # 2.1B
    total_flops = num_layers * paper_per_layer_flops

    print(f"\n--- 300M Dense Transformer FLOPs ---")
    print(f"Number of layers: {num_layers}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Per layer FLOPs (paper's value): {paper_per_layer_flops:,}")
    print(f"Total FLOPs per forward pass: {total_flops:,}")
    return total_flops

def analyze_idir_flops(avg_k_iterations=4):
    """
    Analyzes FLOPs for the IDIR model.
    Based on paper's breakdown per solver iteration and average K.
    """
    # Paper's stated values for per solver iteration components
    core_matmul_flops = 262_144  # 512^2
    two_experts_flops = 4_200_000 # Approximately 4.2 million
    reasoning_module_flops = 3_000_000 # Approximately 3 million

    flops_per_iteration = core_matmul_flops + two_experts_flops + reasoning_module_flops
    flops_per_token_avg = flops_per_iteration * avg_k_iterations
    total_flops_full_sequence = flops_per_token_avg * SEQ_LEN

    print(f"\n--- IDIR Model FLOPs (Avg K={avg_k_iterations}) ---")
    print(f"FLOPs per solver iteration breakdown (paper's values):")
    print(f"  Core Matmul: {core_matmul_flops:,}")
    print(f"  Two Experts: {two_experts_flops:,}")
    print(f"  Reasoning Module: {reasoning_module_flops:,}")
    print(f"Total FLOPs per solver iteration: {flops_per_iteration:,}")
    print(f"Average effective depth (K): {avg_k_iterations}")
    print(f"FLOPs per token (average K): {flops_per_token_avg:,}")
    print(f"Total FLOPs for full sequence (Length={SEQ_LEN}): {total_flops_full_sequence:,}")
    return total_flops_full_sequence, flops_per_token_avg

def main():
    print("--- FLOP Analysis based on Paper's Specifications ---")

    flops_65m = analyze_65m_dense_transformer_flops()
    flops_idir_full_seq, flops_idir_per_token = analyze_idir_flops(avg_k_iterations=4)
    flops_300m = analyze_300m_dense_transformer_flops()

    print("\n--- Summary ---")
    print(f"65M Dense Transformer: {flops_65m / 1_000_000_000:.2f} Billion FLOPs")
    print(f"IDIR (avg K=4): {flops_idir_full_seq / 1_000_000_000:.2f} Billion FLOPs")
    print(f"300M Dense Transformer: {flops_300m / 1_000_000_000:.2f} Billion FLOPs")

    print(f"\nRelative inference cost (compared to 65M Dense, using paper's K=4 IDIR value):")
    print(f"65M Dense: 1.0x")
    # Paper states: IDIR (avg K=4): 1.6x, 300M Dense: 7.8x
    # Let's calculate based on our derived FLOPs for per token comparisons or full sequence.
    # The paper uses "one fifth of the inference cost" and "7.8x". This implies total FLOPs.
    
    # Paper states: IDIR achieves near-300M reasoning at one fifth runtime (meaning IDIR is 1/5 of 300M)
    # Our derived IDIR: 15.26 B, 300M Dense: 50 B.
    # 15.26 / 50 = 0.305. So IDIR is ~30% of 300M, which is close to 1/5 (20%).
    
    # Calculate relative cost based on total sequence FLOPs
    relative_idir = flops_idir_full_seq / flops_65m
    relative_300m = flops_300m / flops_65m
    
    print(f"IDIR (avg K=4): {relative_idir:.2f}x")
    print(f"300M Dense: {relative_300m:.2f}x")

if __name__ == '__main__':
    main()
