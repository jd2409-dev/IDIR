"""Example: Using IDIR-KS for Language Modeling"""

import torch
from idir_ks.model import IDIRKSModel, create_idir_ks_base
from idir_ks.utils import get_small_config


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("IDIR-KS Basic Usage Example")
    print("=" * 60)

    # Create a small model for testing
    print("\n1. Creating model...")
    model = create_idir_ks_base(
        vocab_size=10000,
        dim=256,
        num_layers=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {total_params:,} parameters")

    # Create dummy input
    print("\n2. Creating input...")
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")

    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, 10000)")

    assert logits.shape == (batch_size, seq_len, 10000), "Shape mismatch!"
    print("   Forward pass successful!")

    # Show model statistics
    print("\n4. Model Statistics:")
    stats = model.get_model_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:,}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def example_multi_trajectory():
    """Multi-trajectory inference example"""
    print("\n" + "=" * 60)
    print("Multi-Trajectory Inference Example")
    print("=" * 60)

    # Create model with multi-trajectory enabled
    print("\n1. Creating model with multi-trajectory...")
    model = create_idir_ks_base(
        vocab_size=10000,
        dim=256,
        num_layers=2,
        num_trajectories=3,
        trajectory_noise=0.01,
    )
    print(f"   Model created")

    # Create input
    print("\n2. Creating input...")
    input_ids = torch.randint(0, 10000, (2, 32))

    # Multi-trajectory forward
    print("\n3. Running multi-trajectory inference...")
    model.eval()
    with torch.no_grad():
        consistent_logits, trajectory_logits = model.forward_multi_trajectory(
            input_ids, num_trajectories=3
        )

    print(f"   Trajectory logits shape: {trajectory_logits.shape}")
    print(f"   Consistent logits shape: {consistent_logits.shape}")
    print("   Multi-trajectory inference successful!")

    print("\n" + "=" * 60)


def example_ablation_variants():
    """Test different ablation variants"""
    print("\n" + "=" * 60)
    print("Ablation Variants Test")
    print("=" * 60)

    from idir_ks.evaluation.ablations import create_ablation_variant

    variants = ["A", "B", "C", "D", "E", "F", "G"]

    for variant in variants:
        print(f"\nTesting variant {variant}...")
        model = create_ablation_variant(
            variant, vocab_size=10000, dim=256, num_layers=2
        )

        # Test forward pass
        input_ids = torch.randint(0, 10000, (2, 16))
        with torch.no_grad():
            logits = model(input_ids)

        params = sum(p.numel() for p in model.parameters())
        print(f"   Variant {variant}: {params:,} parameters - OK")

    print("\n" + "=" * 60)
    print("All ablation variants tested successfully!")
    print("=" * 60)


def example_configuration():
    """Configuration example"""
    print("\n" + "=" * 60)
    print("Configuration Example")
    print("=" * 60)

    from idir_ks.utils import get_base_config, get_large_config, get_small_config

    print("\n1. Base configuration:")
    config = get_base_config()
    print(f"   Model dim: {config.model.dim}")
    print(f"   Num layers: {config.model.num_layers}")
    print(f"   Num experts: {config.model.num_experts}")

    print("\n2. Large configuration:")
    config = get_large_config()
    print(f"   Model dim: {config.model.dim}")
    print(f"   Num layers: {config.model.num_layers}")
    print(f"   Num experts: {config.model.num_experts}")

    print("\n3. Small configuration:")
    config = get_small_config()
    print(f"   Model dim: {config.model.dim}")
    print(f"   Num layers: {config.model.num_layers}")
    print(f"   Num experts: {config.model.num_experts}")

    print("\n" + "=" * 60)


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  IDIR-KS: Knowledge-Dense Implicit Reasoning  ".center(78) + "║")
    print("║" + "  " + "=" * 72 + "  ║")
    print("║" + "  Usage Examples  ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    example_basic_usage()
    example_multi_trajectory()
    example_ablation_variants()
    example_configuration()

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  All examples completed successfully!  ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    main()
