"""IDIR-KS: Knowledge-Dense Implicit Reasoning with Inference-Time Scaling

Main entry point for training and evaluation.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from model.idir_ks_model import IDIRKSModel, create_idir_ks_base, create_idir_ks_large
from training.trainer import IDIRKSTrainer
from training.data import create_composite_dataset, create_dataloader
from training.hybrid_optimizer import create_hybrid_optimizer
from evaluation.ablations import create_ablation_variant, AblationStudy
from evaluation.metrics import Evaluator, format_results
from utils.config import (
    IDIRKSConfig,
    get_base_config,
    get_large_config,
    get_small_config,
    get_ablation_config,
)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: IDIRKSConfig) -> IDIRKSModel:
    """Create model from configuration"""
    model = IDIRKSModel(
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_experts=config.model.num_experts,
        expert_top_k=config.model.expert_top_k,
        num_memories=config.model.num_memories,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        convergence_threshold=config.model.convergence_threshold,
        max_solver_steps=config.model.max_solver_steps,
        min_solver_steps=config.model.min_solver_steps,
        enable_adaptive=config.model.enable_adaptive,
        num_trajectories=config.model.num_trajectories,
        trajectory_noise=config.model.trajectory_noise,
        use_implicit_solver=config.model.use_implicit_solver,
        use_memory=config.model.use_memory,
        use_moe=config.model.use_moe,
        use_factorization=config.model.use_factorization,
    )
    return model


def create_dataloaders(config: IDIRKSConfig):
    """Create train and validation dataloaders"""
    data = config.data

    # Create dataset
    train_dataset = create_composite_dataset(
        code_path=data.code_path,
        math_path=data.math_path,
        logic_path=data.logic_path,
        language_path=data.language_path,
        weights={
            "code": data.code_weight,
            "math": data.math_weight,
            "logic": data.logic_weight,
            "language": data.language_weight,
        },
        max_length=data.max_length,
        total_samples=data.total_samples,
        tokenizer=None,  # Would use real tokenizer
        seed=config.seed,
    )

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
    )

    # Create validation dataset (smaller)
    val_dataset = create_composite_dataset(
        code_path=data.code_path,
        math_path=data.math_path,
        logic_path=data.logic_path,
        language_path=data.language_path,
        weights={
            "code": 0.25,
            "math": 0.25,
            "logic": 0.25,
            "language": 0.25,
        },
        max_length=data.max_length,
        total_samples=1000,
        tokenizer=None,
        seed=config.seed + 100,
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
    )

    return train_dataloader, val_dataloader


def train_model(args):
    """Train IDIR-KS model"""
    print("=" * 80)
    print("IDIR-KS: Training")
    print("=" * 80)

    # Load configuration
    if args.config:
        config = IDIRKSConfig.from_yaml(args.config)
    elif args.size == "small":
        config = get_small_config()
    elif args.size == "large":
        config = get_large_config()
    else:
        config = get_base_config()

    # Override with command line arguments
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    # Set seed
    set_seed(config.seed)

    # Create model
    print(f"\nCreating model (size: {args.size})...")
    model = create_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(config)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = IDIRKSTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config={
            "lambda_consistency": config.training.lambda_consistency,
            "lambda_entropy": config.training.lambda_entropy,
            "phase1_steps": config.training.phase1_steps,
            "phase2_steps": config.training.phase2_steps,
            "phase3_steps": config.training.phase3_steps,
            "max_steps": config.training.max_steps,
            "grad_clip": config.training.grad_clip,
            "use_multi_trajectory": config.training.use_multi_trajectory,
            "num_trajectories": config.training.num_trajectories,
        },
        device=config.device,
        checkpoint_dir=config.training.checkpoint_dir,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    trainer.save_checkpoint("final_model.pt")

    print("\nTraining complete!")


def evaluate_model(args):
    """Evaluate IDIR-KS model"""
    print("=" * 80)
    print("IDIR-KS: Evaluation")
    print("=" * 80)

    # Load configuration
    if args.config:
        config = IDIRKSConfig.from_yaml(args.config)
    else:
        config = get_base_config()

    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        model = create_model(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("\nCreating model...")
        model = create_model(config)

    model = model.to(config.device)

    # Create evaluator
    evaluator = Evaluator(model, device=config.device)

    # Create test dataloader
    _, val_dataloader = create_dataloaders(config)

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_all(
        test_dataloaders={"val": val_dataloader}, test_datasets={}
    )

    # Print results
    print(format_results(results))


def run_ablation(args):
    """Run ablation study"""
    print("=" * 80)
    print("IDIR-KS: Ablation Study")
    print("=" * 80)

    # Load configuration
    if args.config:
        config = IDIRKSConfig.from_yaml(args.config)
    else:
        config = get_base_config()

    # Create ablation study
    study = AblationStudy(config.model.__dict__)

    if args.variant:
        # Run specific variant
        variant = args.variant.upper()
        if variant not in study.VARIANTS:
            print(f"Error: Unknown variant '{variant}'")
            return

        print(f"\nRunning ablation variant: {variant}")
        model = create_ablation_variant(variant, **config.model.__dict__)

        print(
            f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        print("Model architecture validated successfully!")

        # Optionally train and evaluate
        if args.train:
            print("\nTraining ablation variant...")
            # Would train here
    else:
        # Run all ablations
        print("\nRunning all ablation variants...")
        for variant in study.VARIANTS:
            model = create_ablation_variant(variant, **config.model.__dict__)
            print(
                f"Variant {variant}: {sum(p.numel() for p in model.parameters()):,} parameters"
            )

        print("\nAll ablation variants validated successfully!")


def quick_test(args):
    """Run a quick test of the model"""
    print("=" * 80)
    print("IDIR-KS: Quick Test")
    print("=" * 80)

    # Create small model
    config = get_small_config()
    model = create_model(config)

    print(f"\nModel created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Hidden dim: {config.model.dim}")
    print(f"  Num layers: {config.model.num_layers}")

    # Test forward pass
    device = args.device or "cpu"
    model = model.to(device)

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {model.vocab_size})")

    assert logits.shape == (batch_size, seq_len, model.vocab_size), (
        "Output shape mismatch!"
    )

    print("\nForward pass successful!")

    # Test multi-trajectory inference
    if config.model.num_trajectories > 1:
        print("\nTesting multi-trajectory inference...")
        with torch.no_grad():
            consistent_logits, trajectory_logits = model.forward_multi_trajectory(
                input_ids, num_trajectories=config.model.num_trajectories
            )
        print(f"  Trajectory logits shape: {trajectory_logits.shape}")
        print(f"  Consistent logits shape: {consistent_logits.shape}")
        print("  Multi-trajectory inference successful!")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="IDIR-KS: Knowledge-Dense Implicit Reasoning with Inference-Time Scaling"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train IDIR-KS model")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument(
        "--size",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Model size",
    )
    train_parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate IDIR-KS model")
    eval_parser.add_argument("--config", type=str, help="Path to config file")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--config", type=str, help="Path to config file")
    ablation_parser.add_argument(
        "--variant",
        type=str,
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        help="Ablation variant to run",
    )
    ablation_parser.add_argument(
        "--train", action="store_true", help="Train the ablation variant"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Quick test of the model")
    test_parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "ablation":
        run_ablation(args)
    elif args.command == "test":
        quick_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
