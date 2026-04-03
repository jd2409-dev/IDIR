#!/usr/bin/env python3
"""
RTX 3050 4GB Optimized Training Script for IDIR-KS

Runs for exactly 1 hour on an RTX 3050 with 4GB VRAM.
- Mixed precision (FP16) training
- Gradient accumulation for effective batch size
- Streaming high-quality open datasets
- Synthetic data fallback (zero storage)
- Auto-checkpointing every 10 minutes
- OOM recovery with automatic batch reduction

Usage:
    python train_rtx3050.py                  # Default 1-hour run
    python train_rtx3050.py --hours 0.5      # 30-minute run
    python train_rtx3050.py --resume         # Resume from latest checkpoint
    python train_rtx3050.py --cpu            # CPU fallback for testing
"""

import sys
import os
import argparse
import time
import torch
import gc
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idir_ks.model.idir_ks_model import IDIRKSModel
from idir_ks.training.trainer import IDIRKSTrainer
from idir_ks.training.data import create_composite_dataset, create_dataloader
from idir_ks.data.open_datasets import create_open_dataloader
from idir_ks.data.synthetic_data import SyntheticDataGenerator
from idir_ks.utils.config import get_rtx3050_config
from idir_ks.main import create_model, set_seed


def get_device(force_cpu: bool = False) -> str:
    """Get best available device"""
    if force_cpu:
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Check if it's actually a low-VRAM GPU
        if gpu_mem < 6:
            print(f"  Low VRAM detected ({gpu_mem:.1f} GB), using optimized settings")
        return device

    print("No GPU available, using CPU (will be slow)")
    return "cpu"


def create_rtx3050_dataloaders(config, device: str, seed: int = 42):
    """
    Create dataloaders optimized for RTX 3050 4GB.

    Strategy:
    1. Try loading high-quality open datasets (streaming, zero storage)
    2. Fall back to synthetic data if HF datasets unavailable
    3. Use short sequences (128 tokens) for speed
    4. Small batch size (4) with gradient accumulation
    """
    max_length = config.model.max_seq_len
    batch_size = config.training.batch_size
    total_samples = config.data.total_samples

    print(f"\nData loading (max_length={max_length}, samples={total_samples})...")

    # Try open datasets first (streaming)
    try:
        print("Attempting to load high-quality open datasets...")
        train_dl = create_open_dataloader(
            weights={
                "code": config.data.code_weight,
                "math": config.data.math_weight,
                "logic": config.data.logic_weight,
                "language": config.data.language_weight,
            },
            batch_size=batch_size,
            max_samples=total_samples,
            max_length=max_length,
            seed=seed,
            shuffle=True,
        )

        # Small validation set
        val_dl = create_open_dataloader(
            weights={
                "code": 0.25,
                "math": 0.25,
                "logic": 0.25,
                "language": 0.25,
            },
            batch_size=batch_size,
            max_samples=min(500, total_samples // 20),
            max_length=max_length,
            seed=seed + 100,
            shuffle=False,
        )

        print("Using open datasets (streaming)")
        return train_dl, val_dl

    except Exception as e:
        print(f"Open datasets failed ({str(e)[:60]}), using synthetic data...")

    # Fallback: synthetic data
    max_length = min(config.data.max_length, config.model.max_seq_len)

    train_dataset = create_composite_dataset(
        code_path=config.data.code_path,
        math_path=config.data.math_path,
        logic_path=config.data.logic_path,
        language_path=config.data.language_path,
        weights={
            "code": config.data.code_weight,
            "math": config.data.math_weight,
            "logic": config.data.logic_weight,
            "language": config.data.language_weight,
        },
        max_length=max_length,
        total_samples=total_samples,
        tokenizer=None,
        seed=seed,
    )

    train_dl = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_dataset = create_composite_dataset(
        code_path=config.data.code_path,
        math_path=config.data.math_path,
        logic_path=config.data.logic_path,
        language_path=config.data.language_path,
        weights={"code": 0.25, "math": 0.25, "logic": 0.25, "language": 0.25},
        max_length=max_length,
        total_samples=min(500, total_samples // 20),
        tokenizer=None,
        seed=seed + 100,
    )

    val_dl = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("Using synthetic data")
    return train_dl, val_dl


def test_model_quality(model, device, vocab_size):
    """Quick quality check: generate a sample and verify it's not garbage"""
    model.eval()
    prompt_len = 8
    input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    with torch.no_grad():
        try:
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50,
            )
            generated = output[0, prompt_len:].tolist()

            # Check: output should have variety (not all same token)
            unique_tokens = len(set(generated))
            quality = "GOOD" if unique_tokens > 3 else "LOW"

            print(f"\n  Generation quality check:")
            print(f"    Prompt tokens: {prompt_len}")
            print(f"    Generated tokens: {len(generated)}")
            print(f"    Unique tokens: {unique_tokens}/{len(generated)}")
            print(f"    Quality: {quality}")

            return unique_tokens > 1

        except Exception as e:
            print(f"  Generation test failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="IDIR-KS: RTX 3050 4GB Optimized Training"
    )
    parser.add_argument(
        "--hours", type=float, default=1.0, help="Training time in hours (default: 1.0)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU (for testing)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 70)
    print("IDIR-KS: RTX 3050 4GB Optimized Training")
    print("=" * 70)

    # Device
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    # Seed
    set_seed(args.seed)

    # Config
    config = get_rtx3050_config()
    config.device = device
    config.seed = args.seed

    print(f"\nModel config:")
    print(f"  Vocab: {config.model.vocab_size}")
    print(f"  Dim: {config.model.dim}")
    print(f"  Layers: {config.model.num_layers}")
    print(f"  Heads: {config.model.num_heads}")
    print(f"  Experts: {config.model.num_experts}")
    print(f"  Seq len: {config.model.max_seq_len}")
    print(f"  Batch: {config.training.batch_size}")
    print(f"  Grad accum: 4")
    print(f"  Max steps: {config.training.max_steps}")

    # Create model
    print(f"\nCreating model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb = total_params * 4 / (1024 * 1024)  # FP32 size
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size (FP32): {model_mb:.1f} MB")
    print(f"  Model size (FP16): {model_mb / 2:.1f} MB")

    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory: {gpu_mem:.1f} GB")
        print(
            f"  Model uses ~{model_mb / 2:.1f} MB in FP16 (of {gpu_mem * 1024:.0f} MB)"
        )

    # Dataloaders
    train_dl, val_dl = create_rtx3050_dataloaders(config, device, args.seed)

    # Verify a batch works
    batch = next(iter(train_dl))
    print(f"\nBatch verification:")
    print(f"  Input shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")

    # Trainer
    trainer_config = {
        "lambda_consistency": config.training.lambda_consistency,
        "lambda_entropy": config.training.lambda_entropy,
        "phase1_steps": 200,
        "phase2_steps": 800,
        "phase3_steps": 200,
        "max_steps": config.training.max_steps,
        "grad_accum_steps": 4,
        "grad_clip": config.training.grad_clip,
        "use_multi_trajectory": False,
        "num_trajectories": 1,
    }

    trainer = IDIRKSTrainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config=trainer_config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        max_time_hours=args.hours,
    )

    # Resume if requested
    latest_ckpt = Path(args.checkpoint_dir) / "latest.pt"
    if args.resume and latest_ckpt.exists():
        print(f"\nResuming from {latest_ckpt}...")
        trainer.load_checkpoint(str(latest_ckpt))

    # Pre-training quality check
    print("\nPre-training quality check...")
    test_model_quality(model, device, config.model.vocab_size)

    # Train
    print(f"\nStarting training ({args.hours} hour budget)...")
    trainer.train()

    # Post-training quality check
    print("\nPost-training quality check...")
    quality_ok = test_model_quality(model, device, config.model.vocab_size)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Steps completed: {trainer.step}")
    print(f"  Epochs completed: {trainer.epoch}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    final_loss = f"{trainer.loss_history[-1]:.4f}" if trainer.loss_history else "N/A"
    print(f"  Final loss: {final_loss}")
    print(f"  Model quality: {'PASS' if quality_ok else 'NEEDS MORE TRAINING'}")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print("=" * 70)

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
