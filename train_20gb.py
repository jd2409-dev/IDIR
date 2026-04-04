#!/usr/bin/env python3
"""
Training Script with 20GB Limit for IDIR-KS
Location: IDIR/train_20gb.py

This script includes automatic disk usage management to ensure training
stays under 20GB. All files (checkpoints, logs) are saved to the root
IDIR folder with no subdirectories.

Features:
- Automatic checkpoint cleanup (keeps only 3 most recent)
- Real-time disk usage monitoring
- Estimated space calculation before training
- Warning if approaching 20GB limit
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# Reduce memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Add idir_ks to path
sys.path.insert(0, os.path.dirname(__file__))

import torch
from idir_ks.main import create_model, create_dataloaders, set_seed, train_model
from idir_ks.utils.config import IDIRKSConfig, get_base_config
from idir_ks.training.trainer import IDIRKSTrainer
from idir_ks.utils.tokenizer import IDIRSTokenizer


class DiskManager:
    """Manages disk usage to stay under 20GB"""

    def __init__(self, max_gb=20, keep_checkpoints=3):
        self.max_bytes = max_gb * 1024 * 1024 * 1024
        self.safe_bytes = int(self.max_bytes * 0.9)  # 90% = 18GB
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_files = []

    def get_usage(self):
        """Get current directory usage in bytes"""
        total = 0
        for dirpath, _, filenames in os.walk("."):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total

    def get_usage_gb(self):
        """Get usage in GB"""
        return self.get_usage() / (1024**3)

    def check_space(self):
        """Check if under limit. Returns (ok, usage_gb)"""
        usage = self.get_usage()
        usage_gb = usage / (1024**3)
        safe_gb = self.safe_bytes / (1024**3)
        return usage < self.safe_bytes, usage_gb, safe_gb

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keep only most recent"""
        checkpoints = sorted(
            Path(".").glob("checkpoint_step_*.pt"), key=lambda x: x.stat().st_mtime
        )

        while len(checkpoints) > self.keep_checkpoints:
            old = checkpoints.pop(0)
            try:
                old.unlink()
                print(f"  Cleaned up old checkpoint: {old.name}")
            except Exception as e:
                print(f"  Warning: Could not remove {old}: {e}")

    def before_save(self):
        """Call before saving checkpoint"""
        ok, usage_gb, safe_gb = self.check_space()
        if not ok:
            print(
                f"⚠️  WARNING: Disk usage {usage_gb:.2f}GB approaching {safe_gb:.1f}GB limit!"
            )
            self.cleanup_old_checkpoints()
            ok, usage_gb, _ = self.check_space()
            if not ok:
                print(
                    "❌ CRITICAL: Still over limit! Free up space or reduce model size."
                )
                return False
        return True

    def after_save(self, checkpoint_path):
        """Call after saving checkpoint"""
        self.checkpoint_files.append(checkpoint_path)
        self.cleanup_old_checkpoints()


def estimate_space(model, num_checkpoints=6):
    """Estimate training space requirements"""
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # float32
    buffer_bytes = sum(b.numel() * 4 for b in model.buffers())
    model_mb = (param_bytes + buffer_bytes) / (1024**2)

    # Optimizer state (Adam: momentum + variance)
    opt_mb = model_mb * 2

    # Per checkpoint
    ckpt_mb = model_mb + opt_mb
    total_ckpt_mb = ckpt_mb * num_checkpoints

    # Working memory + cache estimate
    working_mb = 500
    cache_mb = 500

    total_gb = (total_ckpt_mb + working_mb + cache_mb) / 1024

    return {
        "model_mb": model_mb,
        "optimizer_mb": opt_mb,
        "checkpoint_mb": ckpt_mb,
        "num_checkpoints": num_checkpoints,
        "total_checkpoints_mb": total_ckpt_mb,
        "total_estimate_gb": total_gb,
        "safe": total_gb < 20,
    }


def train_with_20gb_limit(
    config_path=None, size="base", device="cuda", seed=42, tokenizer_path=None
):
    """Train with 20GB disk limit"""

    print("=" * 80)
    print("IDIR-KS Training with 20GB Limit")
    print("=" * 80)
    print(f"Location: IDIR/train_20gb.py")
    print(f"All files saved to root IDIR folder (no subdirectories)")
    print("=" * 80)

    # Setup
    set_seed(seed)

    # Load config
    if config_path:
        config = IDIRKSConfig.from_yaml(config_path)
    elif size == "small":
        from idir_ks.utils.config import get_small_config

        config = get_small_config()
    elif size == "large":
        from idir_ks.utils.config import get_large_config

        config = get_large_config()
    else:
        config = get_base_config()

    config.device = device
    config.seed = seed
    # Memory-efficient defaults for low-VRAM GPUs
    config.data.num_workers = 0
    config.data.pin_memory = False
    if config.training.batch_size > 8:
        config.training.gradient_accumulation_steps = max(
            1, config.training.batch_size // 8
        )
        config.training.batch_size = 8

    # Load tokenizer if provided
    tokenizer = None
    if tokenizer_path:
        print(f"\n🔤 Loading tokenizer from {tokenizer_path}...")
        tokenizer = IDIRSTokenizer(model_path=tokenizer_path)
        config.model.vocab_size = tokenizer.vocab_size
        print(f"   Vocabulary size: {tokenizer.vocab_size}")

    # Force checkpoints to root folder
    config.training.checkpoint_dir = "."
    config.training.log_dir = "."

    # Reduce save frequency to minimize disk usage
    config.training.save_interval = 5000  # Every 5000 steps

    print(f"\n📊 Configuration:")
    print(f"   Model size: {size}")
    print(f"   Hidden dim: {config.model.dim}")
    print(f"   Layers: {config.model.num_layers}")
    print(f"   Experts: {config.model.num_experts}")
    print(f"   Max steps: {config.training.max_steps}")
    print(f"   Save interval: every {config.training.save_interval} steps")
    print(f"   Device: {device}")

    # Create model and estimate space
    print(f"\n🔧 Creating model...")
    model = create_model(config)

    expected_checkpoints = config.training.max_steps // config.training.save_interval
    estimate = estimate_space(model, num_checkpoints=expected_checkpoints)

    print(f"\n💾 Space Estimate:")
    print(f"   Model size: {estimate['model_mb']:.1f} MB")
    print(f"   Per checkpoint: {estimate['checkpoint_mb']:.1f} MB")
    print(f"   Expected checkpoints: ~{expected_checkpoints}")
    print(f"   Total checkpoints: {estimate['total_checkpoints_mb']:.1f} MB")
    print(f"   Total estimate: {estimate['total_estimate_gb']:.2f} GB")
    print(f"   Safe under 20GB: {'✅ Yes' if estimate['safe'] else '⚠️  No'}")

    # Create dataloaders
    print(f"\n📚 Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(config, tokenizer)

    # Create trainer with disk manager
    print(f"\n🚀 Initializing trainer...")
    disk_manager = DiskManager(max_gb=20, keep_checkpoints=3)

    # Wrap the trainer's save_checkpoint method
    original_trainer_init = IDIRKSTrainer.__init__
    original_save_checkpoint = IDIRKSTrainer.save_checkpoint

    def patched_init(self, *args, **kwargs):
        original_trainer_init(self, *args, **kwargs)
        self._disk_manager = disk_manager

    def patched_save(self, filename, save_optimizer=False):
        if not self._disk_manager.before_save():
            print(f"Skipping checkpoint save due to disk limit")
            return
        original_save_checkpoint(self, filename, save_optimizer=save_optimizer)
        self._disk_manager.after_save(self.checkpoint_dir / filename)

    IDIRKSTrainer.__init__ = patched_init
    IDIRKSTrainer.save_checkpoint = patched_save

    # Create and run trainer
    trainer = IDIRKSTrainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        device=device,
        checkpoint_dir=".",
        save_interval=config.training.save_interval,
    )

    print(f"\n{'=' * 80}")
    print("Starting Training")
    print(f"{'=' * 80}\n")

    trainer.train()

    # Final save
    print(f"\n💾 Saving final model...")
    trainer.save_checkpoint("final_model.pt")

    # Report final usage
    usage_gb = disk_manager.get_usage_gb()
    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"\n📊 Final Disk Usage:")
    print(f"   Total used: {usage_gb:.2f} GB / 20 GB")
    print(f"   Remaining: {20 - usage_gb:.2f} GB")
    print(f"   Files in IDIR/: checkpoints + final_model.pt")
    print(f"{'=' * 80}")


def cleanup_checkpoints(keep_final=True):
    """Clean up checkpoint files"""
    print("🧹 Cleaning up checkpoints...")

    removed = []
    for ckpt in Path(".").glob("checkpoint_step_*.pt"):
        ckpt.unlink()
        removed.append(ckpt.name)

    if not keep_final:
        final = Path("final_model.pt")
        if final.exists():
            final.unlink()
            removed.append(final.name)

    if removed:
        print(f"   Removed: {', '.join(removed)}")
    else:
        print("   No checkpoints to clean up")

    # Show freed space
    dm = DiskManager()
    print(f"   Current usage: {dm.get_usage_gb():.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="IDIR-KS Training with 20GB Limit")

    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to SentencePiece model file",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up old checkpoints after training"
    )
    parser.add_argument(
        "--cleanup-only", action="store_true", help="Only cleanup, don't train"
    )

    args = parser.parse_args()

    if args.cleanup_only:
        cleanup_checkpoints(keep_final=True)
        return

    # Train
    train_with_20gb_limit(
        config_path=args.config,
        size=args.size,
        device=args.device,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
    )

    if args.cleanup:
        cleanup_checkpoints(keep_final=True)


if __name__ == "__main__":
    main()
