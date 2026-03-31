#!/usr/bin/env python3
"""
Normal Training Script for IDIR-KS
Location: IDIR/train.py

This is the standard training script following the README.md structure.
All checkpoints and logs are saved to the 'checkpoints/' subdirectory by default.
"""

import sys
import os
import argparse

# Add idir_ks to path
sys.path.insert(0, os.path.dirname(__file__))

from idir_ks.main import train_model


def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description="IDIR-KS: Normal Training Script")

    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 80)
    print("IDIR-KS: Normal Training")
    print("=" * 80)
    print(f"Model size: {args.size}")
    print(f"Device: {args.device}")
    print(f"Config: {args.config if args.config else 'default'}")
    print(f"Checkpoints saved to: checkpoints/")
    print("=" * 80)

    # Train using the main module
    train_model(args)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("Checkpoints saved in: checkpoints/")
    print("=" * 80)


if __name__ == "__main__":
    main()
