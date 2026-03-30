"""Ablation Study Variants for IDIR-KS"""

import torch
import torch.nn as nn
from typing import Dict
from ..model.idir_ks_model import IDIRKSModel, create_idir_ks_base


def create_ablation_variant(variant: str, **kwargs) -> IDIRKSModel:
    """
    Create an ablation variant of IDIR-KS.

    Ablation Variants (from paper):

    (A) Full Model (IDIR-KS + Hybrid Optimizer)
    (B) No Implicit Solver → replace with single pass
    (C) No Memory Module → remove Memory(h, M)
    (D) No MoE → replace with dense MLP
    (E) No Factorization → use full dense matrices
    (F) No Multi-Trajectory Inference → single trajectory
    (G) No Adaptive Compute → fixed solver steps
    (H) Adam Only Optimization → remove RMSProp
    (I) RMSProp Only Optimization → remove AdamW

    Args:
        variant: One of ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'full']
        **kwargs: Additional model configuration

    Returns:
        IDIRKSModel configured for the ablation
    """

    # Base configuration
    config = {
        "use_implicit_solver": True,
        "use_memory": True,
        "use_moe": True,
        "use_factorization": True,
        "num_trajectories": 1,
        "enable_adaptive": True,
        "hybrid_optimizer": True,
    }

    # Apply ablation
    if variant == "A" or variant == "full":
        # Full model - no changes
        pass

    elif variant == "B":
        # No Implicit Solver
        config["use_implicit_solver"] = False

    elif variant == "C":
        # No Memory Module
        config["use_memory"] = False

    elif variant == "D":
        # No MoE
        config["use_moe"] = False

    elif variant == "E":
        # No Factorization
        config["use_factorization"] = False

    elif variant == "F":
        # No Multi-Trajectory Inference
        config["num_trajectories"] = 1
        config["use_multi_trajectory"] = False

    elif variant == "G":
        # No Adaptive Compute
        config["enable_adaptive"] = False

    elif variant == "H":
        # Adam Only Optimization
        config["hybrid_optimizer"] = False
        config["use_adam_only"] = True

    elif variant == "I":
        # RMSProp Only Optimization
        config["hybrid_optimizer"] = False
        config["use_rmsprop_only"] = True

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    config.update(kwargs)

    return create_idir_ks_base(**config)


class AblationStudy:
    """
    Manages ablation study experiments.
    """

    VARIANTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    VARIANT_DESCRIPTIONS = {
        "A": "Full Model (IDIR-KS + Hybrid Optimizer)",
        "B": "No Implicit Solver → replace with single pass",
        "C": "No Memory Module → remove Memory(h, M)",
        "D": "No MoE → replace with dense MLP",
        "E": "No Factorization → use full dense matrices",
        "F": "No Multi-Trajectory Inference → single trajectory",
        "G": "No Adaptive Compute → fixed solver steps",
        "H": "Adam Only Optimization → remove RMSProp",
        "I": "RMSProp Only Optimization → remove AdamW",
    }

    def __init__(self, base_config: Dict = None):
        self.base_config = base_config or {}
        self.results = {}

    def run_ablation(self, variant: str, trainer, evaluator, **kwargs) -> Dict:
        """
        Run a single ablation experiment.

        Args:
            variant: Variant letter ('A' through 'I')
            trainer: Training function
            evaluator: Evaluation function
            **kwargs: Additional arguments

        Returns:
            Results dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"Running Ablation {variant}: {self.VARIANT_DESCRIPTIONS[variant]}")
        print(f"{'=' * 60}\n")

        # Create model
        model = create_ablation_variant(variant, **self.base_config)

        # Train
        train_results = trainer(model, **kwargs)

        # Evaluate
        eval_results = evaluator(model, **kwargs)

        results = {
            "variant": variant,
            "description": self.VARIANT_DESCRIPTIONS[variant],
            "train": train_results,
            "eval": eval_results,
        }

        self.results[variant] = results

        return results

    def run_all(self, trainer, evaluator, **kwargs) -> Dict:
        """
        Run all ablation experiments.

        Args:
            trainer: Training function
            evaluator: Evaluation function
            **kwargs: Additional arguments

        Returns:
            Dictionary of all results
        """
        for variant in self.VARIANTS:
            self.run_ablation(variant, trainer, evaluator, **kwargs)

        return self.results

    def compare_results(self) -> Dict:
        """
        Compare results across all ablations.

        Returns:
            Comparison dictionary
        """
        if not self.results:
            return {}

        # Get baseline (Full Model)
        baseline = self.results.get("A", {})

        comparison = {}
        for variant, results in self.results.items():
            if variant == "A":
                continue

            # Compare metrics
            comparison[variant] = {}

            # Expected observations from paper:
            if variant == "B":
                comparison[variant]["expected"] = "Significant drop in reasoning depth"
            elif variant == "C":
                comparison[variant]["expected"] = "Weaker generalization"
            elif variant == "D":
                comparison[variant]["expected"] = "Reduced specialization"
            elif variant == "E":
                comparison[variant]["expected"] = "Increased redundancy"
            elif variant == "F":
                comparison[variant]["expected"] = "Lower robustness"
            elif variant == "G":
                comparison[variant]["expected"] = "Inefficient inference"
            elif variant == "H":
                comparison[variant]["expected"] = "Unstable routing and memory"
            elif variant == "I":
                comparison[variant]["expected"] = "Slower convergence"

        return comparison

    def print_summary(self):
        """Print summary of ablation results"""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)

        for variant, results in self.results.items():
            print(f"\nVariant {variant}: {results['description']}")
            print(f"  Train Loss: {results['train'].get('final_loss', 'N/A'):.4f}")
            print(f"  Perplexity: {results['eval'].get('perplexity', 'N/A'):.2f}")
            print(f"  GSM8K Acc: {results['eval'].get('gsm8k_accuracy', 'N/A'):.4f}")
            print(f"  MBPP Pass@k: {results['eval'].get('mbpp_pass_at_k', 'N/A'):.4f}")

        print("\n" + "=" * 80)


def run_quick_ablation_test(variant: str = "A", device: str = "cpu") -> IDIRKSModel:
    """
    Quick test of an ablation variant.

    Args:
        variant: Ablation variant to test
        device: Device to use

    Returns:
        Created model
    """
    model = create_ablation_variant(variant)
    model = model.to(device)

    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Variant {variant} test passed!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model
