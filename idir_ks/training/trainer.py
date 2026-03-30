"""Training Loop for IDIR-KS"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
import os
from pathlib import Path

from ..model.idir_ks_model import IDIRKSModel
from .hybrid_optimizer import HybridOptimizer, create_hybrid_optimizer


class IDIRKSTrainer:
    """
    Trainer for IDIR-KS model with multi-phase training schedule.
    """

    def __init__(
        self,
        model: IDIRKSModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        save_interval: int = 1000,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval

        # Default config
        self.config = {
            # Loss coefficients
            "lambda_consistency": 0.1,
            "lambda_entropy": 0.01,
            # Training phases
            "phase1_steps": 5000,
            "phase2_steps": 45000,
            "phase3_steps": 5000,
            "max_steps": 55000,
            # Gradient clipping
            "grad_clip": 1.0,
            # Multi-trajectory
            "num_trajectories": 1,
            "use_multi_trajectory": False,
        }
        if config:
            self.config.update(config)

        self.step = 0
        self.epoch = 0
        self.phase = "warmup"

        # Initialize optimizer
        self.optimizer = None
        self._update_optimizer()

        # Loss tracking
        self.loss_history = []
        self.step_times = []

    def _update_optimizer(self):
        """Update optimizer based on current phase"""
        if self.step < self.config["phase1_steps"]:
            phase = "warmup"
        elif self.step < self.config["phase1_steps"] + self.config["phase2_steps"]:
            phase = "full"
        else:
            phase = "convergence"

        if phase != self.phase:
            print(f"Switching to phase: {phase}")
            self.phase = phase
            self.optimizer = create_hybrid_optimizer(self.model, phase=self.phase)

    def _compute_loss(self, batch: Dict) -> tuple:
        """
        Compute training loss.

        L = CE + λ1 * consistency_loss + λ2 * entropy_regularization
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        if self.config["use_multi_trajectory"] and self.config["num_trajectories"] > 1:
            logits, trajectory_logits = self.model.forward_multi_trajectory(
                input_ids, num_trajectories=self.config["num_trajectories"]
            )

            # Cross-entropy loss on consistent output
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

            # Consistency loss: encourage trajectories to agree
            consistency_loss = self._compute_consistency_loss(trajectory_logits, labels)

            # Entropy regularization
            entropy_loss = self._compute_entropy_regularization(trajectory_logits)
        else:
            logits = self.model(input_ids)

            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

            consistency_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)

        # Combine losses
        total_loss = (
            ce_loss
            + self.config["lambda_consistency"] * consistency_loss
            + self.config["lambda_entropy"] * entropy_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "ce": ce_loss.item(),
            "consistency": consistency_loss.item(),
            "entropy": entropy_loss.item(),
        }

        return total_loss, loss_dict

    def _compute_consistency_loss(
        self, trajectory_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency loss: trajectories should agree on correct answers.

        Args:
            trajectory_logits: (num_traj, batch, seq, vocab)
            labels: (batch, seq)
        """
        num_traj = trajectory_logits.size(0)

        # Get predictions for each trajectory
        preds = trajectory_logits.argmax(dim=-1)  # (num_traj, batch, seq)

        # Count agreement with labels
        correct_mask = preds == labels.unsqueeze(0)  # (num_traj, batch, seq)

        # Variance in correctness: want all trajectories to be correct
        consistency = correct_mask.float().mean(dim=0)  # (batch, seq)

        # Penalize inconsistency (low agreement)
        loss = 1.0 - consistency.mean()

        return loss

    def _compute_entropy_regularization(
        self, trajectory_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularization: encourage diverse but confident predictions.

        Args:
            trajectory_logits: (num_traj, batch, seq, vocab)
        """
        # Compute entropy for each trajectory
        probs = F.softmax(trajectory_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(
            dim=-1
        )  # (num_traj, batch, seq)

        # Average entropy across trajectories
        avg_entropy = entropy.mean()

        # We want moderate entropy (not too peaked, not too flat)
        target_entropy = 2.0  # Adjust based on vocab size
        loss = F.mse_loss(
            avg_entropy, torch.tensor(target_entropy, device=entropy.device)
        )

        return loss

    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        self.model.train()

        # Check phase transition
        self._update_optimizer()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward and compute loss
        loss, loss_dict = self._compute_loss(batch)

        # Backward
        loss.backward()

        # Gradient clipping
        if self.config["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )

        # Optimizer step
        self.optimizer.step()

        self.step += 1

        return loss_dict

    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        epoch_losses = []
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            step_start = time.time()

            loss_dict = self.train_step(batch)
            epoch_losses.append(loss_dict["total"])

            step_time = time.time() - step_start
            self.step_times.append(step_time)

            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = sum(epoch_losses[-self.log_interval :]) / len(
                    epoch_losses[-self.log_interval :]
                )
                lr = self.optimizer.get_lr()
                print(
                    f"Step {self.step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"CE: {loss_dict['ce']:.4f} | "
                    f"Phase: {self.phase} | "
                    f"Adam LR: {lr['adam']:.2e} | "
                    f"RMSProp LR: {lr['rmsprop']:.2e} | "
                    f"Time: {step_time:.3f}s"
                )

            # Checkpointing
            if self.step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")

            # Check max steps
            if self.step >= self.config["max_steps"]:
                break

        self.epoch += 1

        epoch_stats = {
            "epoch": self.epoch,
            "step": self.step,
            "avg_loss": sum(epoch_losses) / len(epoch_losses),
            "epoch_time": time.time() - start_time,
        }

        return epoch_stats

    def validate(self) -> Dict:
        """Validation loop"""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss, loss_dict = self._compute_loss(batch)
                val_losses.append(loss_dict["total"])

        avg_val_loss = sum(val_losses) / len(val_losses)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

        return {
            "val_loss": avg_val_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "phase": self.phase,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.phase = checkpoint["phase"]
        print(f"Loaded checkpoint from step {self.step}")

    def train(self, num_epochs: int = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = 1000  # Large number, will stop based on max_steps

        print(f"Starting training. Target: {self.config['max_steps']} steps")

        for epoch in range(num_epochs):
            if self.step >= self.config["max_steps"]:
                break

            print(f"\n=== Epoch {self.epoch + 1} ===")
            epoch_stats = self.train_epoch()

            print(
                f"Epoch complete. "
                f"Avg Loss: {epoch_stats['avg_loss']:.4f} | "
                f"Time: {epoch_stats['epoch_time']:.2f}s"
            )

            # Validation
            if self.val_dataloader is not None:
                val_stats = self.validate()
                print(
                    f"Validation | "
                    f"Loss: {val_stats['val_loss']:.4f} | "
                    f"Perplexity: {val_stats['perplexity']:.2f}"
                )

        # Final checkpoint
        self.save_checkpoint("final_model.pt")
        print("Training complete!")
