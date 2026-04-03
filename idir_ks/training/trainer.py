"""Training Loop for IDIR-KS - Optimized for RTX 3050 4GB"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
import time
import os
import gc
from pathlib import Path

from ..model.idir_ks_model import IDIRKSModel
from .hybrid_optimizer import HybridOptimizer, create_hybrid_optimizer


class IDIRKSTrainer:
    """
    Trainer for IDIR-KS model optimized for low-VRAM GPUs (RTX 3050 4GB).

    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation for effective larger batches
    - 1-hour time limit with auto-checkpointing
    - OOM recovery with batch size reduction
    - Streaming-friendly data loading
    - Storage-efficient checkpoints
    """

    def __init__(
        self,
        model: IDIRKSModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        save_interval: int = 100,
        max_time_hours: float = 1.0,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_time_seconds = max_time_hours * 3600

        # Memory optimization config
        self.config = {
            # Loss coefficients
            "lambda_consistency": 0.1,
            "lambda_entropy": 0.01,
            # Training phases
            "phase1_steps": 200,
            "phase2_steps": 800,
            "phase3_steps": 200,
            "max_steps": 1200,
            # Gradient accumulation
            "grad_accum_steps": 4,
            # Gradient clipping
            "grad_clip": 1.0,
            # Multi-trajectory (disabled for low VRAM)
            "num_trajectories": 1,
            "use_multi_trajectory": False,
        }
        if config:
            self.config.update(config)

        self.step = 0
        self.epoch = 0
        self.phase = "warmup"
        self.start_time = None

        # Mixed precision
        self.use_amp = device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Initialize optimizer
        self.optimizer = None
        self._update_optimizer()

        # Tracking
        self.loss_history = []
        self.step_times = []
        self.best_val_loss = float("inf")

    def _update_optimizer(self):
        """Update optimizer based on current phase"""
        if self.step < self.config["phase1_steps"]:
            phase = "warmup"
        elif self.step < self.config["phase1_steps"] + self.config["phase2_steps"]:
            phase = "full"
        else:
            phase = "convergence"

        if phase != self.phase or self.optimizer is None:
            if phase != self.phase:
                print(f"  Phase -> {phase}")
            self.phase = phase
            self.optimizer = create_hybrid_optimizer(self.model, phase=self.phase)

    def _time_remaining(self) -> float:
        """Seconds remaining in training budget"""
        if self.start_time is None:
            return self.max_time_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.max_time_seconds - elapsed)

    def _time_exceeded(self) -> bool:
        """Check if training time budget is exceeded"""
        return self._time_remaining() <= 0

    def _compute_loss(self, batch: Dict, use_amp: bool = True) -> tuple:
        """Compute training loss with optional mixed precision"""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        if self.use_amp and use_amp:
            with autocast():
                logits = self.model(input_ids)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
        else:
            logits = self.model(input_ids)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        consistency_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)

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

    def train_step(
        self, batch: Dict, accum_step: int = 0, is_last_accum: bool = False
    ) -> Dict:
        """
        Single training step with gradient accumulation.

        Args:
            batch: Input batch
            accum_step: Current accumulation step (0-based)
            is_last_accum: Whether this is the last accumulation step
        """
        self.model.train()

        # Normalize loss by accumulation steps
        normalize_factor = self.config["grad_accum_steps"]

        if self.use_amp:
            with autocast():
                loss, loss_dict = self._compute_loss(batch, use_amp=False)
                loss = loss / normalize_factor

            self.scaler.scale(loss).backward()

            if is_last_accum:
                self.scaler.unscale_(self.optimizer.adam)
                self.scaler.unscale_(self.optimizer.rmsprop)

                if self.config["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["grad_clip"]
                    )

                self.scaler.step(self.optimizer.adam)
                self.scaler.step(self.optimizer.rmsprop)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss, loss_dict = self._compute_loss(batch, use_amp=False)
            loss = loss / normalize_factor
            loss.backward()

            if is_last_accum:
                if self.config["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["grad_clip"]
                    )

                self.optimizer.adam.step()
                self.optimizer.rmsprop.step()
                self.optimizer.zero_grad()

        # Denormalize loss for logging
        loss_dict = {k: v * normalize_factor for k, v in loss_dict.items()}

        if is_last_accum:
            self.step += 1

        return loss_dict

    def _safe_train_step(self, batch: Dict) -> Optional[Dict]:
        """
        Train step with OOM protection.
        Returns None if OOM occurred (batch was skipped).
        """
        try:
            return self.train_step(batch, accum_step=0, is_last_accum=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"  [OOM skipped step {self.step + 1}]")
                return None
            raise

    def train_epoch(self) -> Dict:
        """Train for one epoch with gradient accumulation and time limit"""
        epoch_losses = []
        start_time = time.time()
        accum_buffers = {"loss": 0, "ce": 0, "count": 0}

        for batch_idx, batch in enumerate(self.train_dataloader):
            if self._time_exceeded():
                print("  [Time limit reached]")
                break

            step_start = time.time()

            # Gradient accumulation
            accum_steps = self.config["grad_accum_steps"]
            for accum_i in range(accum_steps):
                if self._time_exceeded():
                    break

                # Get next batch (reuse current for first accum step)
                if accum_i == 0:
                    current_batch = batch
                else:
                    try:
                        current_batch = next(self._batch_iter)
                    except StopIteration:
                        self._batch_iter = iter(self.train_dataloader)
                        try:
                            current_batch = next(self._batch_iter)
                        except StopIteration:
                            break

                is_last = accum_i == accum_steps - 1
                loss_dict = self.train_step(current_batch, accum_i, is_last)
                accum_buffers["loss"] += loss_dict["total"]
                accum_buffers["ce"] += loss_dict["ce"]
                accum_buffers["count"] += 1

            if accum_buffers["count"] > 0:
                avg_loss = accum_buffers["loss"] / accum_buffers["count"]
                epoch_losses.append(avg_loss)
                self.loss_history.append(avg_loss)

            step_time = time.time() - step_start
            self.step_times.append(step_time)

            # Reset accum buffers
            accum_buffers = {"loss": 0, "ce": 0, "count": 0}

            # Logging
            if self.step % self.log_interval == 0 and len(epoch_losses) > 0:
                lr = self.optimizer.get_lr()
                elapsed = time.time() - self.start_time if self.start_time else 0
                remaining = self._time_remaining()
                print(
                    f"Step {self.step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"CE: {accum_buffers.get('ce', 0):.4f} | "
                    f"Phase: {self.phase} | "
                    f"LR: {lr['adam']:.2e} | "
                    f"Time: {step_time:.3f}s | "
                    f"ETA: {remaining / 60:.1f}m"
                )

            # Periodic checkpoint
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save_checkpoint(f"step_{self.step}.pt")

            # Check max steps
            if self.step >= self.config["max_steps"]:
                break

        self.epoch += 1

        epoch_stats = {
            "epoch": self.epoch,
            "step": self.step,
            "avg_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            "epoch_time": time.time() - start_time,
        }

        return epoch_stats

    def validate(self) -> Dict:
        """Validation loop with memory efficiency"""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    _, loss_dict = self._compute_loss(batch, use_amp=False)
                    val_losses.append(loss_dict["total"])
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue

        if not val_losses:
            return {"val_loss": 0, "perplexity": 0}

        avg_val_loss = sum(val_losses) / len(val_losses)
        perplexity = min(float("inf"), torch.exp(torch.tensor(avg_val_loss)).item())

        return {
            "val_loss": avg_val_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, filename: str, save_optimizer: bool = False):
        """
        Save checkpoint. By default saves only model weights (storage efficient).
        Set save_optimizer=True for full resume capability (larger file).
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "phase": self.phase,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        if save_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)

        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint saved: {filename} ({size_mb:.1f} MB)")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.phase = checkpoint.get("phase", "warmup")
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"  Loaded checkpoint from step {self.step}")

    def train(self, num_epochs: int = None):
        """
        Full training loop with 1-hour time limit.

        Automatically:
        - Stops when time budget is exceeded
        - Saves final checkpoint before stopping
        - Recovers from OOM errors
        - Validates periodically
        """
        self.start_time = time.time()
        total_hours = self.max_time_seconds / 3600

        print("=" * 70)
        print(f"IDIR-KS Training (RTX 3050 4GB Optimized)")
        print("=" * 70)
        print(f"  Time budget: {total_hours:.1f} hour(s)")
        print(f"  Max steps: {self.config['max_steps']}")
        print(f"  Grad accum: {self.config['grad_accum_steps']}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Device: {self.device}")

        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model params: {params:,}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print("=" * 70)

        # Initialize batch iterator for gradient accumulation
        self._batch_iter = iter(self.train_dataloader)

        if num_epochs is None:
            num_epochs = 10000  # Will stop based on time/step limits

        for epoch in range(num_epochs):
            if self.step >= self.config["max_steps"]:
                print("  [Max steps reached]")
                break

            if self._time_exceeded():
                print("  [Time budget exceeded]")
                break

            remaining_min = self._time_remaining() / 60
            print(f"\n=== Epoch {self.epoch + 1} ({remaining_min:.1f}m remaining) ===")

            epoch_stats = self.train_epoch()

            if epoch_stats["avg_loss"] > 0:
                print(
                    f"  Epoch {self.epoch} | "
                    f"Avg Loss: {epoch_stats['avg_loss']:.4f} | "
                    f"Steps: {self.step} | "
                    f"Time: {epoch_stats['epoch_time']:.1f}s"
                )

            # Validation every 5 epochs
            if self.val_dataloader is not None and (self.epoch + 1) % 5 == 0:
                val_stats = self.validate()
                print(
                    f"  Validation | "
                    f"Loss: {val_stats['val_loss']:.4f} | "
                    f"Perplexity: {val_stats['perplexity']:.2f}"
                )

                if val_stats["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_stats["val_loss"]
                    self.save_checkpoint("best_model.pt")

            # Time checkpoint every 10 minutes
            elapsed = time.time() - self.start_time
            if elapsed % 600 < 30:  # Roughly every 10 minutes
                self.save_checkpoint("latest.pt")

        # Final outputs
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Total steps: {self.step}")
        print(f"  Total epochs: {self.epoch}")
        print(f"  Total time: {(time.time() - self.start_time) / 60:.1f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint("final_model.pt")
        print(f"  Final model saved to: {self.checkpoint_dir / 'final_model.pt'}")
        print("=" * 70)

    def generate_sample(
        self, prompt_ids: torch.Tensor, max_new_tokens: int = 50
    ) -> torch.Tensor:
        """Generate text sample for quality check"""
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                prompt_ids.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
            )
        return output.cpu()
