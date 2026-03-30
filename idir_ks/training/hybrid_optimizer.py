"""Hybrid Optimizer: AdamW + RMSProp for IDIR-KS"""

import torch
from torch.optim import Optimizer
from typing import Dict, List, Iterable, Callable
import math


class HybridOptimizer:
    """
    Hybrid optimization strategy combining AdamW and RMSProp.

    AdamW parameters:
        - core implicit layers
        - factorized transformations
        - output layers

    RMSProp parameters:
        - memory modules
        - MoE routing networks

    This stabilizes training across heterogeneous components.
    """

    def __init__(
        self,
        model_parameters: Iterable,
        adam_lr: float = 2e-4,
        adam_betas: tuple = (0.9, 0.95),
        adam_weight_decay: float = 0.01,
        rmsprop_lr: float = 1e-4,
        rmsprop_alpha: float = 0.99,
        rmsprop_momentum: float = 0.0,
        rmsprop_weight_decay: float = 0.0,
        eps: float = 1e-8,
        adam_param_groups: List[str] = None,
        rmsprop_param_groups: List[str] = None,
    ):
        """
        Args:
            model_parameters: Iterable of (name, param) tuples or param groups
            adam_lr: Learning rate for AdamW
            adam_betas: Betas for AdamW
            adam_weight_decay: Weight decay for AdamW
            rmsprop_lr: Learning rate for RMSProp
            rmsprop_alpha: Smoothing constant for RMSProp
            rmsprop_momentum: Momentum for RMSProp
            rmsprop_weight_decay: Weight decay for RMSProp
            eps: Epsilon for numerical stability
            adam_param_groups: List of parameter name patterns for AdamW
            rmsprop_param_groups: List of parameter name patterns for RMSProp
        """
        self.eps = eps

        # Default parameter group patterns
        if adam_param_groups is None:
            adam_param_groups = [
                "idir",  # Core implicit layers
                "factorized",  # Factorized transformations
                "embedding",  # Embedding layers
                "output_proj",  # Output projection
                "token_embed",  # Token embeddings
                "pos_embed",  # Positional embeddings
            ]

        if rmsprop_param_groups is None:
            rmsprop_param_groups = [
                "memory",  # Memory modules
                "router",  # MoE routing networks
                "gate",  # Gating mechanisms
                "compressor",  # Compression networks
            ]

        self.adam_patterns = adam_param_groups
        self.rmsprop_patterns = rmsprop_param_groups

        # Separate parameters
        adam_params = []
        rmsprop_params = []

        for name, param in model_parameters:
            if not param.requires_grad:
                continue

            # Check which optimizer this parameter belongs to
            is_adam = any(pattern in name for pattern in self.adam_patterns)
            is_rmsprop = any(pattern in name for pattern in self.rmsprop_patterns)

            if is_rmsprop and not is_adam:
                rmsprop_params.append(param)
            else:
                # Default to AdamW for everything else
                adam_params.append(param)

        # Create optimizers
        self.adam = AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            weight_decay=adam_weight_decay,
            eps=eps,
        )

        self.rmsprop = RMSProp(
            rmsprop_params,
            lr=rmsprop_lr,
            alpha=rmsprop_alpha,
            momentum=rmsprop_momentum,
            weight_decay=rmsprop_weight_decay,
            eps=eps,
        )

        self._param_counts = {"adam": len(adam_params), "rmsprop": len(rmsprop_params)}

    def zero_grad(self):
        """Zero gradients for both optimizers"""
        self.adam.zero_grad()
        self.rmsprop.zero_grad()

    def step(self, closure: Callable = None):
        """Perform optimization step for both optimizers"""
        adam_loss = self.adam.step(closure)
        rmsprop_loss = self.rmsprop.step(closure)
        return adam_loss, rmsprop_loss

    def state_dict(self) -> Dict:
        """Return state dict for both optimizers"""
        return {
            "adam": self.adam.state_dict(),
            "rmsprop": self.rmsprop.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict for both optimizers"""
        self.adam.load_state_dict(state_dict["adam"])
        self.rmsprop.load_state_dict(state_dict["rmsprop"])

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {
            "adam": self.adam.param_groups[0]["lr"],
            "rmsprop": self.rmsprop.param_groups[0]["lr"],
        }

    def set_lr(self, adam_lr: float = None, rmsprop_lr: float = None):
        """Set learning rates"""
        if adam_lr is not None:
            for param_group in self.adam.param_groups:
                param_group["lr"] = adam_lr
        if rmsprop_lr is not None:
            for param_group in self.rmsprop.param_groups:
                param_group["lr"] = rmsprop_lr

    def get_stats(self) -> Dict:
        """Get optimizer statistics"""
        adam_lr = self.adam.param_groups[0]["lr"]
        rmsprop_lr = self.rmsprop.param_groups[0]["lr"]

        return {
            "adam_lr": adam_lr,
            "rmsprop_lr": rmsprop_lr,
            "adam_params": self._param_counts["adam"],
            "rmsprop_params": self._param_counts["rmsprop"],
        }


class AdamW(Optimizer):
    """
    AdamW optimizer (decoupled weight decay).
    Implementation based on PyTorch AdamW.
    """

    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                # Update
                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    Better for non-stationary objectives like routing and memory.
    """

    def __init__(
        self, params, lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum
        )
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSProp does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Update running average of squared gradients
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Compute adaptive learning rate
                avg = square_avg.sqrt().add_(group["eps"])

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-group["lr"])
                else:
                    p.data.addcdiv_(grad, avg, value=-group["lr"])

        return loss


def create_hybrid_optimizer(
    model: torch.nn.Module, phase: str = "full", **kwargs
) -> HybridOptimizer:
    """
    Create hybrid optimizer with phase-specific configuration.

    Args:
        model: Model to optimize
        phase: 'warmup', 'full', or 'convergence'
        **kwargs: Additional optimizer arguments

    Returns:
        HybridOptimizer configured for the phase
    """
    # Get all named parameters
    model_params = [(name, param) for name, param in model.named_parameters()]

    # Phase-specific learning rates
    if phase == "warmup":
        # Phase 1: Adam only (stability warm-up)
        config = {
            "adam_lr": 1e-4,
            "rmsprop_lr": 0.0,  # Disable RMSProp initially
        }
    elif phase == "full":
        # Phase 2: Adam + RMSProp (full system)
        config = {
            "adam_lr": 2e-4,
            "rmsprop_lr": 1e-4,
        }
    elif phase == "convergence":
        # Phase 3: Reduced learning rates
        config = {
            "adam_lr": 5e-5,
            "rmsprop_lr": 2e-5,
        }
    else:
        raise ValueError(f"Unknown phase: {phase}")

    config.update(kwargs)

    return HybridOptimizer(model_params, **config)
