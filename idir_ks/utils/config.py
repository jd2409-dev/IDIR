"""Configuration Management for IDIR-KS"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    vocab_size: int = 50000
    dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    num_experts: int = 8
    expert_top_k: int = 2
    num_memories: int = 4096
    max_seq_len: int = 4096
    dropout: float = 0.1

    # IDIR specific
    convergence_threshold: float = 1e-4
    max_solver_steps: int = 12
    min_solver_steps: int = 4
    enable_adaptive: bool = True

    # Inference scaling
    num_trajectories: int = 1
    trajectory_noise: float = 0.0

    # Ablation flags (all True by default)
    use_implicit_solver: bool = True
    use_memory: bool = True
    use_moe: bool = True
    use_factorization: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    # AdamW settings
    adam_lr: float = 2e-4
    adam_betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    adam_weight_decay: float = 0.01

    # RMSProp settings
    rmsprop_lr: float = 1e-4
    rmsprop_alpha: float = 0.99
    rmsprop_momentum: float = 0.0
    rmsprop_weight_decay: float = 0.0

    # Parameter grouping patterns
    adam_param_groups: List[str] = field(
        default_factory=lambda: [
            "idir",
            "factorized",
            "embedding",
            "output_proj",
            "token_embed",
            "pos_embed",
        ]
    )
    rmsprop_param_groups: List[str] = field(
        default_factory=lambda: ["memory", "router", "gate", "compressor"]
    )

    eps: float = 1e-8


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Loss coefficients
    lambda_consistency: float = 0.1
    lambda_entropy: float = 0.01

    # Training phases (steps)
    phase1_steps: int = 5000
    phase2_steps: int = 45000
    phase3_steps: int = 5000
    max_steps: int = 55000

    # Gradient clipping
    grad_clip: float = 1.0

    # Multi-trajectory
    use_multi_trajectory: bool = False
    num_trajectories: int = 1

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class DataConfig:
    """Data configuration"""

    # Dataset composition
    code_weight: float = 0.40
    math_weight: float = 0.25
    logic_weight: float = 0.20
    language_weight: float = 0.15

    # Data paths
    code_path: Optional[str] = None
    math_path: Optional[str] = None
    logic_path: Optional[str] = None
    language_path: Optional[str] = None

    # Sequence settings
    max_length: int = 1024

    # Dataloader settings
    num_workers: int = 4
    pin_memory: bool = True

    # Total samples
    total_samples: int = 100000


@dataclass
class IDIRKSConfig:
    """Complete IDIR-KS configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Device
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "IDIRKSConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "IDIRKSConfig":
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "IDIRKSConfig":
        """Create configuration from dictionary"""
        model_config = ModelConfig(**config_dict.get("model", {}))
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        return cls(
            model=model_config,
            optimizer=optimizer_config,
            training=training_config,
            data=data_config,
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", 42),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "device": self.device,
            "seed": self.seed,
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations


def get_base_config() -> IDIRKSConfig:
    """Base IDIR-KS configuration"""
    return IDIRKSConfig()


def get_large_config() -> IDIRKSConfig:
    """Large IDIR-KS configuration"""
    config = IDIRKSConfig()
    config.model.dim = 1024
    config.model.num_layers = 12
    config.model.num_heads = 16
    config.model.num_experts = 16
    config.model.num_memories = 8192
    config.model.max_seq_len = 8192
    config.model.max_solver_steps = 16
    config.model.min_solver_steps = 6
    return config


def get_small_config() -> IDIRKSConfig:
    """Small IDIR-KS configuration for testing / low-VRAM GPUs"""
    config = IDIRKSConfig()
    config.model.vocab_size = 16384
    config.model.dim = 256
    config.model.num_layers = 2
    config.model.num_heads = 4
    config.model.num_experts = 4
    config.model.num_memories = 256
    config.model.max_seq_len = 128
    config.model.max_solver_steps = 4
    config.model.min_solver_steps = 2
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 8
    config.training.max_steps = 5000
    config.data.num_workers = 0
    config.data.pin_memory = False
    return config


def get_rtx3050_config() -> IDIRKSConfig:
    """
    RTX 3050 4GB optimized configuration.
    Designed to run for ~1 hour with no OOM or storage errors.

    Model: ~50M params (fits in 4GB with mixed precision)
    Vocab: 8192 (byte-level, no external tokenizer needed)
    Seq: 128 tokens (short but trains effectively)
    Batch: 4 with grad_accum=8 (effective batch 32)
    """
    config = IDIRKSConfig()

    # Tiny model that fits in 4GB
    config.model.vocab_size = 8192
    config.model.dim = 192
    config.model.num_layers = 4
    config.model.num_heads = 4
    config.model.num_experts = 4
    config.model.expert_top_k = 2
    config.model.num_memories = 512
    config.model.max_seq_len = 128
    config.model.dropout = 0.1
    config.model.convergence_threshold = 1e-3
    config.model.max_solver_steps = 4
    config.model.min_solver_steps = 2
    config.model.enable_adaptive = True
    config.model.num_trajectories = 1
    config.model.use_implicit_solver = True
    config.model.use_memory = True
    config.model.use_moe = True
    config.model.use_factorization = True

    # Data: short sequences for speed + memory
    config.data.max_length = 128
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.total_samples = 10000

    # Training: optimized for 1-hour RTX 3050 run
    config.training.batch_size = 4
    config.training.log_interval = 10
    config.training.save_interval = 100
    config.training.max_steps = 1200
    config.training.grad_clip = 1.0
    config.training.use_multi_trajectory = False
    config.training.num_trajectories = 1

    # Device
    config.device = "cuda"
    config.seed = 42

    return config


def get_ablation_config(variant: str) -> IDIRKSConfig:
    """Get configuration for ablation variant"""
    config = IDIRKSConfig()

    # Apply ablation settings
    if variant == "B":
        config.model.use_implicit_solver = False
    elif variant == "C":
        config.model.use_memory = False
    elif variant == "D":
        config.model.use_moe = False
    elif variant == "E":
        config.model.use_factorization = False
    elif variant == "F":
        config.model.num_trajectories = 1
    elif variant == "G":
        config.model.enable_adaptive = False
    elif variant == "H":
        # Adam only - will be handled in optimizer creation
        pass
    elif variant == "I":
        # RMSProp only - will be handled in optimizer creation
        pass

    return config
