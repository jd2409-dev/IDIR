import os
import random
import time
import json
import math
from contextlib import nullcontext
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from idir_model import IDIR

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("Install transformers: pip install transformers")
    raise

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token


def get_vram_config():
    """Detect GPU and return appropriate VRAM configuration."""
    if not torch.cuda.is_available():
        return "low"
    gpu_name = torch.cuda.get_device_name(0).lower()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if "3050" in gpu_name and total_memory >= 7:
        return "rtx_3050_8gb"
    elif "3050" in gpu_name:
        return "rtx_3050_4gb"
    elif (
        "3060" in gpu_name
        or "3070" in gpu_name
        or "3080" in gpu_name
        or "3090" in gpu_name
    ):
        return "high_end"
    elif total_memory >= 16:
        return "16gb"
    return "low"


VRAM_MODE = get_vram_config()
print(f"Detected VRAM mode: {VRAM_MODE}")

# Extended VRAM configurations for different GPU tiers
# Optimized for maximum coding capability training
VRAM_CONFIGS = {
    "high_end": {
        "batch_size": 256,
        "seq_len": 1024,
        "hidden_dim": 768,
        "expert_hidden_dim": 3072,
        "max_iterations": 20,
        "compile_model": True,
        "gradient_accumulation_steps": 1,
    },
    "16gb": {
        "batch_size": 192,
        "seq_len": 768,
        "hidden_dim": 640,
        "expert_hidden_dim": 2560,
        "max_iterations": 16,
        "compile_model": True,
        "gradient_accumulation_steps": 1,
    },
    "rtx_3050_8gb": {
        "batch_size": 96,  # Optimized for 8GB with mixed precision
        "seq_len": 512,
        "hidden_dim": 512,
        "expert_hidden_dim": 2048,
        "max_iterations": 12,
        "compile_model": True,
        "gradient_accumulation_steps": 2,  # Effective batch size 192
    },
    "rtx_3050_4gb": {
        "batch_size": 48,
        "seq_len": 256,
        "hidden_dim": 384,
        "expert_hidden_dim": 1536,
        "max_iterations": 8,
        "compile_model": True,
        "gradient_accumulation_steps": 2,
    },
    "low": {
        "batch_size": 32,
        "seq_len": 128,
        "hidden_dim": 256,
        "expert_hidden_dim": 1024,
        "max_iterations": 6,
        "compile_model": False,
        "gradient_accumulation_steps": 2,
    },
}

base_vram_cfg = VRAM_CONFIGS[VRAM_MODE]

SAFE_BATCH_SIZES = {
    "high_end": 8,
    "16gb": 8,
    "rtx_3050_8gb": 8,
    "rtx_3050_4gb": 4,
    "low": 4,
}

safe_limit = SAFE_BATCH_SIZES.get(VRAM_MODE, 8)
clamped_batch_size = min(base_vram_cfg["batch_size"], safe_limit)
if clamped_batch_size != base_vram_cfg["batch_size"]:
    print(
        f"Clamping batch_size "
        f"{base_vram_cfg['batch_size']} -> {clamped_batch_size} "
        f"for VRAM mode '{VRAM_MODE}'"
    )

vram_cfg = dict(base_vram_cfg)
vram_cfg["batch_size"] = clamped_batch_size

# Training time configuration - Aggressive optimization for < 1 hour on RTX 3050 6GB
TARGET_TRAINING_TIME_HOURS = 0.90  # Target 54 minutes to allow for overhead
TOTAL_STEPS_PHASE1 = 600  # Language pretraining - heavy code focus (reduced for time)
TOTAL_STEPS_PHASE2 = 300  # Algorithmic reasoning (reduced for time)
TOTAL_STEPS_PHASE3 = 400  # Distillation from code teacher (reduced for time)
TOTAL_STEPS_PHASE4 = 200  # Self-consistency on code (reduced for time)

# Phase rebalancing: More time on code-heavy phases
PHASE_TIME_ALLOCATION = {
    "phase_1": 0.30,  # Language pretraining with code emphasis (reduced)
    "phase_2": 0.10,  # Algorithmic tasks (reduced)
    "phase_3": 0.40,  # Teacher distillation (code-focused) (increased)
    "phase_4": 0.20,  # Self-consistency (increased)
}
PHASE_TIME_BUDGETS = {
    phase: int(TARGET_TRAINING_TIME_HOURS * 3600 * ratio)
    for phase, ratio in PHASE_TIME_ALLOCATION.items()
}

# Extended high-quality dataset configuration - all ungated public datasets
# Focused on coding, math, and reasoning content.
LANGUAGE_DATASETS_CONFIG = [
    {"path": "codeparrot/github-code", "split": "train", "text_field": "code", "weight": 0.20},
    {"path": "code_search_net", "name": "python", "split": "train", "text_field": "code", "weight": 0.06},
    {"path": "code_search_net", "name": "java", "split": "train", "text_field": "code", "weight": 0.05},
    {"path": "code_search_net", "name": "cpp", "split": "train", "text_field": "code", "weight": 0.04},
    {"path": "gsm8k", "split": "train", "text_field": "question", "weight": 0.05},
    {"path": "gsm8k", "split": "train", "text_field": "answer", "weight": 0.05},
    {"path": "math_dataset", "name": "main", "split": "train", "text_field": "question", "weight": 0.04},
    {"path": "math_dataset", "name": "main", "split": "train", "text_field": "solution", "weight": 0.04},
    {"path": "openwebtext", "split": "train", "text_field": "text", "weight": 0.06},
    {"path": "wikitext", "name": "wikitext-103-raw-v1", "split": "train", "text_field": "text", "weight": 0.06},
    {"path": "allenai/c4", "name": "en", "split": "train", "text_field": "text", "weight": 0.05},
]


# Extended high-quality synthetic word pool for rich text generation
# Enhanced with OS-level and systems programming terminology
REASONING_WORD_POOL = [
    # Core reasoning terms
    "reasoning",
    "inference",
    "deduction",
    "induction",
    "abduction",
    "logical",
    "conclusion",
    "premise",
    "argument",
    "validity",
    "syllogism",
    "proposition",
    "implication",
    "entailment",
    "proof",
    "verification",
    "validation",
    "justification",
    # Mathematical terms
    "arithmetic",
    "computation",
    "calculation",
    "algorithm",
    "equation",
    "function",
    "variable",
    "constant",
    "operation",
    "sequence",
    "series",
    "pattern",
    "structure",
    "relation",
    "theorem",
    "lemma",
    "corollary",
    "axiom",
    "postulate",
    "derivative",
    "integral",
    "probability",
    "statistics",
    # Scientific terms
    "hypothesis",
    "experiment",
    "evidence",
    "observation",
    "analysis",
    "synthesis",
    "evaluation",
    "interpretation",
    "theory",
    "model",
    "framework",
    "principle",
    "law",
    "methodology",
    "empirical",
    "phenomenon",
    "mechanism",
    # Cognitive terms
    "cognition",
    "perception",
    "memory",
    "learning",
    "understanding",
    "comprehension",
    "knowledge",
    "belief",
    "thought",
    "concept",
    "metacognition",
    "intuition",
    "insight",
    "awareness",
    "consciousness",
    # Process terms
    "iteration",
    "recursion",
    "convergence",
    "divergence",
    "transformation",
    "translation",
    "mapping",
    "correspondence",
    "abstraction",
    "generalization",
    "specialization",
    "optimization",
    "refinement",
    "progression",
    "regression",
    # System terms
    "system",
    "component",
    "module",
    "interface",
    "protocol",
    "architecture",
    "design",
    "implementation",
    "integration",
    "hierarchy",
    "structure",
    "organization",
    "coordination",
    # Context terms
    "context",
    "domain",
    "scope",
    "range",
    "field",
    "area",
    "environment",
    "setting",
    "situation",
    "circumstance",
    "perspective",
    "viewpoint",
    "standpoint",
    "framework",
    # Instruction/QA terms
    "instruction",
    "question",
    "answer",
    "query",
    "response",
    "explanation",
    "clarification",
    "elaboration",
    "example",
    "demonstration",
    "illustration",
    "guidance",
    "direction",
    # Code/programming terms
    "function",
    "variable",
    "loop",
    "condition",
    "algorithm",
    "implementation",
    "execution",
    "compilation",
    "runtime",
    "syntax",
    "semantics",
    "structure",
    "module",
    "class",
    # ========== OS-LEVEL SYSTEMS PROGRAMMING TERMS ==========
    # Operating System Core Concepts
    "kernel",
    "operating",
    "system",
    "scheduler",
    "process",
    "thread",
    "task",
    "interrupt",
    "handler",
    "trap",
    "exception",
    "syscall",
    "privilege",
    "ring",
    "mode",
    "supervisor",
    "userland",
    "kernelspace",
    "monolithic",
    "microkernel",
    "hybrid",
    "exokernel",
    "nanokernel",
    # Memory Management
    "memory",
    "allocation",
    "deallocation",
    "malloc",
    "free",
    "heap",
    "stack",
    "paging",
    "page",
    "frame",
    "segmentation",
    "virtual",
    "physical",
    "address",
    "translation",
    "MMU",
    "TLB",
    "cache",
    "buffer",
    "swap",
    "paging",
    "demand",
    "copy-on-write",
    "fork",
    # Process Management
    "process",
    "thread",
    "concurrency",
    "parallelism",
    "synchronization",
    "mutex",
    "semaphore",
    "spinlock",
    "deadlock",
    "livelock",
    "race",
    "condition",
    "barrier",
    "atomic",
    "critical",
    "section",
    "context",
    "switch",
    "preemption",
    "dispatch",
    "quantum",
    "timeslice",
    "priority",
    # File Systems
    "filesystem",
    "inode",
    "block",
    "superblock",
    "directory",
    "metadata",
    "journal",
    "journaling",
    "ext2",
    "ext3",
    "ext4",
    "btrfs",
    "zfs",
    "fat32",
    "ntfs",
    "mount",
    "unmount",
    "path",
    "inode",
    "dentry",
    # Device Drivers
    "driver",
    "device",
    "character",
    "block",
    "network",
    "pci",
    "usb",
    "serial",
    "parallel",
    "dma",
    "irq",
    "port",
    "register",
    "firmware",
    "hardware",
    "controller",
    "bus",
    "i2c",
    "spi",
    "uart",
    "gpio",
    # Networking Stack
    "network",
    "socket",
    "tcp",
    "udp",
    "ip",
    "ethernet",
    "packet",
    "frame",
    "datagram",
    "protocol",
    "stack",
    "layer",
    "transport",
    "network",
    "datalink",
    "physical",
    "routing",
    "switching",
    "bridge",
    "firewall",
    "nat",
    "port",
    "bind",
    "listen",
    "accept",
    "connect",
    # Boot and Initialization
    "boot",
    "bootloader",
    "grub",
    "uefi",
    "bios",
    "mbr",
    "gpt",
    "initrd",
    "initramfs",
    "kernel",
    "parameter",
    "command",
    "line",
    "startup",
    "initialization",
    "daemon",
    "service",
    "init",
    "systemd",
    # Assembly and Low-level
    "assembly",
    "assembler",
    "instruction",
    "opcode",
    "operand",
    "register",
    "accumulator",
    "stack",
    "pointer",
    "instruction",
    "counter",
    "flag",
    "carry",
    "zero",
    "overflow",
    "sign",
    "interrupt",
    "enable",
    "push",
    "pop",
    "call",
    "ret",
    "jump",
    "branch",
    "label",
    "section",
    "text",
    "data",
    "bss",
    "heap",
    "stack",
    "frame",
    "prologue",
    "epilogue",
    "calling",
    "convention",
    "abi",
    "elf",
    "binary",
    "object",
    "linker",
    "loader",
    # C Programming Systems
    "pointer",
    "dereference",
    "reference",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "sizeof",
    "offsetof",
    "struct",
    "union",
    "typedef",
    "enum",
    "volatile",
    "const",
    "restrict",
    "inline",
    "static",
    "extern",
    "header",
    "source",
    "preprocessor",
    "macro",
    "conditional",
    "compilation",
    "library",
    "shared",
    "static",
    "dynamic",
    "linking",
    "loading",
    "symbol",
    "export",
    # Security
    "security",
    "permission",
    "access",
    "control",
    "authentication",
    "authorization",
    "encryption",
    "decryption",
    "hash",
    "checksum",
    "signature",
    "certificate",
    "public",
    "private",
    "key",
    "cipher",
    "plaintext",
    "ciphertext",
    "nonce",
    "salt",
    "stretching",
    "bcrypt",
    "selinux",
    "apparmor",
    "capabilities",
    "chroot",
    "namespace",
    "cgroup",
    # Virtualization
    "virtualization",
    "hypervisor",
    "vm",
    "guest",
    "host",
    "kvm",
    "qemu",
    "xen",
    "vmware",
    "virtualbox",
    "container",
    "docker",
    "namespace",
    "cgroup",
    "control",
    "group",
    "isolation",
    "sandbox",
    "jail",
    # Debugging and Development
    "debugger",
    "gdb",
    "breakpoint",
    "watchpoint",
    "single",
    "step",
    "trace",
    "profiling",
    "profiler",
    "valgrind",
    "asan",
    "tsan",
    "sanitizer",
    "memory",
    "leak",
    "buffer",
    "overflow",
    "underflow",
]

# Training configuration - Aggressive settings for elite coding performance
TRAINING_CONFIG = {
    "device": device,
    "vocab_size": 50257,
    "hidden_dim": vram_cfg["hidden_dim"],
    "num_experts": 4,
    "expert_hidden_dim": vram_cfg["expert_hidden_dim"],
    "reasoning_steps": 3,
    "k_features": 32,
    "reasoning_hidden_dim": 2560,
    "max_iterations": vram_cfg["max_iterations"],
    "tolerance": 1e-4,
    "batch_size": vram_cfg["batch_size"],
    "seq_len": vram_cfg["seq_len"],
    "learning_rate": 5e-4,  # Increased for faster convergence
    "min_learning_rate": 1e-7,  # Lower for fine-tuning
    "aggressive_lr_multiplier": 3.0,  # More aggressive LR scaling
    "warmup_fraction": 0.03,  # Shorter warmup
    "phase1_steps": TOTAL_STEPS_PHASE1,
    "phase2_steps": TOTAL_STEPS_PHASE2,
    "phase3_steps": TOTAL_STEPS_PHASE3,
    "phase4_steps": TOTAL_STEPS_PHASE4,
    "save_every": 500,  # Less frequent saves for speed (reduced storage)
    "log_every": 75,
    "eval_every": 150,
    "max_grad_norm": 1.0,
    "use_mixed_precision": True,
    "mixed_precision_dtype": "bfloat16",
    "use_gradient_checkpointing": True,
    "compile_model": vram_cfg["compile_model"],
    "gradient_accumulation_steps": vram_cfg.get("gradient_accumulation_steps", 1),
    "checkpoint_dir": "checkpoints",
    "stream_buffer_size": 256,  # Keep a small replay buffer when streaming datasets
    "num_workers": 0,  # Use main process for data loading (more stable)
    "prefetch_factor": 4,  # Balanced for RTX 3050
    "phase_time_budgets": PHASE_TIME_BUDGETS,
    "data_expansion_factor": 20,  # Higher variety per batch
    "synthetic_text_ratio": 0.10,  # Favor real code heavily
    "synthetic_word_pool": REASONING_WORD_POOL,
    "language_datasets": LANGUAGE_DATASETS_CONFIG,
    "use_weighted_sampling": True,
    "validation_split": 0.003,  # Minimal validation

    "validation_batches": 5,
    "patience": 5,  # More patience for better convergence
    "min_delta": 0.0005,  # Tighter convergence threshold
    "code_focus": True,  # Enable code-specific optimizations
    "error_correction_training": True,  # Train on error correction
}

os.makedirs(TRAINING_CONFIG["checkpoint_dir"], exist_ok=True)


class StreamingDatasetSampler:
    """Lazy sampler that keeps a small buffer from a streaming dataset."""

    def __init__(self, dataset_iterable, buffer_size=256):
        self.dataset_iterable = dataset_iterable
        self.buffer_size = max(8, buffer_size)
        self.iterator = iter(self.dataset_iterable)
        self.buffer = []

    def _refill(self):
        attempts = 0
        while len(self.buffer) < self.buffer_size and attempts < self.buffer_size * 4:
            try:
                item = next(self.iterator)
            except StopIteration:
                try:
                    self.iterator = iter(self.dataset_iterable)
                except Exception:
                    break
                attempts += 1
                continue
            self.buffer.append(item)
            attempts += 1

    def sample(self):
        """Return a random example from the buffer, refilling if needed."""
        if not self.buffer:
            self._refill()
        if not self.buffer:
            raise RuntimeError("Could not sample from dataset buffer.")
        if len(self.buffer) < self.buffer_size:
            self._refill()
        return random.choice(self.buffer)


class ImprovedDataLoader:
    """Improved data loader with better sampling, caching, and validation."""

    def __init__(self, datasets, batch_size, seq_len, config, is_validation=False):
        self.datasets = datasets
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.expansion_factor = max(1, int(config.get("data_expansion_factor", 8)))
        self.synthetic_ratio = min(
            max(float(config.get("synthetic_text_ratio", 0.25)), 0.0), 0.5
        )
        self.synthetic_word_pool = (
            config.get("synthetic_word_pool") or REASONING_WORD_POOL
        )
        self.use_weighted_sampling = config.get("use_weighted_sampling", True)
        self.is_validation = is_validation

        # Build weights for weighted sampling
        if self.use_weighted_sampling and datasets:
            total_weight = sum(d.get("weight", 1.0) for d in datasets)
            self.weights = [d.get("weight", 1.0) / total_weight for d in datasets]
        else:
            self.weights = None

        # Statistics tracking
        self.stats = {
            "total_batches": 0,
            "synthetic_batches": 0,
            "dataset_batches": [0] * len(datasets) if datasets else [],
        }

    def _sample_dataset_index(self):
        """Sample dataset index using weighted sampling."""
        if self.weights:
            return random.choices(range(len(self.datasets)), weights=self.weights, k=1)[
                0
            ]
        return random.randint(0, len(self.datasets) - 1)

    def _extract_text(self, example, text_field):
        """Safely extract text from dataset example."""
        if isinstance(example, dict):
            text = example.get(text_field)
            if text is None and example:
                # Try common text fields
                for field in ["text", "content", "article", "code", "response"]:
                    if field in example:
                        text = example[field]
                        break
                if text is None:
                    text = next(iter(example.values()))
        else:
            text = str(example)

        text = "" if text is None else str(text)

        # Filter out very short or low-quality text
        if len(text.strip()) < 20:
            return None

        # Basic quality filtering - allow code patterns with braces, semicolons, comments
        has_prose_punctuation = (
            text.count(".") >= 1 or text.count("?") >= 1 or text.count("!") >= 1
        )
        has_code_indicators = (
            text.count(";") >= 1
            or text.count("{") >= 1
            or text.count("}") >= 1
            or text.count("(") >= 1
            or text.count(")") >= 1
            or text.count("//") >= 1
            or text.count("/*") >= 1
            or text.count("#include") >= 1
            or text.count("#define") >= 1
            or text.count("def ") >= 1
            or text.count("class ") >= 1
            or text.count("int ") >= 1
            or text.count("void ") >= 1
            or text.count("void*") >= 1
            or text.count("struct ") >= 1
        )
        if not has_prose_punctuation and not has_code_indicators:
            return None

        return text

    def _sample_text(self):
        """Sample text from datasets with quality filtering."""
        if not self.datasets:
            return "This is a placeholder sentence to avoid errors."

        max_attempts = 10
        for _ in range(max_attempts):
            idx = self._sample_dataset_index()
            source = self.datasets[idx]
            sampler = source.get("sampler")
            text_field = source["text_field"]

            try:
                if sampler is None:
                    raise RuntimeError("No sampler configured for dataset.")
                sample = sampler.sample()
                text = self._extract_text(sample, text_field)
                if text:
                    self.stats["dataset_batches"][idx] += 1
                    return text
            except Exception:
                continue

        return "Fallback text for data loading."

    def _make_synthetic_text(self):
        """Generate higher-quality synthetic text with diverse patterns."""
        # Extended pattern types for richer synthetic data including OS-level coding
        pattern = random.choice(
            [
                "reasoning_chain",
                "definition",
                "explanation",
                "comparison",
                "process",
                "instruction",
                "qa_format",
                "mathematical",
                "code_like",
                "deduction",
                "syscall_pattern",
                "driver_pattern",
                "memory_mgmt",
                "boot_sequence",
                "interrupt_handler",
                "humaneval_pattern",
                "c_function",
                "rust_function",
                "code_debug",
                "api_usage",
                "data_structure",
                "design_pattern",
                "error_handling",
                "concurrency",
            ]
        )

        if pattern == "reasoning_chain":
            # Create a simple reasoning chain
            num_steps = random.randint(2, 4)
            steps = []
            for i in range(num_steps):
                word = random.choice(self.synthetic_word_pool)
                num = random.randint(1, 100)
                steps.append(f"Step {i + 1}: The {word} value is {num}.")
            result = random.randint(1, 1000)
            text = " ".join(steps) + f" Therefore, the result is {result}."

        elif pattern == "definition":
            term = random.choice(self.synthetic_word_pool)
            aspects = random.sample(self.synthetic_word_pool, 2)
            connectors = [
                "involving",
                "related to",
                "connected with",
                "associated with",
                "derived from",
            ]
            text = f"{term} refers to a process {random.choice(connectors)} {aspects[0]} and {aspects[1]}."

        elif pattern == "explanation":
            concept = random.choice(self.synthetic_word_pool)
            detail = random.choice(self.synthetic_word_pool)
            starters = [
                "When analyzing",
                "In understanding",
                "To comprehend",
                "When examining",
            ]
            text = f"{random.choice(starters)} {concept}, we consider the {detail} aspect carefully."

        elif pattern == "comparison":
            terms = random.sample(self.synthetic_word_pool, 2)
            connectors = [
                "focuses on",
                "emphasizes",
                "prioritizes",
                "centers on",
                "deals with",
            ]
            c1, c2 = random.choice(connectors), random.choice(connectors)
            text = f"While {terms[0]} {c1} structure, {terms[1]} {c2} function."

        elif pattern == "process":
            words = random.sample(self.synthetic_word_pool, 3)
            starters = ["First", "Initially", "To begin", "Start by"]
            connectors = ["then", "next", "subsequently", "afterward"]
            endings = ["to complete", "to finish", "to conclude", "for completion"]
            text = f"{random.choice(starters)} apply {words[0]}, {random.choice(connectors)} integrate with {words[1]} using {words[2]} {random.choice(endings)}."

        elif pattern == "instruction":
            action = random.choice(
                ["Explain", "Describe", "Analyze", "Define", "Compare"]
            )
            term = random.choice(self.synthetic_word_pool)
            contexts = [
                "in detail",
                "briefly",
                "with examples",
                "step by step",
                "in your own words",
            ]
            text = f"Instruction: {action} the concept of {term} {random.choice(contexts)}."

        elif pattern == "qa_format":
            question_words = [
                "What is",
                "How does",
                "Why is",
                "When should",
                "Where does",
            ]
            term = random.choice(self.synthetic_word_pool)
            aspect = random.choice(self.synthetic_word_pool)
            q = f"{random.choice(question_words)} {term} related to {aspect}?"
            a = f"It involves a systematic approach connecting {term} with {aspect} principles."
            text = f"Question: {q} Answer: {a}"

        elif pattern == "mathematical":
            ops = ["+addition", "-subtraction", "×multiplication", "÷division"]
            a, b = random.randint(1, 100), random.randint(1, 100)
            op = random.choice(ops)
            symbol, name = op[0], op[1:]
            if name == "subtraction":
                a, b = max(a, b), min(a, b)
            result = {
                "addition": a + b,
                "subtraction": a - b,
                "multiplication": a * b,
                "division": a // max(b, 1),
            }[name]
            text = f"Calculate {a} {symbol} {b} using {name}. The answer is {result}."

        elif pattern == "code_like":
            func = random.choice(["function", "procedure", "method", "routine"])
            var = random.choice(self.synthetic_word_pool)
            action = random.choice(
                ["computes", "calculates", "determines", "evaluates"]
            )
            text = f"def {var}_{func}(): # This {func} {action} the {var} value."

        elif pattern == "deduction":
            premises = random.sample(self.synthetic_word_pool, 2)
            conclusion = random.choice(self.synthetic_word_pool)
            text = f"Given {premises[0]} and {premises[1]}, we deduce {conclusion}."

        elif pattern == "syscall_pattern":
            # Generate synthetic syscall examples
            syscalls = [
                "read",
                "write",
                "open",
                "close",
                "mmap",
                "fork",
                "exec",
                "wait",
                "exit",
                "ioctl",
            ]
            syscall = random.choice(syscalls)
            fd = random.randint(0, 255)
            size = random.choice([64, 128, 256, 512, 1024, 4096])
            text = f"// System call: {syscall}(fd={fd}, buf=buffer, count={size}); // Returns bytes processed or -1 on error."

        elif pattern == "driver_pattern":
            # Generate synthetic device driver code patterns
            devices = [
                "pci",
                "usb",
                "serial",
                "block",
                "char",
                "network",
                "gpio",
                "i2c",
                "spi",
            ]
            device = random.choice(devices)
            reg = random.randint(0x00, 0xFF)
            text = f"// {device}_driver: initialize device at register 0x{reg:02X}, set interrupt handler, enable DMA."

        elif pattern == "memory_mgmt":
            # Generate synthetic memory management patterns
            ops = [
                "malloc",
                "calloc",
                "realloc",
                "free",
                "mmap",
                "munmap",
                "brk",
                "sbrk",
            ]
            op = random.choice(ops)
            size = random.choice([64, 256, 1024, 4096, 16384, 65536])
            text = f"// Memory operation: {op}(size={size}); // Allocate virtual memory, update page tables, check limits."

        elif pattern == "boot_sequence":
            # Generate synthetic boot sequence steps
            stages = [
                "BIOS/UEFI",
                "bootloader",
                "kernel decompression",
                "initrd",
                "init",
                "user space",
            ]
            stage1, stage2 = random.sample(stages, 2)
            text = f"// Boot: {stage1} -> {stage2}: Load segments, setup stack, enable paging, jump to entry point."

        elif pattern == "interrupt_handler":
            # Generate synthetic interrupt handler patterns
            irqs = [
                "timer",
                "keyboard",
                "mouse",
                "disk",
                "network",
                "DMA",
                "system call",
            ]
            irq = random.choice(irqs)
            text = f"// IRQ handler: {irq}_interrupt() {{ save context; handle event; send EOI; restore context; iret; }}"

        elif pattern == "humaneval_pattern":
            # HumanEval-style function completion patterns
            funcs = [
                (
                    "def add(a: int, b: int) -> int:",
                    '    """Add two numbers."""',
                    "    return a + b",
                ),
                (
                    "def factorial(n: int) -> int:",
                    '    """Calculate factorial."""',
                    "    if n <= 1: return 1; return n * factorial(n-1)",
                ),
                (
                    "def is_palindrome(s: str) -> bool:",
                    '    """Check if string is palindrome."""',
                    "    return s == s[::-1]",
                ),
                (
                    "def fibonacci(n: int) -> int:",
                    '    """Return nth Fibonacci number."""',
                    "    if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)",
                ),
                (
                    "def reverse_list(lst: list) -> list:",
                    '    """Reverse a list."""',
                    "    return lst[::-1]",
                ),
                (
                    "def binary_search(arr: list, target: int) -> int:",
                    '    """Binary search implementation."""',
                    "    # Binary search algorithm",
                ),
                (
                    "def merge_sort(arr: list) -> list:",
                    '    """Merge sort implementation."""',
                    "    # Merge sort algorithm",
                ),
                (
                    "def quick_sort(arr: list) -> list:",
                    '    """Quick sort implementation."""',
                    "    # Quick sort algorithm",
                ),
            ]
            header, docstring, impl = random.choice(funcs)
            text = f"{header}\n{docstring}\n{impl}"

        elif pattern == "c_function":
            # C function implementation patterns
            funcs = [
                ("int add(int a, int b)", "// Add two integers", "    return a + b;"),
                (
                    "void swap(int *a, int *b)",
                    "// Swap two integers",
                    "    int temp = *a; *a = *b; *b = temp;",
                ),
                (
                    "size_t strlen(const char *s)",
                    "// Calculate string length",
                    "    size_t len = 0; while (*s++) len++; return len;",
                ),
                (
                    "void *memcpy(void *dest, const void *src, size_t n)",
                    "// Copy memory",
                    "    // Memory copy implementation",
                ),
                (
                    "int strcmp(const char *s1, const char *s2)",
                    "// Compare strings",
                    "    // String compare implementation",
                ),
                (
                    "void *malloc(size_t size)",
                    "// Allocate memory",
                    "    // Memory allocation",
                ),
                (
                    "void free(void *ptr)",
                    "// Free memory",
                    "    // Memory deallocation",
                ),
            ]
            sig, comment, impl = random.choice(funcs)
            text = f"{sig} {{\n{comment}\n{impl}\n}}"

        elif pattern == "rust_function":
            # Rust function patterns
            funcs = [
                ("fn add(a: i32, b: i32) -> i32", "// Add two numbers", "    a + b"),
                (
                    "fn factorial(n: u64) -> u64",
                    "// Calculate factorial",
                    "    if n <= 1 { 1 } else { n * factorial(n-1) }",
                ),
                ("fn is_even(n: i32) -> bool", "// Check if even", "    n % 2 == 0"),
                (
                    "fn swap<T>(a: &mut T, b: &mut T)",
                    "// Swap values",
                    "    std::mem::swap(a, b);",
                ),
                (
                    "fn binary_search(arr: &[i32], target: i32) -> Option<usize>",
                    "// Binary search",
                    "    // Implementation",
                ),
                (
                    "impl Drop for MyType",
                    "// Custom destructor",
                    "    fn drop(&mut self) { /* cleanup */ }",
                ),
            ]
            sig, comment, impl = random.choice(funcs)
            text = f"{sig} {{\n{comment}\n{impl}\n}}"

        elif pattern == "code_debug":
            # Code debugging/repair patterns
            bugs = [
                (
                    "def divide(a, b):\n    return a / b",
                    "# Bug: division by zero\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b",
                ),
                (
                    "for i in range(len(lst)):\n    print(lst[i])",
                    "# Better: iterate directly\nfor item in lst:\n    print(item)",
                ),
                ("if x == True:", "# Bug: boolean comparison\nif x:"),
                (
                    "file = open('data.txt')\ndata = file.read()",
                    "# Bug: file not closed\nwith open('data.txt') as file:\n    data = file.read()",
                ),
                ("lst = []\nlst.append(1)", "# Optimization: list literal\nlst = [1]"),
            ]
            buggy, fixed = random.choice(bugs)
            text = f"# Fix this code:\n{buggy}\n\n# Fixed version:\n{fixed}"

        elif pattern == "api_usage":
            # API usage examples
            apis = [
                (
                    "requests",
                    "import requests\nresponse = requests.get(url)\ndata = response.json()",
                ),
                (
                    "sqlite",
                    "import sqlite3\nconn = sqlite3.connect('db.db')\ncursor = conn.cursor()",
                ),
                (
                    "threading",
                    "import threading\nthread = threading.Thread(target=func)\nthread.start()",
                ),
                (
                    "asyncio",
                    "import asyncio\nasync def main():\n    await asyncio.sleep(1)",
                ),
                (
                    "json",
                    "import json\nwith open('data.json') as f:\n    data = json.load(f)",
                ),
                ("regex", "import re\nmatch = re.search(r'pattern', text)"),
            ]
            lib, code = random.choice(apis)
            text = f"# Using {lib} library:\n{code}"

        elif pattern == "data_structure":
            # Data structure implementations
            dss = [
                (
                    "class Stack:",
                    "def __init__(self): self.items = []\n    def push(self, item): self.items.append(item)\n    def pop(self): return self.items.pop()",
                ),
                (
                    "class Queue:",
                    "def __init__(self): self.items = []\n    def enqueue(self, item): self.items.append(item)\n    def dequeue(self): return self.items.pop(0)",
                ),
                (
                    "class Node:",
                    "def __init__(self, val): self.val = val; self.next = None",
                ),
                (
                    "class BinaryTree:",
                    "def __init__(self, val): self.val = val; self.left = None; self.right = None",
                ),
                (
                    "class LinkedList:",
                    "def __init__(self): self.head = None\n    def append(self, val): # append logic",
                ),
            ]
            header, impl = random.choice(dss)
            text = f"{header}\n{impl}"

        elif pattern == "design_pattern":
            # Design pattern examples
            patterns = [
                (
                    "# Singleton Pattern",
                    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
                ),
                (
                    "# Factory Pattern",
                    "class Factory:\n    @staticmethod\n    def create_product(type):\n        if type == 'A': return ProductA()\n        elif type == 'B': return ProductB()",
                ),
                (
                    "# Observer Pattern",
                    "class Subject:\n    def __init__(self): self.observers = []\n    def attach(self, observer): self.observers.append(observer)",
                ),
                (
                    "# Decorator Pattern",
                    "def decorator(func):\n    def wrapper(*args, **kwargs):\n        # pre-processing\n        result = func(*args, **kwargs)\n        # post-processing\n        return result\n    return wrapper",
                ),
            ]
            name, impl = random.choice(patterns)
            text = f"{name}\n{impl}"

        elif pattern == "error_handling":
            # Error handling patterns
            patterns = [
                (
                    "try-except",
                    "try:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f'Error: {e}')\nfinally:\n    cleanup()",
                ),
                (
                    "try-except-else",
                    "try:\n    result = operation()\nexcept Exception as e:\n    handle_error(e)\nelse:\n    process_result(result)",
                ),
                ("raise", "if not valid_input:\n    raise ValueError('Invalid input')"),
                ("custom exception", "class ValidationError(Exception):\n    pass"),
            ]
            name, code = random.choice(patterns)
            text = f"# {name} pattern:\n{code}"

        elif pattern == "concurrency":
            # Concurrency patterns
            patterns = [
                (
                    "Thread",
                    "import threading\ndef worker(): pass\nthread = threading.Thread(target=worker)\nthread.start()\nthread.join()",
                ),
                (
                    "Lock",
                    "import threading\nlock = threading.Lock()\nwith lock:\n    # critical section\n    pass",
                ),
                (
                    "Process",
                    "from multiprocessing import Process\np = Process(target=function)\np.start()\np.join()",
                ),
                (
                    "Pool",
                    "from multiprocessing import Pool\nwith Pool(4) as p:\n    results = p.map(func, items)",
                ),
                (
                    "async",
                    "import asyncio\nasync def task():\n    await asyncio.sleep(1)\nawait task()",
                ),
            ]
            name, code = random.choice(patterns)
            text = f"# {name} concurrency:\n{code}"

        return text

    def _generate_batch(self):
        """Generate a batch of tokenized sequences."""
        batch_text = []
        for _ in range(self.batch_size):
            fragments = []
            num_fragments = random.randint(1, self.expansion_factor)

            for _ in range(num_fragments):
                if self.datasets and random.random() > self.synthetic_ratio:
                    fragments.append(self._sample_text())
                else:
                    fragments.append(self._make_synthetic_text())
                    self.stats["synthetic_batches"] += 1

            text = " ".join(fragments).strip()
            batch_text.append(text or "This is a placeholder.")

        self.stats["total_batches"] += 1

        # Tokenize with proper handling
        encoded = tokenizer(
            batch_text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Create labels by shifting input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return input_ids, attention_mask, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self._generate_batch()

    def get_stats(self):
        """Return data loading statistics."""
        return self.stats.copy()


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Remove old checkpoints to save disk space, keeping only the most recent ones."""
    import glob
    import os

    try:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        # Sort by modification time (oldest first)
        checkpoints.sort(key=os.path.getmtime)
        # Remove old checkpoints, keeping only the last N
        for old_checkpoint in checkpoints[:-keep_last_n]:
            try:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
            except OSError as e:
                print(f"Warning: Could not remove {old_checkpoint}: {e}")
    except Exception as e:
        print(f"Warning: Error during checkpoint cleanup: {e}")


def generate_arithmetic_data(batch_size, seq_len, difficulty="mixed"):
    """Generate synthetic arithmetic examples with varying difficulty and formats."""
    prompts = []

    templates = [
        "Calculate {expr}. The answer is {result}.",
        "What is {expr}? The answer is {result}.",
        "Solve: {expr} = {result}",
        "Compute {expr}. Result: {result}",
        "Find the value of {expr}. Answer: {result}",
    ]

    for _ in range(batch_size):
        template = random.choice(templates)

        # Choose difficulty level
        if difficulty == "easy":
            num_range = (1, 50)
        elif difficulty == "hard":
            num_range = (1, 1000)
        else:  # mixed
            num_range = random.choice([(1, 50), (1, 100), (1, 500), (1, 1000)])

        a = random.randint(*num_range)
        b = random.randint(*num_range)
        op = random.randint(0, 7)  # Extended range for more variety

        if op == 0:  # Addition
            expr = f"{a} + {b}"
            result = a + b
        elif op == 1:  # Subtraction (ensure non-negative)
            bigger = max(a, b)
            smaller = min(a, b)
            expr = f"{bigger} - {smaller}"
            result = bigger - smaller
        elif op == 2:  # Multiplication
            a = min(a, 50)
            b = min(b, 50)
            expr = f"{a} × {b}"
            result = a * b
        elif op == 3:  # Division (ensure clean division)
            result = random.randint(1, 20)
            b = max(1, min(b, 20))
            a = result * b
            expr = f"{a} ÷ {b}"
        elif op == 4:  # Multi-step expression
            c = random.randint(1, 20)
            result = a + b - c
            expr = f"{a} + {b} - {c}"
        elif op == 5:  # Parentheses expression
            c = random.randint(1, 10)
            d = random.randint(1, 10)
            result = (a + b) * c - d
            expr = f"({a} + {b}) × {c} - {d}"
        elif op == 6:  # Sequential operations
            c = random.randint(1, 50)
            result = a + b + c
            expr = f"{a} + {b} + {c}"
        else:  # Comparison (not a calculation, but logical)
            result = "true" if a > b else "false"
            expr = f"{a} > {b}"

        prompt = template.format(expr=expr, result=result)
        prompts.append(prompt)

    encoded = tokenizer(
        prompts,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def generate_reasoning_chain_data(batch_size, seq_len):
    """Generate chain-of-thought reasoning examples with diverse templates."""
    templates = [
        "If {premise1} and {premise2}, then {conclusion}.",
        "Given that {premise1}, we can infer {conclusion}.",
        "Since {premise1} and {premise2}, it follows that {conclusion}.",
        "Starting from {premise1}, we derive {conclusion}.",
        "From {premise1} and {premise2}, we conclude {conclusion}.",
        "Premise 1: {premise1}. Premise 2: {premise2}. Conclusion: {conclusion}.",
        "If {premise1} holds, and {premise2} is true, then logically {conclusion}.",
        "Assuming {premise1} and given {premise2}, we deduce {conclusion}.",
        "Reasoning: {premise1} → {premise2} → {conclusion}.",
    ]

    # Extended premise/conclusion pairs
    reasoning_pairs = [
        # Classic syllogisms
        ("all birds can fly", "penguins are birds", "penguins can fly"),
        ("all humans are mortal", "Socrates is human", "Socrates is mortal"),
        (
            "all squares are rectangles",
            "this shape is a square",
            "this shape is a rectangle",
        ),
        ("A implies B", "B implies C", "A implies C"),
        # Cause and effect
        ("if it rains, the ground is wet", "it is raining", "the ground is wet"),
        ("if the switch is on, the light is on", "the switch is on", "the light is on"),
        ("if temperature rises, ice melts", "temperature rises", "ice melts"),
        # Mathematical
        ("x is greater than y", "y is greater than z", "x is greater than z"),
        ("A equals B", "B equals C", "A equals C"),
        ("all even numbers are divisible by 2", "4 is even", "4 is divisible by 2"),
        # Categorical
        ("all dogs are mammals", "mammals are animals", "dogs are animals"),
        ("no reptiles are mammals", "snakes are reptiles", "snakes are not mammals"),
        ("some birds cannot fly", "penguins are birds", "some penguins cannot fly"),
        # Temporal/conditional
        ("if study then pass", "if pass then graduate", "if study then graduate"),
        ("if A then B", "not B", "not A"),
    ]

    prompts = []
    for _ in range(batch_size):
        template = random.choice(templates)
        premise1, premise2, conclusion = random.choice(reasoning_pairs)
        text = template.format(
            premise1=premise1, premise2=premise2, conclusion=conclusion
        )
        prompts.append(text)

    encoded = tokenizer(
        prompts,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def load_language_datasets(config):
    """Load and prepare language datasets with streaming-friendly samplers."""
    loaded = []
    buffer_size = config.get("stream_buffer_size", 256)

    for spec in config["language_datasets"]:
        dataset_id = spec["path"]
        if spec.get("name"):
            dataset_id = f"{dataset_id}/{spec['name']}"

        try:
            dataset_stream = load_dataset(
                spec["path"],
                spec.get("name"),
                split=spec.get("split", "train"),
                streaming=True,
            )
            sampler = StreamingDatasetSampler(dataset_stream, buffer_size=buffer_size)
            loaded.append(
                {
                    "id": dataset_id,
                    "sampler": sampler,
                    "text_field": spec.get("text_field", "text"),
                    "weight": spec.get("weight", 1.0),
                }
            )
            print(
                f"Streaming dataset ready: {dataset_id} "
                f"(buffer={buffer_size}, weight={spec.get('weight', 1.0)})"
            )

        except Exception as exc:
            print(
                f"Streaming unavailable for {dataset_id}: {exc}. Falling back to eager iterator."
            )
            dataset = load_dataset(
                spec["path"],
                spec.get("name"),
                split=spec.get("split", "train"),
                streaming=False,
            )
            sampler = StreamingDatasetSampler(dataset, buffer_size=buffer_size)
            loaded.append(
                {
                    "id": dataset_id,
                    "sampler": sampler,
                    "text_field": spec.get("text_field", "text"),
                    "weight": spec.get("weight", 1.0),
                }
            )

    if not loaded:
        print("WARNING: No datasets loaded. Using synthetic data only.")

    return loaded


def get_amp_settings(config):
    """Get automatic mixed precision settings."""
    use_amp = config["use_mixed_precision"] and device.type == "cuda"
    amp_dtype = torch.float16

    if config.get("mixed_precision_dtype", "bfloat16") == "bfloat16":
        if use_amp and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif use_amp:
            print("bfloat16 not supported on this GPU, falling back to float16.")

    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    return use_amp, amp_dtype, scaler


def create_optimizer_and_scheduler(
    model, config, lr_scale, total_steps, warmup_steps=None
):
    """Create optimizer and learning rate scheduler."""
    max_lr = config["learning_rate"] * config["aggressive_lr_multiplier"] * lr_scale
    min_lr = config.get("min_learning_rate", 1e-6)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8
    )

    if warmup_steps is None:
        warmup_steps = int(total_steps * config["warmup_fraction"])

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / max_lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def evaluate_model(model, data_loader, criterion, config, num_batches=10):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            input_ids, attention_mask, labels = batch

            logits, _ = model(input_ids)

            # Calculate loss only on non-padding tokens
            loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))

            # Count valid tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

    return avg_loss, perplexity


def run_training_phase(
    model,
    config,
    phase_name,
    total_steps,
    batch_generator,
    lr_scale,
    checkpoint_name,
    time_budget=None,
    validation_generator=None,
):
    """Run a training phase with validation and early stopping."""
    use_amp, amp_dtype, scaler = get_amp_settings(config)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, lr_scale, total_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    model.train()
    start_time = time.time()
    completed_steps = 0

    # Tracking
    best_loss = float("inf")
    patience_counter = 0
    losses = []

    for step in range(total_steps):
        input_ids, attention_mask, labels = batch_generator()

        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

        with amp_context:
            logits, iterations = model(input_ids)
            loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

        completed_steps = step + 1
        now = time.time()
        elapsed = now - start_time

        losses.append(loss.item() * grad_accum_steps)

        # Logging
        if step % config["log_every"] == 0:
            lr = scheduler.get_last_lr()[0]
            steps_per_sec = completed_steps / max(elapsed, 0.001)
            eta = (total_steps - step) / max(steps_per_sec, 0.001)
            avg_loss = sum(losses[-10:]) / min(len(losses), 10)

            print(
                f"{phase_name} | Step {completed_steps}/{total_steps} "
                f"Loss: {avg_loss:.4f} LR: {lr:.2e} "
                f"Iter: {iterations} "
                f"Speed: {steps_per_sec:.1f} steps/s ETA: {eta / 60:.1f}min"
            )
            if torch.cuda.is_available():
                print(
                    f"  VRAM usage: "
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB used"
                )

        # Validation
        if (
            validation_generator
            and step % config.get("eval_every", 100) == 0
            and step > 0
        ):
            val_loss, perplexity = evaluate_model(
                model,
                validation_generator,
                criterion,
                config,
                num_batches=config.get("validation_batches", 5),
            )
            print(f"  Validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Early stopping check
        if val_loss < best_loss - config.get("min_delta", 0.001):
            best_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            best_path = os.path.join(
                config["checkpoint_dir"], f"{checkpoint_name}_best.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "loss": val_loss,
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 5):
                print(f"Early stopping triggered at step {step}")
                break

        # Time budget check
        if time_budget and elapsed >= time_budget:
            print(f"{phase_name} reached time budget after {completed_steps} steps.")
            break

        # Checkpoint saving
        if (step + 1) % config["save_every"] == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"{checkpoint_name}_step_{step + 1}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step + 1,
                    "loss": loss.item() * grad_accum_steps,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            # Cleanup old checkpoints to save space
            cleanup_old_checkpoints(config["checkpoint_dir"], keep_last_n=3)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(
        f"{phase_name} completed in {total_time / 60:.1f} minutes "
        f"({completed_steps} steps)."
    )

    # Save phase checkpoint
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{checkpoint_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    print(f"{phase_name} checkpoint saved: {checkpoint_path}")
    # Cleanup old checkpoints to save space (keep phase checkpoint + recent ones)
    cleanup_old_checkpoints(config["checkpoint_dir"], keep_last_n=5)

    return model


def train_phase_1(model, config, language_datasets):
    """Phase 1: Language Pretraining."""
    print("\n" + "=" * 60)
    print("Phase 1: Language Pretraining (Rich Datasets)")
    print("=" * 60)

    train_loader = ImprovedDataLoader(
        language_datasets, config["batch_size"], config["seq_len"], config
    )

    val_loader = None
    if language_datasets and config.get("validation_split", 0) > 0:
        val_loader = ImprovedDataLoader(
            language_datasets,
            config["batch_size"],
            config["seq_len"],
            config,
            is_validation=True,
        )

    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 1",
        total_steps=config["phase1_steps"],
        batch_generator=lambda: next(train_loader),
        lr_scale=1.0,
        checkpoint_name="phase_1",
        time_budget=config["phase_time_budgets"].get("phase_1"),
        validation_generator=val_loader,
    )


def train_phase_2(model, config):
    """Phase 2: Algorithmic Curriculum."""
    print("\n" + "=" * 60)
    print("Phase 2: Algorithmic Curriculum")
    print("=" * 60)

    step_counter = [0]
    total_steps = config["phase2_steps"]

    def batch_generator():
        step_counter[0] += 1
        # Gradually increase difficulty
        if step_counter[0] < total_steps * 0.3:
            difficulty = "easy"
        elif step_counter[0] < total_steps * 0.7:
            difficulty = "mixed"
        else:
            difficulty = "hard"
        return generate_arithmetic_data(
            config["batch_size"], config["seq_len"], difficulty
        )

    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 2",
        total_steps=total_steps,
        batch_generator=batch_generator,
        lr_scale=0.8,
        checkpoint_name="phase_2",
        time_budget=config["phase_time_budgets"].get("phase_2"),
    )


def train_phase_3(model, config, language_datasets):
    """Phase 3: Reasoning Distillation from GPT-2."""
    print("\n" + "=" * 60)
    print("Phase 3: Reasoning Distillation")
    print("=" * 60)

    teacher = None
    try:
        print("Loading GPT-2 as teacher model...")
        teacher = GPT2LMHeadModel.from_pretrained("gpt2")
        teacher.to(device)
        teacher.eval()
        print("Teacher model loaded successfully.")
    except Exception as exc:
        print(f"Could not load teacher model: {exc}")
        print("Continuing without distillation.")

    train_loader = ImprovedDataLoader(
        language_datasets, config["batch_size"], config["seq_len"], config
    )

    # Custom training loop with distillation
    use_amp, amp_dtype, scaler = get_amp_settings(config)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, 0.6, config["phase3_steps"]
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    model.train()
    start_time = time.time()

    for step in range(config["phase3_steps"]):
        input_ids, attention_mask, labels = next(train_loader)

        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

        with amp_context:
            student_logits, _ = model(input_ids)

            # Standard cross-entropy loss
            ce_loss = criterion(
                student_logits.view(-1, config["vocab_size"]), labels.view(-1)
            )

            # Distillation loss from teacher
            distill_loss = torch.tensor(0.0, device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_outputs.logits

                # Soft targets with temperature
                temperature = 2.0
                student_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                distill_loss = kl_loss(
                    student_probs.view(-1, config["vocab_size"]),
                    teacher_probs.view(-1, config["vocab_size"]),
                ) * (temperature**2)

            # Combined loss
            alpha = 0.7  # Weight for standard loss
            loss = alpha * ce_loss + (1 - alpha) * distill_loss
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

        if step % config["log_every"] == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Phase 3 | Step {step + 1}/{config['phase3_steps']} "
                f"CE: {ce_loss.item():.4f} Dist: {distill_loss.item():.4f} "
                f"LR: {lr:.2e}"
            )
            if torch.cuda.is_available():
                print(
                    f"  VRAM usage: "
                    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB used"
                )

        if (step + 1) % config["save_every"] == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"phase_3_step_{step + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Phase 3 completed in {(time.time() - start_time) / 60:.1f} minutes")

    checkpoint_path = os.path.join(config["checkpoint_dir"], "phase_3.pt")
    torch.save(model.state_dict(), checkpoint_path)

    return model


def train_phase_4(model, config, language_datasets):
    """Phase 4: Self-Consistency Training."""
    print("\n" + "=" * 60)
    print("Phase 4: Self-Consistency Training")
    print("=" * 60)

    train_loader = ImprovedDataLoader(
        language_datasets, config["batch_size"], config["seq_len"], config
    )

    # Mix language data with reasoning chains
    step_counter = [0]
    reasoning_data_gen = lambda: generate_reasoning_chain_data(
        config["batch_size"], config["seq_len"]
    )

    def mixed_batch_generator():
        step_counter[0] += 1
        if step_counter[0] % 4 == 0:  # Every 4th batch, use reasoning data
            return reasoning_data_gen()
        return next(train_loader)

    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 4",
        total_steps=config["phase4_steps"],
        batch_generator=mixed_batch_generator,
        lr_scale=0.4,
        checkpoint_name="phase_4",
        time_budget=config["phase_time_budgets"].get("phase_4"),
    )


def initialize_model_weights(model):
    """Initialize model weights with proper schemes."""
    for name, p in model.named_parameters():
        if "embedding" in name.lower():
            nn.init.normal_(p, mean=0, std=0.02)
        elif "weight" in name and p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)


def main():
    print(f"Using device: {device}")
    print(f"VRAM config: {VRAM_MODE}")
    print(f"Training for {TARGET_TRAINING_TIME_HOURS} hours")
    print("-" * 60)

    # Create model
    model = IDIR(
        vocab_size=TRAINING_CONFIG["vocab_size"],
        hidden_dim=TRAINING_CONFIG["hidden_dim"],
        num_experts=TRAINING_CONFIG["num_experts"],
        expert_hidden_dim=TRAINING_CONFIG["expert_hidden_dim"],
        reasoning_steps=TRAINING_CONFIG["reasoning_steps"],
        k_features=TRAINING_CONFIG["k_features"],
        reasoning_hidden_dim=TRAINING_CONFIG["reasoning_hidden_dim"],
        max_iterations=TRAINING_CONFIG["max_iterations"],
        tolerance=TRAINING_CONFIG["tolerance"],
    )
    model.to(device)

    # Initialize weights
    initialize_model_weights(model)
    print("Model initialized with proper weight initialization")

    # Enable gradient checkpointing
    if TRAINING_CONFIG["use_gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Compile model if available
    if TRAINING_CONFIG.get("compile_model") and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully")
        except Exception as e:
            print(f"Could not compile model: {e}")

    # Load datasets
    print("\nLoading datasets...")
    language_datasets = load_language_datasets(TRAINING_CONFIG)
    print(f"Loaded {len(language_datasets)} datasets")

    # Training phases
    try:
        model = train_phase_1(model, TRAINING_CONFIG, language_datasets)
        model = train_phase_2(model, TRAINING_CONFIG)
        model = train_phase_3(model, TRAINING_CONFIG, language_datasets)
        model = train_phase_4(model, TRAINING_CONFIG, language_datasets)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    # Save final model
    final_path = os.path.join(TRAINING_CONFIG["checkpoint_dir"], "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": TRAINING_CONFIG,
        },
        final_path,
    )
    print(f"\nFinal model saved to: {final_path}")

    # Also save as idir_model.pt for compatibility
    torch.save(model.state_dict(), "idir_model.pt")
    print(f"Model also saved to: idir_model.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
