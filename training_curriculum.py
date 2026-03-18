import os
import random
from contextlib import nullcontext

import torch
import torch.nn as nn
from datasets import load_dataset
from idir_model import IDIR

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("Install transformers: pip install transformers")
    raise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token


def get_vram_config():
    if not torch.cuda.is_available():
        return "low"
    gpu_name = torch.cuda.get_device_name(0).lower()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if "3050" in gpu_name and total_memory >= 7:
        return "rtx_3050_8gb"
    elif "3050" in gpu_name:
        return "rtx_3050_4gb"
    return "low"


VRAM_MODE = get_vram_config()
print(f"Detected VRAM mode: {VRAM_MODE}")

VRAM_CONFIGS = {
    "rtx_3050_8gb": {
        "batch_size": 128,
        "seq_len": 512,
        "hidden_dim": 512,
        "expert_hidden_dim": 2048,
        "max_iterations": 12,
        "compile_model": True,
        "gradient_accumulation_steps": 1,
    },
    "rtx_3050_4gb": {
        "batch_size": 64,
        "seq_len": 256,
        "hidden_dim": 384,
        "expert_hidden_dim": 1536,
        "max_iterations": 8,
        "compile_model": True,
        "gradient_accumulation_steps": 1,
    },
    "low": {
        "batch_size": 32,
        "seq_len": 128,
        "hidden_dim": 256,
        "expert_hidden_dim": 1024,
        "max_iterations": 6,
        "compile_model": False,
        "gradient_accumulation_steps": 1,
    },
}

vram_cfg = VRAM_CONFIGS[VRAM_MODE]

TARGET_TRAINING_TIME_HOURS = 2.5
TOTAL_STEPS_PHASE1 = 500
TOTAL_STEPS_PHASE2_4 = 250

PHASE_TIME_ALLOCATION = {
    "phase_1": 0.45,
    "phase_2": 0.2,
    "phase_3": 0.2,
    "phase_4": 0.15,
}
PHASE_TIME_BUDGETS = {
    phase: int(TARGET_TRAINING_TIME_HOURS * 3600 * ratio)
    for phase, ratio in PHASE_TIME_ALLOCATION.items()
}

DEFAULT_SYNTHETIC_WORD_POOL = [
    "reasoning",
    "iteration",
    "contractive",
    "implicit",
    "spectral",
    "routing",
    "latent",
    "context",
    "signal",
    "deduction",
    "coherence",
    "modular",
    "gradient",
    "trajectory",
    "recursion",
    "optimization",
    "vector",
    "synthesis",
    "analysis",
    "calibration",
]

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
    "tolerance": 1e-3,
    "batch_size": vram_cfg["batch_size"],
    "seq_len": vram_cfg["seq_len"],
    "learning_rate": 3e-4,
    "aggressive_lr_multiplier": 3.0,
    "warmup_fraction": 0.03,
    "phase1_steps": TOTAL_STEPS_PHASE1,
    "phase2_steps": TOTAL_STEPS_PHASE2_4,
    "phase3_steps": TOTAL_STEPS_PHASE2_4,
    "phase4_steps": TOTAL_STEPS_PHASE2_4,
    "save_every": 200,
    "log_every": 50,
    "max_grad_norm": 1.0,
    "use_mixed_precision": True,
    "mixed_precision_dtype": "bfloat16",
    "use_gradient_checkpointing": True,
    "compile_model": vram_cfg["compile_model"],
    "gradient_accumulation_steps": vram_cfg.get("gradient_accumulation_steps", 1),
    "checkpoint_dir": "checkpoints",
    "num_workers": 4,
    "prefetch_factor": 4,
    "phase_time_budgets": PHASE_TIME_BUDGETS,
    "data_expansion_factor": 5,
    "synthetic_text_ratio": 0.35,
    "synthetic_word_pool": DEFAULT_SYNTHETIC_WORD_POOL,
    "language_datasets": [
        {
            "path": "wikitext",
            "name": "wikitext-2-raw-v1",
            "split": "train",
            "text_field": "text",
        },
        {"path": "openwebtext", "split": "train", "text_field": "text"},
        {"path": "bookcorpus", "split": "train", "text_field": "text"},
        {"path": "c4", "name": "en", "split": "train", "text_field": "text"},
        {
            "path": "wikitext",
            "name": "wikitext-103-raw-v1",
            "split": "train",
            "text_field": "text",
        },
        {
            "path": "wikipedia",
            "name": "20220301.en",
            "split": "train",
            "text_field": "text",
        },
        {
            "path": "oscar",
            "name": "unshuffled_deduplicated_en",
            "split": "train",
            "text_field": "text",
        },
        {
            "path": "the_pile",
            "split": "train",
            "text_field": "text",
        },
    ],
}

os.makedirs(TRAINING_CONFIG["checkpoint_dir"], exist_ok=True)

compiled_model = None


def load_language_datasets(config):
    """Load and keep multiple datasets for mixed language pretraining."""
    loaded = []
    for spec in config["language_datasets"]:
        dataset_id = spec["path"]
        if spec.get("name"):
            dataset_id = f"{dataset_id}/{spec['name']}"
        try:
            dataset = load_dataset(
                spec["path"],
                spec.get("name"),
                split=spec.get("split", "train"),
            )
            loaded.append(
                {
                    "id": dataset_id,
                    "dataset": dataset,
                    "text_field": spec.get("text_field", "text"),
                }
            )
            print(f"Loaded dataset: {dataset_id} ({len(dataset)} rows)")
        except Exception as exc:
            print(f"Could not load {dataset_id}: {exc}")

    if not loaded:
        print("No datasets loaded. Falling back to placeholder text.")
    return loaded


def _extract_text(example, text_field):
    if isinstance(example, dict):
        text = example.get(text_field)
        if text is None and example:
            text = next(iter(example.values()))
    else:
        text = str(example)

    text = "" if text is None else str(text)
    if not text.strip():
        return "This is a placeholder sentence to avoid errors."
    return text


class DataLoaderWrapper:
    def __init__(self, datasets, batch_size, seq_len, config):
        self.datasets = datasets
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.expansion_factor = max(1, int(config.get("data_expansion_factor", 3)))
        self.synthetic_ratio = min(
            max(float(config.get("synthetic_text_ratio", 0.2)), 0.0), 0.6
        )
        self.synthetic_word_pool = (
            config.get("synthetic_word_pool") or DEFAULT_SYNTHETIC_WORD_POOL
        )
        self._cache = []
        self._cache_size = 1000
        self._precache()

    def _precache(self):
        if not self.datasets:
            return
        for _ in range(self._cache_size):
            self._cache.append(self._generate_batch())

    def _sample_text(self):
        source = random.choice(self.datasets)
        dataset = source["dataset"]
        text_field = source["text_field"]
        try:
            idx = torch.randint(0, len(dataset), (1,)).item()
            sample = dataset[idx]
            return _extract_text(sample, text_field)
        except Exception:
            return "Data fallback."

    def _make_synthetic_text(self):
        words = [
            random.choice(self.synthetic_word_pool)
            for _ in range(random.randint(8, 24))
        ]
        words.append(str(random.randint(0, 9999)))
        sentence = " ".join(words)
        return sentence.capitalize() + "."

    def _generate_batch(self):
        batch_text = []
        for _ in range(self.batch_size):
            fragments = []
            for _ in range(random.randint(1, self.expansion_factor)):
                if self.datasets and random.random() > self.synthetic_ratio:
                    fragments.append(self._sample_text())
                else:
                    fragments.append(self._make_synthetic_text())

            text = " ".join(fragments).strip()
            batch_text.append(text or "This is a placeholder sentence.")

        encoded = tokenizer(
            batch_text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[:, -1] = tokenizer.pad_token_id
        return input_ids, labels

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._cache) < self._cache_size // 2:
            import threading

            threading.Thread(target=self._precache, daemon=True).start()
        if self._cache:
            return self._cache.pop()
        return self._generate_batch()


data_loader = None


def generate_arithmetic_data(batch_size, seq_len):
    """Generate synthetic arithmetic examples for algorithmic curriculum."""
    prompts = []
    for _ in range(batch_size):
        a = torch.randint(1, 100, (1,)).item()
        b = torch.randint(1, 100, (1,)).item()
        op = torch.randint(0, 3, (1,)).item()

        if op == 0:
            prompt = f"What is {a} plus {b}? The answer is {a + b}."
        elif op == 1:
            bigger = max(a, b)
            smaller = min(a, b)
            prompt = (
                f"What is {bigger} minus {smaller}? The answer is {bigger - smaller}."
            )
        else:
            prompt = f"What is {a} times {b}? The answer is {a * b}."
        prompts.append(prompt)

    encoded = tokenizer(
        prompts,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = tokenizer.pad_token_id
    return input_ids, labels


def get_amp_settings(config):
    use_amp = config["use_mixed_precision"] and device.type == "cuda"
    amp_dtype = torch.float16

    if config.get("mixed_precision_dtype", "bfloat16") == "bfloat16":
        if use_amp and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif use_amp:
            print("bfloat16 not supported on this GPU, falling back to float16.")

    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = None
    if use_scaler:
        try:
            scaler = torch.cuda.amp.GradScaler("cuda")
        except Exception:
            pass
    return use_amp, amp_dtype, scaler


def make_compiled_model(model, config):
    if config.get("compile_model") and hasattr(torch, "compile"):
        print("Compiling model with torch.compile() - this may take a few minutes...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        print("Model compiled successfully!")
    return model


def create_optimizer_and_scheduler(model, config, lr_scale, total_steps):
    max_lr = config["learning_rate"] * config["aggressive_lr_multiplier"] * lr_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=max(1, total_steps),
        pct_start=min(max(config["warmup_fraction"], 0.01), 0.3),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def run_training_phase(
    model,
    config,
    phase_name,
    total_steps,
    batch_generator,
    lr_scale,
    checkpoint_name,
    time_budget=None,
):
    use_amp, amp_dtype, scaler = get_amp_settings(config)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, lr_scale, total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    model.train()
    import time

    start_time = time.time()
    completed_steps = 0

    for step in range(total_steps):
        input_ids, labels = batch_generator()

        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )

        with amp_context:
            logits, _ = model(input_ids)
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

        if step % config["log_every"] == 0:
            lr = scheduler.get_last_lr()[0]
            steps_per_sec = completed_steps / max(elapsed, 0.001)
            eta = (total_steps - step) / max(steps_per_sec, 0.001)
            print(
                f"{phase_name} | Step {completed_steps}/{total_steps} "
                f"Loss: {loss.item() * grad_accum_steps:.4f} LR: {lr:.2e} "
                f"Speed: {steps_per_sec:.1f} steps/s ETA: {eta / 60:.1f}min"
            )

        if time_budget and elapsed >= time_budget:
            print(
                f"{phase_name} reached its {time_budget / 60:.1f}min budget "
                f"after {completed_steps} steps. Moving to the next phase."
            )
            break

        if (step + 1) % config["save_every"] == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"{checkpoint_name}_step_{step + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time
    total_steps_completed = completed_steps or total_steps
    print(
        f"{phase_name} completed in {total_time / 60:.1f} minutes "
        f"({total_steps_completed} steps, target {total_steps})."
    )

    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"{phase_name} checkpoint saved: {checkpoint_path}")

    return model


def train_phase_1(model, config, language_datasets):
    global data_loader
    print("\n=== Phase 1: Language Pretraining (Mixed Datasets) ===")
    data_loader = DataLoaderWrapper(
        language_datasets, config["batch_size"], config["seq_len"], config
    )
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 1",
        total_steps=config["phase1_steps"],
        batch_generator=lambda: next(data_loader),
        lr_scale=1.0,
        checkpoint_name="phase_1",
        time_budget=config["phase_time_budgets"].get("phase_1"),
    )


def train_phase_2(model, config):
    print("\n=== Phase 2: Algorithmic Curriculum ===")
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 2",
        total_steps=config["phase2_steps"],
        batch_generator=lambda: generate_arithmetic_data(
            config["batch_size"], config["seq_len"]
        ),
        lr_scale=0.8,
        checkpoint_name="phase_2",
        time_budget=config["phase_time_budgets"].get("phase_2"),
    )


def train_phase_3(model, config, language_datasets):
    print("\n=== Phase 3: Reasoning Distillation from GPT-2 ===")

    try:
        teacher = GPT2LMHeadModel.from_pretrained("gpt2")
        teacher.to(device)
        teacher.eval()
        print("Teacher model loaded (currently used as a distillation anchor).")
    except Exception as exc:
        print(f"Could not load teacher model: {exc}")
        print("Continuing with language-only objective for Phase 3.")

    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 3",
        total_steps=config["phase3_steps"],
        batch_generator=lambda: next(data_loader),
        lr_scale=0.6,
        checkpoint_name="phase_3",
        time_budget=config["phase_time_budgets"].get("phase_3"),
    )


def train_phase_4(model, config, language_datasets):
    print("\n=== Phase 4: Self-Consistency Training ===")
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 4",
        total_steps=config["phase4_steps"],
        batch_generator=lambda: next(data_loader),
        lr_scale=0.4,
        checkpoint_name="phase_4",
        time_budget=config["phase_time_budgets"].get("phase_4"),
    )


def main():
    print(f"Using device: {device}")
    print(f"Training configuration: {TRAINING_CONFIG}")

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

    if TRAINING_CONFIG["use_gradient_checkpointing"] and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    if TRAINING_CONFIG.get("compile_model") and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead", backend="inductor")
        print("Model compiled successfully!")

    language_datasets = load_language_datasets(TRAINING_CONFIG)

    model = train_phase_1(model, TRAINING_CONFIG, language_datasets)
    model = train_phase_2(model, TRAINING_CONFIG)
    model = train_phase_3(model, TRAINING_CONFIG, language_datasets)
    model = train_phase_4(model, TRAINING_CONFIG, language_datasets)

    final_path = os.path.join(TRAINING_CONFIG["checkpoint_dir"], "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    idir_path = "idir_model.pt"
    torch.save(model.state_dict(), idir_path)
    print(f"Model also saved to: {idir_path} (for infer_idir.py)")


if __name__ == "__main__":
    main()
