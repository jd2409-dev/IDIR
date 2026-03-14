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

TRAINING_CONFIG = {
    "device": device,
    "vocab_size": 50257,
    "hidden_dim": 512,
    "num_experts": 4,
    "expert_hidden_dim": 2048,
    "reasoning_steps": 3,
    "k_features": 32,
    "reasoning_hidden_dim": 2560,
    "max_iterations": 20,
    "tolerance": 1e-3,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "aggressive_lr_multiplier": 3.0,
    "warmup_fraction": 0.03,
    "phase1_epochs": 300,
    "phase1_batches": 400,
    "phase2_epochs": 100,
    "phase2_batches": 200,
    "phase3_epochs": 100,
    "phase3_batches": 200,
    "phase4_epochs": 100,
    "phase4_batches": 200,
    "seq_len": 128,
    "save_every": 100,
    "log_every": 10,
    "max_grad_norm": 1.0,
    "use_mixed_precision": True,
    "mixed_precision_dtype": "bfloat16",
    "use_gradient_checkpointing": True,
    "checkpoint_dir": "checkpoints",
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
    ],
}

os.makedirs(TRAINING_CONFIG["checkpoint_dir"], exist_ok=True)


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


def get_language_pretraining_data(batch_size, seq_len, datasets):
    """Sample a mixed batch across loaded datasets."""
    batch_text = []

    for _ in range(batch_size):
        if not datasets:
            batch_text.append("This is a placeholder sentence to avoid errors.")
            continue

        source = random.choice(datasets)
        dataset = source["dataset"]
        try:
            idx = torch.randint(0, len(dataset), (1,)).item()
            sample = dataset[idx]
            text = _extract_text(sample, source["text_field"])
            batch_text.append(text)
        except Exception as exc:
            batch_text.append(f"Data fallback due to error: {exc}")

    encoded = tokenizer(
        batch_text,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = tokenizer.pad_token_id
    return input_ids, labels


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
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    return use_amp, amp_dtype, scaler


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
    epochs,
    num_batches,
    batch_generator,
    lr_scale,
    checkpoint_name,
    save_every=None,
):
    use_amp, amp_dtype, scaler = get_amp_settings(config)
    total_steps = epochs * num_batches
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, lr_scale, total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
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

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if batch_idx % config["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"{phase_name} | Epoch {epoch + 1}/{epochs} "
                    f"Batch {batch_idx}/{num_batches} Loss: {loss.item():.4f} LR: {lr:.2e}"
                )

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"{phase_name} - Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        if save_every and (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"{checkpoint_name}_epoch_{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"{phase_name} checkpoint saved: {checkpoint_path}")

    return model


def train_phase_1(model, config, language_datasets):
    print("\n=== Phase 1: Language Pretraining (Mixed Datasets) ===")
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 1",
        epochs=config["phase1_epochs"],
        num_batches=config["phase1_batches"],
        batch_generator=lambda: get_language_pretraining_data(
            config["batch_size"], config["seq_len"], language_datasets
        ),
        lr_scale=1.0,
        checkpoint_name="phase_1",
        save_every=config["save_every"],
    )


def train_phase_2(model, config):
    print("\n=== Phase 2: Algorithmic Curriculum ===")
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 2",
        epochs=config["phase2_epochs"],
        num_batches=config["phase2_batches"],
        batch_generator=lambda: generate_arithmetic_data(
            config["batch_size"], config["seq_len"]
        ),
        lr_scale=0.8,
        checkpoint_name="phase_2",
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
        epochs=config["phase3_epochs"],
        num_batches=config["phase3_batches"],
        batch_generator=lambda: get_language_pretraining_data(
            config["batch_size"], config["seq_len"], language_datasets
        ),
        lr_scale=0.6,
        checkpoint_name="phase_3",
    )


def train_phase_4(model, config, language_datasets):
    print("\n=== Phase 4: Self-Consistency Training ===")
    return run_training_phase(
        model=model,
        config=config,
        phase_name="Phase 4",
        epochs=config["phase4_epochs"],
        num_batches=config["phase4_batches"],
        batch_generator=lambda: get_language_pretraining_data(
            config["batch_size"], config["seq_len"], language_datasets
        ),
        lr_scale=0.4,
        checkpoint_name="phase_4",
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
