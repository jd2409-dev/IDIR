"""High-Quality Open Dataset Loader for IDIR-KS

Streams high-quality ungated datasets from HuggingFace.
Falls back to synthetic data if unavailable.
Zero storage footprint - uses streaming mode only.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Callable
import random
import os


# High-quality ungated datasets (confirmed open, no loading scripts, streaming-safe)
OPEN_DATASETS = {
    "code": [
        {
            "name": "nampdn-ai/tiny-codes",
            "text_col": "code",
            "desc": "Tiny codes dataset",
        },
    ],
    "math": [
        {
            "name": "openai/gsm8k",
            "text_col": "text",
            "split": "train",
            "desc": "GSM8K math word problems",
        },
    ],
    "logic": [
        {
            "name": "allenai/ai2_arc",
            "text_col": "question",
            "split": "train",
            "desc": "AI2 ARC reasoning questions",
        },
    ],
    "language": [
        {
            "name": "Salesforce/wikitext",
            "split": "train",
            "config": "wikitext-2-raw-v1",
            "text_col": "text",
            "desc": "WikiText-2 (high quality prose)",
        },
        {
            "name": "stas/openwebtext-10k",
            "text_col": "text",
            "desc": "OpenWebText sample (10k)",
        },
    ],
}


class StreamingTextDataset(Dataset):
    """
    Memory-efficient dataset that loads samples on-demand.
    Works with both HuggingFace streaming and local data.
    """

    def __init__(
        self,
        samples: List[str],
        max_length: int = 128,
        tokenizer: Optional[Callable] = None,
    ):
        self.samples = samples
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text = self.samples[idx]

        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length, truncation=True)
            input_ids = tokens["input_ids"]
        else:
            input_ids = [ord(c) % 256 for c in text[: self.max_length]]
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        input_ids = input_ids[: self.max_length]
        labels = input_ids[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def try_load_hf_dataset(
    dataset_info: Dict,
    max_samples: int,
    max_length: int,
    seed: int = 42,
) -> Optional[StreamingTextDataset]:
    """
    Try to load a HuggingFace dataset in streaming mode.
    Returns None if loading fails.
    """
    try:
        from datasets import load_dataset

        name = dataset_info["name"]
        text_col = dataset_info.get("text_col", "text")
        split = dataset_info.get("split", "train")
        config_name = dataset_info.get("config", None)

        print(f"  Trying: {name} ({dataset_info.get('desc', '')})")

        if config_name:
            ds = load_dataset(
                name,
                config_name,
                split=split,
                streaming=True,
            )
        else:
            ds = load_dataset(
                name,
                split=split,
                streaming=True,
            )

        samples = []
        rng = random.Random(seed)

        for i, item in enumerate(ds):
            if i >= max_samples * 3:
                break

            if isinstance(item, dict):
                text = str(item.get(text_col, ""))
            else:
                text = str(item)

            if len(text) > 10:
                samples.append(text)

            if len(samples) >= max_samples:
                break

        if len(samples) > 10:
            print(f"    Loaded {len(samples)} samples")
            return StreamingTextDataset(samples, max_length=max_length)
        else:
            print(f"    Only {len(samples)} samples, skipping")
            return None

    except Exception as e:
        print(f"    Failed: {str(e)[:80]}")
        return None


def load_open_datasets(
    weights: Dict[str, float] = None,
    max_samples: int = 10000,
    max_length: int = 128,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """
    Load high-quality open datasets with streaming.
    Falls back to synthetic data for any domain that fails.

    Args:
        weights: Domain weights (code, math, logic, language)
        max_samples: Max samples per domain
        max_length: Max sequence length
        seed: Random seed

    Returns:
        Dict of {domain: Dataset}
    """
    if weights is None:
        weights = {
            "code": 0.40,
            "math": 0.25,
            "logic": 0.20,
            "language": 0.15,
        }

    datasets = {}

    for domain in weights:
        domain_datasets = OPEN_DATASETS.get(domain, [])
        loaded = False

        for ds_info in domain_datasets:
            ds = try_load_hf_dataset(ds_info, max_samples, max_length, seed)
            if ds is not None:
                datasets[domain] = ds
                loaded = True
                break

        if not loaded:
            print(f"  Using synthetic data for {domain}")
            from ..data.synthetic_data import SyntheticDataGenerator

            generator = SyntheticDataGenerator(seed=seed)
            samples_per_domain = max(100, max_samples)

            if domain == "code":
                raw_samples = generator.generate_code_samples(samples_per_domain)
            elif domain == "math":
                raw_samples = generator.generate_math_samples(samples_per_domain)
            elif domain == "logic":
                raw_samples = generator.generate_logic_samples(samples_per_domain)
            else:
                raw_samples = generator.generate_language_samples(samples_per_domain)

            texts = [s["text"] for s in raw_samples]
            datasets[domain] = StreamingTextDataset(texts, max_length=max_length)

    return datasets


def create_open_dataloader(
    weights: Dict[str, float] = None,
    batch_size: int = 4,
    max_samples: int = 10000,
    max_length: int = 128,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from high-quality open datasets.
    Falls back to synthetic data automatically.

    This is the recommended data loading method for training.
    """
    print("Loading datasets...")
    domain_datasets = load_open_datasets(weights, max_samples, max_length, seed)

    # Combine all domains
    all_samples = []
    for domain, ds in domain_datasets.items():
        weight = weights.get(domain, 0.25)
        n = int(max_samples * weight)
        rng = random.Random(seed + hash(domain) % 10000)

        for _ in range(n):
            idx = rng.randint(0, len(ds) - 1)
            item = ds[idx]
            item["domain"] = domain
            all_samples.append(item)

    random.Random(seed).shuffle(all_samples)
    print(f"Total samples: {len(all_samples)}")

    class CombinedDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = CombinedDataset(all_samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=default_collate_fn,
        drop_last=False,
    )


def default_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for batched data"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    result = {
        "input_ids": input_ids,
        "labels": labels,
    }

    if "domain" in batch[0]:
        result["domain"] = [item["domain"] for item in batch]

    return result
