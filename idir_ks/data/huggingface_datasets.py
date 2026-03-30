"""Open HuggingFace Datasets for IDIR-KS

Uses only open (non-gated) datasets from HuggingFace.
If datasets are not available, falls back to .py data generators.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Callable
import random
import os


# List of open datasets to use
OPEN_DATASETS = {
    "code": [
        "codeparrot/github-code",  # Open subset
        "iamtarun/python_code_instructions",  # Open Python code
        "huggingface-course/code-50k",  # Small open code dataset
    ],
    "math": [
        "competition_math",  # Competition math
        "hendrycks/competition_math",  # Alternative name
        "math_dataset",  # General math
        " EleutherAI/proof-pile",  # Mathematical proofs (check if open)
    ],
    "logic": [
        "tasksource/bigbench",  # Logical reasoning tasks
        "facebook/logical_fallacy",  # Logical reasoning
        "openai/logiqa",  # Logic QA
        "allenai/ai2_arc",  # ARC dataset
    ],
    "language": [
        "openwebtext",  # Open web text
        "pile",  # The Pile (open version)
        "c4",  # C4 dataset
        "wikitext",  # Wikitext
    ],
}


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_samples: int = 10000,
        max_length: int = 1024,
        tokenizer: Optional[Callable] = None,
        cache_dir: str = "./cache",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.max_samples = max_samples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

        # Try to load from HuggingFace
        self.dataset = self._load_dataset()

        if self.dataset is None:
            print(f"Warning: Could not load {dataset_name}, using fallback data")
            self.dataset = self._create_fallback_data()

    def _load_dataset(self):
        """Try to load dataset from HuggingFace"""
        try:
            from datasets import load_dataset

            # Try loading with different configurations
            configs_to_try = [None, "default", "train"]

            for config in configs_to_try:
                try:
                    if config:
                        dataset = load_dataset(
                            self.dataset_name,
                            config,
                            split=self.split,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                        )
                    else:
                        dataset = load_dataset(
                            self.dataset_name,
                            split=self.split,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                        )

                    # Limit samples
                    if hasattr(dataset, "select"):
                        dataset = dataset.select(
                            range(min(self.max_samples, len(dataset)))
                        )

                    return dataset
                except Exception as e:
                    continue

            return None
        except ImportError:
            print("Warning: datasets library not installed")
            return None
        except Exception as e:
            print(f"Error loading {self.dataset_name}: {e}")
            return None

    def _create_fallback_data(self):
        """Create synthetic fallback data"""
        from .synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator()

        # Generate based on dataset type
        if "code" in self.dataset_name.lower():
            return generator.generate_code_samples(self.max_samples)
        elif "math" in self.dataset_name.lower():
            return generator.generate_math_samples(self.max_samples)
        elif "logic" in self.dataset_name.lower():
            return generator.generate_logic_samples(self.max_samples)
        else:
            return generator.generate_language_samples(self.max_samples)

    def __len__(self) -> int:
        if isinstance(self.dataset, list):
            return len(self.dataset)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]

        # Extract text
        if isinstance(item, dict):
            text = item.get(self.text_column, "")
            if not text:
                # Try common column names
                for col in ["text", "content", "input", "sentence", "code"]:
                    if col in item:
                        text = item[col]
                        break
        else:
            text = str(item)

        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length, truncation=True)
            input_ids = tokens["input_ids"]
        else:
            # Simple character-level fallback
            input_ids = [ord(c) % 256 for c in text[: self.max_length]]
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        # Create labels (shifted input)
        labels = input_ids[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
        }


class CompositeHuggingFaceDataset(Dataset):
    """Composite dataset mixing multiple HuggingFace datasets"""

    def __init__(
        self,
        dataset_weights: Dict[str, float],
        max_samples: int = 100000,
        max_length: int = 1024,
        tokenizer: Optional[Callable] = None,
        cache_dir: str = "./cache",
        seed: int = 42,
    ):
        self.dataset_weights = dataset_weights
        self.max_samples = max_samples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.rng = random.Random(seed)

        # Load datasets
        self.datasets = {}
        self.total_samples_per_dataset = {}

        total_weight = sum(dataset_weights.values())

        for domain, weight in dataset_weights.items():
            dataset_list = OPEN_DATASETS.get(domain, [])

            # Try datasets until one works
            dataset = None
            for dataset_name in dataset_list:
                try:
                    dataset = HuggingFaceDatasetWrapper(
                        dataset_name=dataset_name,
                        split="train",
                        max_samples=int(max_samples * weight / total_weight),
                        max_length=max_length,
                        tokenizer=tokenizer,
                        cache_dir=cache_dir,
                    )
                    print(f"Successfully loaded {dataset_name} for {domain}")
                    break
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")
                    continue

            if dataset is None:
                print(f"Using synthetic data for {domain}")
                from .synthetic_data import SyntheticDataGenerator

                generator = SyntheticDataGenerator(seed=seed)

                if domain == "code":
                    samples = generator.generate_code_samples(
                        int(max_samples * weight / total_weight)
                    )
                elif domain == "math":
                    samples = generator.generate_math_samples(
                        int(max_samples * weight / total_weight)
                    )
                elif domain == "logic":
                    samples = generator.generate_logic_samples(
                        int(max_samples * weight / total_weight)
                    )
                else:
                    samples = generator.generate_language_samples(
                        int(max_samples * weight / total_weight)
                    )

                dataset = samples

            self.datasets[domain] = dataset
            self.total_samples_per_dataset[domain] = len(dataset)

        # Create balanced indices
        self.indices = self._create_indices()

    def _create_indices(self) -> List[tuple]:
        """Create balanced sampling indices"""
        indices = []

        for domain, dataset in self.datasets.items():
            weight = self.dataset_weights[domain]
            num_samples = int(self.max_samples * weight)

            for _ in range(num_samples):
                idx = self.rng.randint(0, len(dataset) - 1)
                indices.append((domain, idx))

        self.rng.shuffle(indices)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        domain, sample_idx = self.indices[idx]
        dataset = self.datasets[domain]

        sample = dataset[sample_idx]
        sample["domain"] = domain

        return sample

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            "total_samples": len(self),
            "datasets": {k: len(v) for k, v in self.datasets.items()},
            "weights": self.dataset_weights,
        }


def create_hf_dataloader(
    dataset_weights: Dict[str, float] = None,
    batch_size: int = 32,
    max_samples: int = 100000,
    max_length: int = 1024,
    tokenizer: Optional[Callable] = None,
    cache_dir: str = "./cache",
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader using HuggingFace datasets"""

    if dataset_weights is None:
        # Default from paper
        dataset_weights = {
            "code": 0.40,
            "math": 0.25,
            "logic": 0.20,
            "language": 0.15,
        }

    dataset = CompositeHuggingFaceDataset(
        dataset_weights=dataset_weights,
        max_samples=max_samples,
        max_length=max_length,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        seed=seed,
    )

    print(f"Dataset stats: {dataset.get_stats()}")

    def collate_fn(batch: List[Dict]) -> Dict:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if "domain" in batch[0]:
            result["domain"] = [item["domain"] for item in batch]

        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def clear_cache(cache_dir: str = "./cache"):
    """Clear HuggingFace cache"""
    import shutil

    if os.path.exists(cache_dir):
        print(f"Clearing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print("Cache cleared successfully")

    # Also try to clear HF cache
    try:
        import huggingface_hub

        hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            print(f"Clearing HF cache: {hf_cache}")
            # Only clear datasets, not models
            datasets_cache = os.path.join(hf_cache, "datasets")
            if os.path.exists(datasets_cache):
                shutil.rmtree(datasets_cache)
                print("HF datasets cache cleared")
    except Exception as e:
        print(f"Could not clear HF cache: {e}")
