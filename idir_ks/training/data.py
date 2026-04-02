"""Data Loading and Dataset Composition for IDIR-KS"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Callable
import random
import json
from pathlib import Path


class TextDataset(Dataset):
    """Basic text dataset for language modeling"""

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Callable] = None,
        max_length: int = 1024,
        cache_dir: str = None,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Load data
        self.samples = self._load_data()

    def _load_data(self) -> List[str]:
        """Load text samples from file"""
        if self.data_path.suffix == ".jsonl":
            samples = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        text = data.get("text", data.get("content", ""))
                    else:
                        text = data
                    samples.append(text)
            return samples
        elif self.data_path.suffix in [".txt", ".md"]:
            with open(self.data_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split into chunks
            samples = self._chunk_text(text)
            return samples
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks"""
        # Simple tokenization approximation (words)
        words = text.split()
        chunks = []

        chunk_size = self.max_length * 2  # Approximate words to tokens ratio
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text = self.samples[idx]

        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length)
            input_ids = tokens["input_ids"]
        else:
            # Simple character-level tokenization for testing
            input_ids = [ord(c) % 256 for c in text[: self.max_length]]
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        # Create labels (shifted input for language modeling)
        labels = input_ids[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CodeDataset(TextDataset):
    """Dataset for code"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.domain = "code"


class MathDataset(TextDataset):
    """Dataset for math problems"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.domain = "math"


class LogicDataset(TextDataset):
    """Dataset for logical reasoning"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.domain = "logic"


class LanguageDataset(TextDataset):
    """Dataset for general language"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.domain = "language"


class WeightedDataset(Dataset):
    """
    Wrapper that samples from multiple datasets according to weights.
    Implements the dataset composition from the paper:
        Code: 40%, Math: 25%, Logic: 20%, Language: 15%
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights: Dict[str, float],
        total_samples: int = 100000,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = weights
        self.total_samples = total_samples

        # Validate weights sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6, (
            f"Weights must sum to 1, got {total_weight}"
        )

        # Calculate samples per dataset
        self.samples_per_dataset = {
            name: int(total_samples * weight) for name, weight in weights.items()
        }

        # Adjust for rounding
        current_total = sum(self.samples_per_dataset.values())
        diff = total_samples - current_total
        if diff > 0:
            # Add to largest dataset
            largest = max(self.samples_per_dataset, key=self.samples_per_dataset.get)
            self.samples_per_dataset[largest] += diff

        # Create index mapping
        self.rng = random.Random(seed)
        self.indices = self._create_indices()

    def _create_indices(self) -> List[tuple]:
        """Create list of (dataset_name, index) tuples"""
        indices = []
        for name, num_samples in self.samples_per_dataset.items():
            dataset_len = len(self.datasets[name])
            # Sample with replacement if needed
            for _ in range(num_samples):
                idx = self.rng.randint(0, dataset_len - 1)
                indices.append((name, idx))

        self.rng.shuffle(indices)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        dataset_name, sample_idx = self.indices[idx]
        sample = self.datasets[dataset_name][sample_idx]
        sample["domain"] = dataset_name
        return sample

    def get_domain_stats(self) -> Dict:
        """Get statistics about dataset composition"""
        return {
            "total_samples": self.total_samples,
            "weights": self.weights,
            "samples_per_dataset": self.samples_per_dataset,
        }


class _SyntheticDatasetWrapper(Dataset):
    """Wraps synthetic data samples into a PyTorch Dataset"""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: Optional[Callable] = None,
        max_length: int = 1024,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text = self.samples[idx]["text"]

        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length)
            input_ids = tokens["input_ids"]
        else:
            input_ids = [ord(c) % 256 for c in text[: self.max_length]]
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        labels = input_ids[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_composite_dataset(
    code_path: Optional[str] = None,
    math_path: Optional[str] = None,
    logic_path: Optional[str] = None,
    language_path: Optional[str] = None,
    weights: Dict[str, float] = None,
    max_length: int = 1024,
    total_samples: int = 100000,
    tokenizer: Optional[Callable] = None,
    seed: int = 42,
) -> WeightedDataset:
    """
    Create composite dataset with specified composition.

    Default weights from paper:
        Code: 40%, Math: 25%, Logic: 20%, Language: 15%

    Args:
        code_path: Path to code dataset
        math_path: Path to math dataset
        logic_path: Path to logic dataset
        language_path: Path to language dataset
        weights: Custom weights (dict with keys: code, math, logic, language)
        max_length: Maximum sequence length
        total_samples: Total number of samples to generate
        tokenizer: Tokenizer function
        seed: Random seed

    Returns:
        WeightedDataset
    """
    if weights is None:
        weights = {
            "code": 0.40,
            "math": 0.25,
            "logic": 0.20,
            "language": 0.15,
        }

    datasets = {}

    # Only include datasets with provided paths
    if code_path:
        datasets["code"] = CodeDataset(
            code_path, tokenizer=tokenizer, max_length=max_length
        )
    if math_path:
        datasets["math"] = MathDataset(
            math_path, tokenizer=tokenizer, max_length=max_length
        )
    if logic_path:
        datasets["logic"] = LogicDataset(
            logic_path, tokenizer=tokenizer, max_length=max_length
        )
    if language_path:
        datasets["language"] = LanguageDataset(
            language_path, tokenizer=tokenizer, max_length=max_length
        )

    # Fall back to synthetic data when no paths provided
    if not datasets:
        from ..data.synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=seed)
        samples_per_domain = max(100, total_samples // len(weights))

        for domain in weights:
            if domain == "code":
                samples = generator.generate_code_samples(samples_per_domain)
            elif domain == "math":
                samples = generator.generate_math_samples(samples_per_domain)
            elif domain == "logic":
                samples = generator.generate_logic_samples(samples_per_domain)
            else:
                samples = generator.generate_language_samples(samples_per_domain)

            datasets[domain] = _SyntheticDatasetWrapper(samples, tokenizer, max_length)

    # Normalize weights for available datasets
    available_weights = {k: weights[k] for k in datasets.keys()}
    total = sum(available_weights.values())
    normalized_weights = {k: v / total for k, v in available_weights.items()}

    return WeightedDataset(datasets, normalized_weights, total_samples, seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader with default settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    result = {
        "input_ids": input_ids,
        "labels": labels,
    }

    # Include domain if available
    if "domain" in batch[0]:
        result["domain"] = [item["domain"] for item in batch]

    return result


def create_dataloaders_for_phases(
    data_paths: Dict[str, str],
    phase1_samples: int = 10000,
    phase2_samples: int = 90000,
    phase3_samples: int = 10000,
    batch_size: int = 32,
    max_length: int = 1024,
    tokenizer: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for different training phases.

    Phase 1: Language + syntax (focus on language dataset)
    Phase 2: Reasoning + structured tasks (code, math, logic)
    Phase 3: Multi-trajectory consistency (full mix)

    Args:
        data_paths: Dict with keys 'code', 'math', 'logic', 'language'
        phase1_samples: Number of samples for phase 1
        phase2_samples: Number of samples for phase 2
        phase3_samples: Number of samples for phase 3
        batch_size: Batch size
        max_length: Maximum sequence length
        tokenizer: Tokenizer function
        seed: Random seed

    Returns:
        Dict of dataloaders for each phase
    """
    dataloaders = {}

    # Phase 1: Focus on language
    phase1_weights = {
        "code": 0.05,
        "math": 0.05,
        "logic": 0.05,
        "language": 0.85,
    }
    phase1_dataset = create_composite_dataset(
        **data_paths,
        weights=phase1_weights,
        max_length=max_length,
        total_samples=phase1_samples,
        tokenizer=tokenizer,
        seed=seed,
    )
    dataloaders["phase1"] = create_dataloader(phase1_dataset, batch_size=batch_size)

    # Phase 2: Reasoning focus
    phase2_weights = {
        "code": 0.45,
        "math": 0.30,
        "logic": 0.20,
        "language": 0.05,
    }
    phase2_dataset = create_composite_dataset(
        **data_paths,
        weights=phase2_weights,
        max_length=max_length,
        total_samples=phase2_samples,
        tokenizer=tokenizer,
        seed=seed + 1,
    )
    dataloaders["phase2"] = create_dataloader(phase2_dataset, batch_size=batch_size)

    # Phase 3: Full mix
    phase3_weights = {
        "code": 0.40,
        "math": 0.25,
        "logic": 0.20,
        "language": 0.15,
    }
    phase3_dataset = create_composite_dataset(
        **data_paths,
        weights=phase3_weights,
        max_length=max_length,
        total_samples=phase3_samples,
        tokenizer=tokenizer,
        seed=seed + 2,
    )
    dataloaders["phase3"] = create_dataloader(phase3_dataset, batch_size=batch_size)

    return dataloaders
