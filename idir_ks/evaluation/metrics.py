"""Evaluation Metrics for IDIR-KS"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time
import math


class Evaluator:
    """
    Comprehensive evaluation metrics for IDIR-KS.

    Metrics (from paper):
    1. Perplexity
    2. GSM8K accuracy (math reasoning)
    3. MBPP pass@k (coding)
    4. Logical reasoning accuracy
    5. Training stability (loss variance)
    6. Throughput (tokens/sec)
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)

    def evaluate_perplexity(self, dataloader) -> float:
        """
        Calculate perplexity on validation set.

        Perplexity = exp(average_cross_entropy_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)

                # Calculate loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

                # Count valid tokens
                valid_tokens = (labels != -100).sum().item()

                total_loss += loss.item()
                total_tokens += valid_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return perplexity

    def evaluate_gsm8k(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate on GSM8K math reasoning benchmark.

        Args:
            test_data: List of {'question': str, 'answer': str} dicts

        Returns:
            Dictionary with accuracy and details
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for item in test_data:
                question = item["question"]
                expected_answer = item["answer"]

                # Tokenize and generate
                # (Simplified - would need proper tokenizer)
                generated = self._generate_simple(question)

                # Check if correct (simplified matching)
                if expected_answer in generated or self._extract_number(
                    generated
                ) == self._extract_number(expected_answer):
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def evaluate_mbpp(self, test_data: List[Dict], k: int = 1) -> Dict:
        """
        Evaluate on MBPP coding benchmark (Mostly Basic Python Programming).

        Args:
            test_data: List of {'prompt': str, 'test_cases': List[str]} dicts
            k: Number of samples for pass@k metric

        Returns:
            Dictionary with pass@k metrics
        """
        self.model.eval()
        passed = 0
        total = 0

        with torch.no_grad():
            for item in test_data:
                prompt = item["prompt"]
                test_cases = item["test_cases"]

                # Generate k samples
                samples = []
                for _ in range(k):
                    code = self._generate_simple(prompt)
                    samples.append(code)

                # Check if any sample passes tests
                # (Simplified - would need actual Python execution)
                any_passed = self._check_code_passes_tests(samples, test_cases)

                if any_passed:
                    passed += 1
                total += 1

        pass_at_k = passed / total if total > 0 else 0.0

        return {
            f"pass@{k}": pass_at_k,
            "passed": passed,
            "total": total,
        }

    def evaluate_logical_reasoning(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate on logical reasoning tasks.

        Args:
            test_data: List of logical reasoning examples

        Returns:
            Dictionary with accuracy
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for item in test_data:
                prompt = item["prompt"]
                expected_answer = item["answer"]

                generated = self._generate_simple(prompt)

                if expected_answer.lower() in generated.lower():
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def evaluate_training_stability(self, loss_history: List[float]) -> Dict:
        """
        Evaluate training stability via loss variance.

        Args:
            loss_history: List of training losses

        Returns:
            Dictionary with stability metrics
        """
        if len(loss_history) < 2:
            return {"variance": 0.0, "std": 0.0}

        losses = torch.tensor(loss_history)
        variance = torch.var(losses).item()
        std = torch.std(losses).item()

        # Check for instability (high variance)
        is_stable = variance < 0.1  # Threshold

        return {
            "variance": variance,
            "std": std,
            "is_stable": is_stable,
        }

    def evaluate_throughput(
        self, batch_size: int = 32, seq_len: int = 1024, num_iterations: int = 10
    ) -> float:
        """
        Measure throughput in tokens/sec.

        Args:
            batch_size: Batch size for test
            seq_len: Sequence length for test
            num_iterations: Number of iterations to average

        Returns:
            Throughput in tokens/sec
        """
        self.model.eval()

        # Warmup
        dummy_input = torch.randint(
            0, self.model.vocab_size, (batch_size, seq_len), device=self.device
        )
        with torch.no_grad():
            _ = self.model(dummy_input)

        # Measure
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()

        end_time = time.time()

        total_tokens = batch_size * seq_len * num_iterations
        total_time = end_time - start_time
        throughput = total_tokens / total_time

        return throughput

    def evaluate_all(self, test_dataloaders: Dict, test_datasets: Dict = None) -> Dict:
        """
        Run all evaluations.

        Args:
            test_dataloaders: Dict of dataloaders for different tasks
            test_datasets: Dict of test datasets for specialized tasks

        Returns:
            Dictionary of all metrics
        """
        results = {}

        print("Evaluating perplexity...")
        if "val" in test_dataloaders:
            results["perplexity"] = self.evaluate_perplexity(test_dataloaders["val"])

        print("Evaluating throughput...")
        results["throughput"] = self.evaluate_throughput()

        if test_datasets:
            if "gsm8k" in test_datasets:
                print("Evaluating GSM8K...")
                results["gsm8k"] = self.evaluate_gsm8k(test_datasets["gsm8k"])

            if "mbpp" in test_datasets:
                print("Evaluating MBPP...")
                results["mbpp"] = self.evaluate_mbpp(test_datasets["mbpp"])

            if "logical" in test_datasets:
                print("Evaluating logical reasoning...")
                results["logical"] = self.evaluate_logical_reasoning(
                    test_datasets["logical"]
                )

        return results

    def _generate_simple(self, prompt: str, max_tokens: int = 100) -> str:
        """Simplified generation for evaluation (would use real tokenizer)"""
        # Placeholder - would need proper tokenization
        return "[Generated output for: " + prompt[:50] + "...]"

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text for math evaluation"""
        import re

        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return float(numbers[-1])
        return None

    def _check_code_passes_tests(
        self, code_samples: List[str], test_cases: List[str]
    ) -> bool:
        """Check if any code sample passes test cases"""
        # Simplified - would need actual Python execution in sandbox
        # For now, just check if code looks reasonable
        for code in code_samples:
            if "def " in code and "return" in code:
                return True
        return False


def format_results(results: Dict) -> str:
    """Format evaluation results as string"""
    lines = ["\n" + "=" * 60]
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)

    for metric, value in results.items():
        if isinstance(value, dict):
            lines.append(f"\n{metric.upper()}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{metric}: {value}")

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)
