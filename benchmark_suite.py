import torch
import numpy as np
import random
from typing import Dict, List, Union
from idir_model import IDIRModel
from transformers import GPT2Tokenizer
from datasets import load_dataset

class BenchmarkSuite:
    """Reasoning benchmark suite for IDIR"""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gsm8k = iter(load_dataset("gsm8k", "main", split="test"))

    def _generate_batch(self, prompts: List[str], max_new_tokens: int = 20) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, early_stopping=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def six_digit_addition(self, num_samples: int = 100) -> Dict[str, Union[str, float]]:
        print("Running Six-Digit Addition Benchmark...")
        prompts, solutions = [], []
        for _ in range(num_samples):
            a, b = np.random.randint(100000, 999999), np.random.randint(100000, 999999)
            prompts.append(f"ADD {a} {b} =")
            solutions.append(str(a + b))
        
        generated = self._generate_batch(prompts)
        correct = sum(1 for gen, sol in zip(generated, solutions) if gen.split('=')[-1].strip() == sol)
        accuracy = (correct / num_samples) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "six_digit_addition", "accuracy": accuracy}

    def gsm_math_problems(self, num_samples: int = 50) -> Dict[str, Union[str, float]]:
        print("Running GSM Math Problems Benchmark...")
        prompts, solutions = [], []
        for _ in range(num_samples):
            try:
                sample = next(self.gsm8k)
                prompts.append(f"SOLVE: {sample['question']} \nANSWER:")
                solutions.append(sample['answer'].split("####")[-1].strip())
            except StopIteration:
                break # Not enough samples in dataset
        
        generated = self._generate_batch(prompts, max_new_tokens=50)
        correct = sum(1 for gen, sol in zip(generated, solutions) if gen.split('ANSWER:')[-1].strip() == sol)
        accuracy = (correct / len(prompts)) * 100 if prompts else 0
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "gsm_math", "accuracy": accuracy}

    def multi_hop_logic_chains(self, num_samples: int = 100) -> Dict[str, Union[str, float]]:
        print("Running Multi-Hop Logic Chains Benchmark...")
        # A is related to B, B to C, C to D. What is the relation between A and D?
        prompts, solutions = [], []
        for _ in range(num_samples):
            # Complex relations
            relations = [("is taller than", ">"), ("is shorter than", "<"), ("weighs the same as", "=")]
            names = ["Alex", "Ben", "Chris", "David"]
            random.shuffle(names)
            r1, s1 = random.choice(relations)
            r2, s2 = random.choice(relations)
            r3, s3 = random.choice(relations)
            prompt = f"{names[0]} {r1} {names[1]}. {names[1]} {r2} {names[2]}. {names[2]} {r3} {names[3]}. What is the relation between {names[0]} and {names[3]}?"
            
            # Simple transitive logic for this setup, assuming a linear scale
            final_relation = eval(f"1 {s1} 0.5 {s2} 0.25 {s3} 0.125") # A simple way to get a boolean
            if final_relation: solution = "is taller than"
            else: solution = "is shorter than"

            prompts.append(prompt)
            solutions.append(solution)

        generated = self._generate_batch(prompts)
        correct = sum(1 for gen, sol in zip(generated, solutions) if sol in gen)
        accuracy = (correct / num_samples) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "multi_hop_logic", "accuracy": accuracy}

    def parentheses_matching(self, depth: int = 15, num_samples: int = 100) -> Dict[str, Union[str, float]]:
        print("Running Parentheses Matching Benchmark...")
        prompts, solutions = [], []
        for _ in range(num_samples):
            sequence = ""
            stack = []
            for _ in range(depth * 2):
                if random.random() > 0.5 and len(stack) > 0:
                    sequence += stack.pop()
                else:
                    char = random.choice(['(', '{', '['])
                    sequence += char
                    if char == '(': stack.append(')')
                    if char == '{': stack.append('}')
                    if char == '[': stack.append(']')
            is_valid = "YES" if not stack and sequence else "NO"
            prompts.append(f"CHECK: {sequence} - Valid?")
            solutions.append(is_valid)
            
        generated = self._generate_batch(prompts, max_new_tokens=3)
        correct = sum(1 for gen, sol in zip(generated, solutions) if sol in gen)
        accuracy = (correct / num_samples) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "parentheses_matching", "accuracy": accuracy}

    def symbol_rewriting(self, num_samples: int = 100) -> Dict[str, Union[str, float]]:
        print("Running Symbol Rewriting Benchmark...")
        prompts, solutions = [], []
        for _ in range(num_samples):
            alphabet = "ABCDE"
            symbols = "".join(np.random.choice(list(alphabet), size=7, replace=True))
            rule = np.random.choice(["reverse", "sort", "double", "remove_first_A"])
            prompt = f"REWRITE: {symbols} using {rule} ->"
            if rule == "reverse": solution = symbols[::-1]
            elif rule == "sort": solution = "".join(sorted(symbols))
            elif rule == "double": solution = "".join([c+c for c in symbols])
            elif rule == "remove_first_A": solution = symbols.replace('A', '', 1)
            prompts.append(prompt)
            solutions.append(solution)
        
        generated = self._generate_batch(prompts)
        correct = sum(1 for gen, sol in zip(generated, solutions) if sol in gen.split('->')[-1].strip())
        accuracy = (correct / num_samples) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "symbol_rewriting", "accuracy": accuracy}

    def simple_code_generation(self, num_samples: int = 50) -> Dict[str, Union[str, float]]:
        print("Running Simple Code Generation Benchmark...")
        prompts = [
            'def add(a, b):\n    """Adds two numbers."""',
            'def subtract(a, b):\n    """Subtracts two numbers."""',
            'def multiply(a, b):\n    """Multiplies two numbers."""',
            'def divide(a, b):\n    """Divides two numbers."""'
        ]
        solutions = [
            'return a + b', 'return a - b', 'return a * b', 'return a / b'
        ]
        
        chosen_prompts = [random.choice(prompts) for _ in range(num_samples)]
        chosen_solutions = [solutions[prompts.index(p)] for p in chosen_prompts]

        generated = self._generate_batch(chosen_prompts, max_new_tokens=10)
        correct = sum(1 for gen, sol in zip(generated, chosen_solutions) if sol in gen)
        accuracy = (correct / num_samples) * 100
        print(f"  Accuracy: {accuracy:.2f}%")
        return {"benchmark": "simple_code_generation", "accuracy": accuracy}


    def run_all_benchmarks(self) -> List[Dict[str, Union[str, float]]]:
        print("\nRunning Complete Benchmark Suite...\n")
        results = [
            self.six_digit_addition(),
            self.gsm_math_problems(),
            self.multi_hop_logic_chains(),
            self.parentheses_matching(),
            self.symbol_rewriting(),
            self.simple_code_generation(),
        ]
        avg_accuracy = np.mean([r["accuracy"] for r in results if "accuracy" in r])
        print(f"\nOverall Average Accuracy: {avg_accuracy:.2f}%")
        return results

if __name__ == "__main__":
    model = IDIRModel(vocab_size=50257, d=512)
    benchmark = BenchmarkSuite(model, device="cpu")
    results = benchmark.run_all_benchmarks()
    print("\nBenchmark Results:")
    for result in results:
        print(f"  {result['benchmark']}: {result['accuracy']:.2f}%")
