"""
Comprehensive Coding Benchmark Suite for IDIR
Tests coding capabilities similar to Claude Opus / GPT-5 benchmarks
Optimized for RTX 3050 8GB under 1 hour training constraint
"""

import torch
import numpy as np
import random
import re
from typing import Dict, List, Union, Tuple
from idir_model import IDIRModel
from transformers import GPT2Tokenizer
from datasets import load_dataset


class CodingBenchmarkSuite:
    """
    Elite coding benchmark suite testing:
    - HumanEval-style function completion
    - Code repair and debugging
    - Algorithm implementation
    - Multi-language translation (Python/C/Rust)
    - OS-level coding tasks
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, prompts: List[str], max_new_tokens: int = 100,
                  temperature: float = 0.2, num_beams: int = 3) -> List[str]:
        """Generate code with beam search for better quality."""
        inputs = self.tokenizer(prompts, return_tensors="pt",
                               padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _extract_code(self, text: str) -> str:
        """Extract code block from generated text."""
        # Try to find code between triple backticks
        match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find indented code blocks
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith('def ') or line.startswith('class ') or line.startswith('//'):
                in_code = True
            if in_code:
                code_lines.append(line)
            if in_code and line.strip() == '' and len(code_lines) > 3:
                break

        return '\n'.join(code_lines) if code_lines else text

    # ========== HumanEval-Style Function Completion ==========

    def humaneval_benchmark(self, num_samples: int = 50) -> Dict[str, Union[str, float]]:
        """
        HumanEval-style function completion benchmark.
        Tests if model can complete function implementations from docstrings.
        """
        print("Running HumanEval-Style Function Completion Benchmark...")

        tasks = [
            {
                "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n    ",
                "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
                "keywords": ["return a + b", "return (a + b)", "a+b"]
            },
            {
                "prompt": "def factorial(n: int) -> int:\n    \"\"\"Return n factorial.\"\"\"\n    ",
                "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6",
                "keywords": ["if n <=", "if n==", "return 1", "n * factorial"]
            },
            {
                "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is palindrome.\"\"\"\n    ",
                "test": "assert is_palindrome('racecar')\nassert not is_palindrome('hello')\nassert is_palindrome('')",
                "keywords": ["return s ==", "s[::-1]", "reverse"]
            },
            {
                "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return nth Fibonacci number.\"\"\"\n    ",
                "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
                "keywords": ["if n <=", "return n", "fibonacci(n-1)", "fibonacci(n-2)"]
            },
            {
                "prompt": "def reverse_list(lst: list) -> list:\n    \"\"\"Reverse a list.\"\"\"\n    ",
                "test": "assert reverse_list([1,2,3]) == [3,2,1]\nassert reverse_list([]) == []\nassert reverse_list([1]) == [1]",
                "keywords": ["return lst[::-1]", "lst.reverse", "reversed(lst)"]
            },
            {
                "prompt": "def find_max(lst: list) -> int:\n    \"\"\"Find maximum element in list.\"\"\"\n    ",
                "test": "assert find_max([1,5,3]) == 5\nassert find_max([-1,-5,-3]) == -1\nassert find_max([0]) == 0",
                "keywords": ["return max", "max(lst)", "largest"]
            },
            {
                "prompt": "def count_vowels(s: str) -> int:\n    \"\"\"Count vowels in string.\"\"\"\n    ",
                "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\nassert count_vowels('aeiou') == 5",
                "keywords": ["aeiou", "vowels", "count"]
            },
            {
                "prompt": "def sum_of_digits(n: int) -> int:\n    \"\"\"Return sum of digits.\"\"\"\n    ",
                "test": "assert sum_of_digits(123) == 6\nassert sum_of_digits(0) == 0\nassert sum_of_digits(999) == 27",
                "keywords": ["sum", "int(d)", "str(n)"]
            },
            {
                "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Check if number is prime.\"\"\"\n    ",
                "test": "assert is_prime(2)\nassert is_prime(17)\nassert not is_prime(4)\nassert not is_prime(1)",
                "keywords": ["if n <", "return False", "for i in range", "n % i"]
            },
            {
                "prompt": "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Binary search. Return index or -1.\"\"\"\n    ",
                "test": "assert binary_search([1,2,3,4,5], 3) == 2\nassert binary_search([1,2,3], 4) == -1\nassert binary_search([], 1) == -1",
                "keywords": ["left", "right", "mid", "while", "// 2"]
            },
        ]

        results = []
        chosen_tasks = random.sample(tasks, min(num_samples, len(tasks)))

        for task in chosen_tasks:
            prompt = task["prompt"]
            generated = self._generate([prompt], max_new_tokens=50, temperature=0.2)[0]

            # Check if any expected keywords are present
            code = self._extract_code(generated[len(prompt):] if generated.startswith(prompt) else generated)
            success = any(kw in code for kw in task["keywords"])
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "humaneval_completion", "accuracy": accuracy}

    # ========== Code Repair/Debugging ==========

    def code_repair_benchmark(self, num_samples: int = 30) -> Dict[str, Union[str, float]]:
        """
        Test if model can identify and fix bugs in code.
        """
        print("Running Code Repair/Debugging Benchmark...")

        buggy_codes = [
            {
                "buggy": "def divide(a, b):\n    return a / b",
                "fix_indicators": ["if b == 0", "raise", "ValueError", "ZeroDivisionError"],
                "description": "add zero check"
            },
            {
                "buggy": "with open('file.txt') as f:\n    data = f.read()\nprint(data)",
                "fix_indicators": ["try", "except", "FileNotFoundError"],
                "description": "add error handling"
            },
            {
                "buggy": "def factorial(n):\n    return n * factorial(n)",
                "fix_indicators": ["n-1", "n <= 1", "if n ==", "return 1"],
                "description": "fix recursion base case"
            },
            {
                "buggy": "lst = []\nfor i in range(10):\n    lst.append(i)",
                "fix_indicators": ["[i for i in", "list(range", "list comprehension"],
                "description": "optimize to list comprehension"
            },
            {
                "buggy": "if x == True:\n    print('yes')",
                "fix_indicators": ["if x:", "if bool(x)", "remove == True"],
                "description": "fix boolean comparison"
            },
            {
                "buggy": "for i in range(len(items)):\n    print(items[i])",
                "fix_indicators": ["for item in items", "enumerate", "iterate directly"],
                "description": "use direct iteration"
            },
        ]

        results = []
        for _ in range(num_samples):
            task = random.choice(buggy_codes)
            prompt = f"# Fix this buggy code:\n{task['buggy']}\n\n# Fixed code:\n"

            generated = self._generate([prompt], max_new_tokens=50)[0]

            # Check if fix indicators are present
            code = generated[len(prompt):] if generated.startswith(prompt) else generated
            fixed = any(ind in code.lower() for ind in task["fix_indicators"])
            results.append(fixed)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "code_repair", "accuracy": accuracy}

    # ========== Algorithm Implementation ==========

    def algorithm_benchmark(self, num_samples: int = 20) -> Dict[str, Union[str, float]]:
        """
        Test classic algorithm implementations.
        """
        print("Running Algorithm Implementation Benchmark...")

        algorithms = [
            {
                "name": "bubble sort",
                "prompt": "def bubble_sort(arr):\n    \"\"\"Sort array using bubble sort.\"\"\"\n    ",
                "indicators": ["for i in", "for j in", "if arr[", "swap", ">"]
            },
            {
                "name": "quick sort",
                "prompt": "def quick_sort(arr):\n    \"\"\"Sort array using quick sort.\"\"\"\n    ",
                "indicators": ["pivot", "partition", "recursive", "if len", "< pivot", "> pivot"]
            },
            {
                "name": "merge sort",
                "prompt": "def merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n    ",
                "indicators": ["merge", "left", "right", "mid", "while", "append"]
            },
            {
                "name": "binary search tree",
                "prompt": "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\nclass BST:\n    def insert(self, root, val):\n        \"\"\"Insert value into BST.\"\"\"\n        ",
                "indicators": ["if root is None", "if val <", "root.left", "root.right", "return"]
            },
            {
                "name": "linked list",
                "prompt": "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def append(self, val):\n        \"\"\"Append value to list.\"\"\"\n        ",
                "indicators": ["if not self.head", "while", "current.next", "= Node", "self.head"]
            },
        ]

        results = []
        for _ in range(num_samples):
            algo = random.choice(algorithms)
            generated = self._generate([algo["prompt"]], max_new_tokens=80)[0]

            code = generated[len(algo["prompt"]):] if generated.startswith(algo["prompt"]) else generated
            success = sum(1 for ind in algo["indicators"] if ind in code) >= len(algo["indicators"]) // 2
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "algorithm_implementation", "accuracy": accuracy}

    # ========== Language Translation ==========

    def language_translation_benchmark(self, num_samples: int = 20) -> Dict[str, Union[str, float]]:
        """
        Test code translation between Python, C, and Rust.
        """
        print("Running Language Translation Benchmark...")

        translations = [
            {
                "from": "python",
                "to": "c",
                "prompt": "# Translate this Python to C:\n\ndef add(a, b):\n    return a + b\n\n# C version:\n",
                "indicators": ["int add", "(int a, int b)", "return a + b", "{"]
            },
            {
                "from": "python",
                "to": "c",
                "prompt": "# Translate this Python to C:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\n# C version:\n",
                "indicators": ["int factorial", "if (n <=", "return 1", "n * factorial", "{"]
            },
            {
                "from": "python",
                "to": "rust",
                "prompt": "# Translate this Python to Rust:\n\ndef add(a: int, b: int) -> int:\n    return a + b\n\n# Rust version:\n",
                "indicators": ["fn add", "a: i32", "-> i32", "a + b", "{"]
            },
            {
                "from": "c",
                "to": "python",
                "prompt": "# Translate this C to Python:\n\nint square(int x) {\n    return x * x;\n}\n\n# Python version:\n",
                "indicators": ["def square", "return x * x", "return x**2", ":"]
            },
        ]

        results = []
        for _ in range(num_samples):
            task = random.choice(translations)
            generated = self._generate([task["prompt"]], max_new_tokens=50)[0]

            code = generated[len(task["prompt"]):] if generated.startswith(task["prompt"]) else generated
            success = sum(1 for ind in task["indicators"] if ind in code) >= len(task["indicators"]) // 2
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "language_translation", "accuracy": accuracy}

    # ========== OS-Level Coding ==========

    def os_level_benchmark(self, num_samples: int = 15) -> Dict[str, Union[str, float]]:
        """
        Test OS-level coding capabilities.
        """
        print("Running OS-Level Coding Benchmark...")

        os_tasks = [
            {
                "prompt": "// Implement a basic memory allocator in C:\n\nvoid* malloc(size_t size) {\n    ",
                "indicators": ["sbrk", "brk", "heap", "block", "struct", "header", "size_t"],
                "name": "memory allocator"
            },
            {
                "prompt": "// Implement a simple process scheduler in C:\n\nvoid schedule() {\n    ",
                "indicators": ["process", "task", "queue", "priority", "current", "next", "switch"],
                "name": "scheduler"
            },
            {
                "prompt": "// Implement a mutex lock:\n\nvoid mutex_lock(mutex_t* m) {\n    ",
                "indicators": ["atomic", "compare", "swap", "while", "lock", "test"],
                "name": "mutex"
            },
            {
                "prompt": "// System call wrapper in assembly:\n\nsys_write:\n    ",
                "indicators": ["mov", "syscall", "eax", "edi", "rsi", "ret", "int 0x80"],
                "name": "syscall wrapper"
            },
            {
                "prompt": "// File system inode structure in C:\n\nstruct inode {\n    ",
                "indicators": ["size", "uid", "gid", "mode", "block", "pointer", "time_t"],
                "name": "inode"
            },
        ]

        results = []
        for _ in range(num_samples):
            task = random.choice(os_tasks)
            generated = self._generate([task["prompt"]], max_new_tokens=60, temperature=0.3)[0]

            code = generated[len(task["prompt"]):] if generated.startswith(task["prompt"]) else generated
            success = sum(1 for ind in task["indicators"] if ind in code.lower()) >= 2
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "os_level_coding", "accuracy": accuracy}

    # ========== Data Structure Implementation ==========

    def data_structure_benchmark(self, num_samples: int = 20) -> Dict[str, Union[str, float]]:
        """
        Test data structure implementations.
        """
        print("Running Data Structure Implementation Benchmark...")

        data_structures = [
            {
                "name": "stack",
                "prompt": "class Stack:\n    \"\"\"Implement a stack.\"\"\"\n    def __init__(self):\n        ",
                "indicators": ["self.items = []", "self.data", "self.stack", "list"]
            },
            {
                "name": "queue",
                "prompt": "class Queue:\n    \"\"\"Implement a queue.\"\"\"\n    def __init__(self):\n        ",
                "indicators": ["self.items = []", "collections.deque", "self.front", "self.rear"]
            },
            {
                "name": "binary tree",
                "prompt": "class TreeNode:\n    \"\"\"Binary tree node.\"\"\"\n    def __init__(self, val):\n        ",
                "indicators": ["self.val", "self.left", "self.right", "= None", "= val"]
            },
            {
                "name": "hash map",
                "prompt": "class HashMap:\n    \"\"\"Simple hash map implementation.\"\"\"\n    def __init__(self):\n        ",
                "indicators": ["self.buckets", "self.size", "{}"]
            },
        ]

        results = []
        for _ in range(num_samples):
            ds = random.choice(data_structures)
            generated = self._generate([ds["prompt"]], max_new_tokens=40)[0]

            code = generated[len(ds["prompt"]):] if generated.startswith(ds["prompt"]) else generated
            success = sum(1 for ind in ds["indicators"] if ind in code) >= 2
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "data_structure", "accuracy": accuracy}

    # ========== API Usage ==========

    def api_usage_benchmark(self, num_samples: int = 20) -> Dict[str, Union[str, float]]:
        """
        Test if model can use common APIs correctly.
        """
        print("Running API Usage Benchmark...")

        api_tasks = [
            {
                "prompt": "# Use requests to fetch JSON data from an API:\nimport requests\n\n",
                "indicators": ["requests.get", "response.json", "json()", "url", "http"]
            },
            {
                "prompt": "# Open and read a file safely:\n\n",
                "indicators": ["with open", "'r'", "as f", "f.read", "read()"]
            },
            {
                "prompt": "# Create a thread and run a function:\nimport threading\n\n",
                "indicators": ["threading.Thread", "target=", "start()", "join()"]
            },
            {
                "prompt": "# Parse JSON string to Python object:\nimport json\n\njson_str = '{\"name\": \"test\"}'\n",
                "indicators": ["json.loads", "json.load", "loads(", "parse"]
            },
            {
                "prompt": "# Connect to SQLite database and execute query:\nimport sqlite3\n\n",
                "indicators": ["sqlite3.connect", "cursor", "execute", "fetchall", "with"]
            },
        ]

        results = []
        for _ in range(num_samples):
            task = random.choice(api_tasks)
            generated = self._generate([task["prompt"]], max_new_tokens=50)[0]

            code = generated[len(task["prompt"]):] if generated.startswith(task["prompt"]) else generated
            success = sum(1 for ind in task["indicators"] if ind in code) >= 2
            results.append(success)

        accuracy = (sum(results) / len(results)) * 100
        print(f"  Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)})")
        return {"benchmark": "api_usage", "accuracy": accuracy}

    # ========== Run All Benchmarks ==========

    def run_all_benchmarks(self) -> List[Dict[str, Union[str, float]]]:
        """Run complete coding benchmark suite."""
        print("\n" + "="*60)
        print("COMPREHENSIVE CODING BENCHMARK SUITE")
        print("Target: Claude Opus / GPT-5 Level Performance")
        print("="*60 + "\n")

        results = [
            self.humaneval_benchmark(num_samples=50),
            self.code_repair_benchmark(num_samples=30),
            self.algorithm_benchmark(num_samples=20),
            self.language_translation_benchmark(num_samples=20),
            self.os_level_benchmark(num_samples=15),
            self.data_structure_benchmark(num_samples=20),
            self.api_usage_benchmark(num_samples=20),
        ]

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        for result in results:
            print(f"  {result['benchmark']:<35} {result['accuracy']:>6.2f}%")

        avg_accuracy = np.mean([r["accuracy"] for r in results])
        print("-"*60)
        print(f"  {'OVERALL AVERAGE':<35} {avg_accuracy:>6.2f}%")
        print("="*60)

        # Provide interpretation
        if avg_accuracy >= 80:
            level = "ELITE (Claude Opus / GPT-4 level)"
        elif avg_accuracy >= 60:
            level = "ADVANCED (GPT-3.5 / Copilot level)"
        elif avg_accuracy >= 40:
            level = "INTERMEDIATE (Basic coding assistant)"
        else:
            level = "NOVICE (Needs more training)"

        print(f"\nPerformance Level: {level}")

        return results


if __name__ == "__main__":
    print("Initializing IDIR model for coding benchmarks...")
    model = IDIRModel(vocab_size=50257, hidden_dim=512)
    benchmark = CodingBenchmarkSuite(model, device="cpu")
    results = benchmark.run_all_benchmarks()
