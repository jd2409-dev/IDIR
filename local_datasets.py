import random
from typing import Iterable, Iterator, Optional

CODE_SNIPPETS = [
    "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "def matrix_mult(a, b):\n    result = [[0] * len(b[0]) for _ in range(len(a))]\n    for i in range(len(a)):\n        for j in range(len(b[0])):\n            for k in range(len(b)):\n                result[i][j] += a[i][k] * b[k][j]\n    return result",
    "def merge_sort(lst):\n    if len(lst) <= 1:\n        return lst\n    mid = len(lst) // 2\n    left = merge_sort(lst[:mid])\n    right = merge_sort(lst[mid:])\n    result, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
]

MATH_QUESTIONS = [
    "Compute 125 * 32.",
    "A train leaves station A at 07:00 traveling at 80 km/h. Another leaves station B at 07:30 traveling at 70 km/h toward station A. They are 290 km apart. When do they meet?",
    "Find the derivative of f(x) = 3x^3 - 5x + 2.",
    "Solve the system:\n3x + 2y = 12\n5x - y = 7.",
    "Integrate 0 to 1 of (4x^3 - 2x + 6) dx.",
]

MATH_SOLUTIONS = [
    "125 * 32 = 4000.",
    "They meet 2 hours after the second train departs.",
    "f'(x) = 9x^2 - 5.",
    "Solution is x = 3, y = 1.\nCheck: 3*3 + 2*1 = 11 (typo?).",
    "Integral evaluates to x^4 - x^2 + 6x | 0^1 = 1 - 1 + 6 = 6.",
]

REASONING_PROMPTS = [
    "If every premium user is a subscriber and every subscriber gets priority support, what can you say about premium users?",
    "Given that all squares are rectangles and this shape has equal sides but not necessarily right angles, what can you conclude?",
    "If car A takes 1 hour to cover 90 km and car B takes 2 hours to cover the same distance, how much faster is car A?",
]

GENERAL_TEXT = [
    "Recent advances in machine reasoning rely on structured internal states and iterative refinement of latent representations.",
    "The Math dataset provides problems paired with worked-out solutions and follows a reasoning chain for each step.",
    "Best practices for coding include clear abstractions, docstrings, and small focused functions.",
]

FALLBACK_SAMPLES = {
    "codeparrot/github-code": CODE_SNIPPETS,
    "code_search_net": CODE_SNIPPETS,
    "gsm8k:question": MATH_QUESTIONS,
    "gsm8k:answer": MATH_SOLUTIONS,
    "math_dataset:question": MATH_QUESTIONS,
    "math_dataset:solution": MATH_SOLUTIONS,
    "openwebtext": GENERAL_TEXT,
    "wikitext:wikitext-103-raw-v1": GENERAL_TEXT,
    "allenai/c4": GENERAL_TEXT,
}


def _cycle_samples(samples: list[str]) -> Iterator[str]:
    while True:
        for sample in samples:
            yield sample


def get_local_dataset_iterable(spec: dict) -> Optional[Iterable[dict]]:
    text_field = spec.get("text_field", "text")
    key = f"{spec['path']}:{spec.get('name', text_field)}"

    samples = FALLBACK_SAMPLES.get(key)
    if samples is None:
        samples = FALLBACK_SAMPLES.get(spec['path'])
    if not samples:
        return None

    iterator = _cycle_samples(samples)

    def generator():
        for sample in iterator:
            yield {text_field: sample}

    return generator()
