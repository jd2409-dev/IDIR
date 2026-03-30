"""Synthetic Data Generator for IDIR-KS

Provides fallback data when HuggingFace datasets are not available.
Generates synthetic code, math, logic, and language samples.
"""

import random
from typing import List, Dict


class SyntheticDataGenerator:
    """Generate synthetic training data"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_code_samples(self, n: int) -> List[Dict]:
        """Generate synthetic code samples"""
        samples = []

        templates = [
            # Python function templates
            "def {name}({params}):\n    {body}\n    return {return_val}",
            "class {name}:\n    def __init__(self, {params}):\n        {body}",
            "for {var} in range({n}):\n    {action}",
            "if {condition}:\n    {action1}\nelse:\n    {action2}",
            "while {condition}:\n    {body}",
            # Algorithm patterns
            "def sort_{name}(arr):\n    for i in range(len(arr)):\n        {sort_logic}\n    return arr",
            "def search_{name}(target, arr):\n    {search_logic}\n    return {result}",
        ]

        names = [
            "foo",
            "bar",
            "baz",
            "helper",
            "process",
            "calculate",
            "find",
            "compute",
        ]
        params = ["x", "y", "data", "items", "value", "array", "list", "obj"]
        actions = ["print(x)", "x += 1", "result.append(x)", "process(x)", "yield x"]
        conditions = ["x > 0", "len(arr) > 0", "x is not None", "True", "i < n"]

        for i in range(n):
            template = self.rng.choice(templates)

            sample = template.format(
                name=self.rng.choice(names) + f"_{i}",
                params=", ".join(self.rng.sample(params, self.rng.randint(1, 3))),
                body=self.rng.choice(actions),
                return_val=self.rng.choice(["x", "result", "None", "True", "0"]),
                var=self.rng.choice(["i", "j", "k", "x", "item"]),
                n=self.rng.randint(1, 100),
                action=self.rng.choice(actions),
                action1=self.rng.choice(actions),
                action2=self.rng.choice(actions),
                condition=self.rng.choice(conditions),
                sort_logic="arr[i], arr[j] = arr[j], arr[i]",
                search_logic="if target == arr[mid]: return mid",
                result="-1",
            )

            samples.append({"text": sample, "type": "code"})

        return samples

    def generate_math_samples(self, n: int) -> List[Dict]:
        """Generate synthetic math samples"""
        samples = []

        templates = [
            # Word problems
            "Problem: If {a} + {b} = {c}, what is {a} + {b} + {d}?\nSolution: {ans}",
            "Calculate: ({a} + {b}) * {c} = ?\nSolution: {ans}",
            "Find x: {a}x + {b} = {c}\nSolution: x = {ans}",
            "Problem: A train travels {d} miles in {t} hours. What is its speed?\nSolution: {ans} mph",
            "Geometry: A rectangle has length {l} and width {w}. What is its area?\nSolution: {ans}",
            # Equations
            "Solve: {a}x² + {b}x + {c} = 0\nSolution: x = {ans}",
            "Simplify: ({a}/{b}) + ({c}/{d}) = ?\nSolution: {ans}",
            "Problem: What is {n}% of {total}?\nSolution: {ans}",
        ]

        for i in range(n):
            a = self.rng.randint(1, 100)
            b = self.rng.randint(1, 100)
            c = self.rng.randint(1, 200)
            d = self.rng.randint(1, 50)

            template = self.rng.choice(templates)

            if "x²" in template:
                ans = (
                    "complex roots"
                    if (b**2 - 4 * a * c) < 0
                    else str((-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a))
                )
            elif "speed" in template:
                ans = str(d // t) if "t" in dir() else str(self.rng.randint(20, 100))
            elif "area" in template:
                l, w = self.rng.randint(2, 20), self.rng.randint(2, 20)
                ans = str(l * w)
            elif "Find x" in template:
                ans = str((c - b) / a) if a != 0 else "undefined"
            elif "Simplify" in template:
                ans = str((a * d + c * b) / (b * d))
            elif "%" in template:
                n = self.rng.randint(1, 100)
                total = self.rng.randint(100, 1000)
                ans = str(n * total // 100)
            else:
                ans = str(a + b + d) if "what is" in template else str((a + b) * c)

            sample = template.format(
                a=a,
                b=b,
                c=c,
                d=d,
                ans=ans,
                t=self.rng.randint(1, 10),
                l=self.rng.randint(1, 20),
                w=self.rng.randint(1, 20),
                n=self.rng.randint(1, 100),
                total=self.rng.randint(100, 1000),
            )

            samples.append({"text": sample, "type": "math"})

        return samples

    def generate_logic_samples(self, n: int) -> List[Dict]:
        """Generate synthetic logic samples"""
        samples = []

        # Logical reasoning templates
        templates = [
            "Premise: {premise1}\nPremise: {premise2}\nConclusion: {conclusion}\nIs this valid? {answer}",
            "If {condition}, then {result}. Given that {given}, what follows? {answer}",
            "All {group1} are {property}. {item} is a {group2}. Therefore, {conclusion}\nValid? {answer}",
            "Question: {question}\nOptions: {options}\nCorrect answer: {answer}",
            "Syllogism: {major}. {minor}. Therefore: {conclusion}",
            "Puzzle: {puzzle}\nSolution: {solution}",
        ]

        premises = [
            "All humans are mortal",
            "Socrates is human",
            "All birds can fly",
            "Penguins are birds",
            "If it rains, the ground is wet",
            "The ground is wet",
        ]

        conditions = [
            "it is raining",
            "x is greater than 5",
            "the switch is on",
            "all premises are true",
        ]

        results = [
            "the ground is wet",
            "y equals 10",
            "the light is on",
            "the conclusion follows",
        ]

        questions = [
            "What is the next number in the sequence: 2, 4, 8, 16, ?",
            "If A implies B, and B implies C, what can we conclude?",
            "Which of these is not like the others?",
            "Complete the analogy: Book is to reading as {a} is to {b}.",
        ]

        puzzles = [
            "Three people: Alice, Bob, and Carol. One always tells the truth, one always lies, and one alternates. Alice says 'Bob is a liar'. Bob says 'Carol tells the truth'. Who is the truth-teller?",
            "You have 8 balls, one is heavier. Find it in minimum weighings.",
            "A farmer needs to cross a river with a fox, chicken, and grain.",
        ]

        for i in range(n):
            template = self.rng.choice(templates)

            if "Puzzle" in template:
                sample = template.format(
                    puzzle=self.rng.choice(puzzles),
                    solution="See logical reasoning above",
                )
            elif "Question" in template:
                q = self.rng.choice(questions)
                if "number" in q:
                    opts = ["24", "32", "30", "64"]
                    ans = "32"
                elif "implies" in q:
                    opts = ["A implies C", "B implies A", "Nothing", "C implies A"]
                    ans = "A implies C"
                else:
                    opts = ["Option A", "Option B", "Option C", "Option D"]
                    ans = self.rng.choice(opts)

                sample = template.format(
                    question=q, options=", ".join(opts), answer=ans
                )
            else:
                sample = template.format(
                    premise1=self.rng.choice(premises),
                    premise2=self.rng.choice(premises),
                    conclusion="Therefore, the conclusion follows"
                    if self.rng.random() > 0.5
                    else "Therefore, nothing follows",
                    condition=self.rng.choice(conditions),
                    result=self.rng.choice(results),
                    given=self.rng.choice(conditions),
                    answer="Yes" if self.rng.random() > 0.3 else "No",
                    group1="mammals",
                    group2="mammal",
                    property="warm-blooded",
                    item="whale",
                )

            samples.append({"text": sample, "type": "logic"})

        return samples

    def generate_language_samples(self, n: int) -> List[Dict]:
        """Generate synthetic language samples"""
        samples = []

        # Text generation templates
        templates = [
            "Once upon a time, {setting}. {protagonist} wanted to {goal}. {obstacle}. {resolution}.",
            "The {adjective} {noun} {verb} through the {place}. It was looking for {object}.",
            "In the year {year}, scientists discovered {discovery}. This led to {consequence}.",
            "{person} said: '{quote}' Everyone {reaction}.",
            "Recipe for {food}:\n1. {step1}\n2. {step2}\n3. {step3}\n4. Enjoy!",
            "Article: {topic}\n\n{intro}\n\n{body}\n\n{conclusion}",
        ]

        settings = [
            "in a small village surrounded by mountains",
            "long ago in a kingdom far away",
            "in the bustling city of tomorrow",
            "on a spaceship traveling to Mars",
            "in a world where magic existed",
        ]

        protagonists = [
            "a young inventor",
            "an old wizard",
            "a curious cat",
            "a brave astronaut",
            "a clever robot",
        ]

        goals = [
            "find the lost treasure",
            "save the kingdom",
            "explore new worlds",
            "solve the mystery",
            "build a better future",
        ]

        obstacles = [
            "But a great storm blocked the path",
            "However, an evil sorcerer stood in the way",
            "But time was running out",
            "However, no one believed it was possible",
            "But the answer was hidden deep",
        ]

        resolutions = [
            "Through courage and wisdom, they succeeded",
            "Working together, they overcame all challenges",
            "In the end, everything worked out",
            "And so, the journey continued",
        ]

        for i in range(n):
            template = self.rng.choice(templates)

            if "Once upon" in template:
                sample = template.format(
                    setting=self.rng.choice(settings),
                    protagonist=self.rng.choice(protagonists),
                    goal=self.rng.choice(goals),
                    obstacle=self.rng.choice(obstacles),
                    resolution=self.rng.choice(resolutions),
                )
            elif "The" in template:
                sample = template.format(
                    adjective=self.rng.choice(
                        ["brave", "small", "mysterious", "ancient", "clever"]
                    ),
                    noun=self.rng.choice(
                        ["fox", "robot", "explorer", "scientist", "artist"]
                    ),
                    verb=self.rng.choice(
                        ["walked", "flew", "journeyed", "searched", "dreamed"]
                    ),
                    place=self.rng.choice(
                        ["forest", "city", "mountains", "ocean", "stars"]
                    ),
                    object=self.rng.choice(
                        ["knowledge", "treasure", "peace", "answers", "home"]
                    ),
                )
            elif "year" in template:
                sample = template.format(
                    year=self.rng.randint(2024, 2150),
                    discovery=self.rng.choice(
                        [
                            "a new form of energy",
                            "intelligent life on another planet",
                            "the secret to eternal youth",
                            "how to travel through time",
                        ]
                    ),
                    consequence=self.rng.choice(
                        [
                            "a new era of prosperity",
                            "profound changes in society",
                            "unexpected challenges",
                            "hope for the future",
                        ]
                    ),
                )
            elif "said" in template:
                sample = template.format(
                    person=self.rng.choice(
                        [
                            "The teacher",
                            "A wise elder",
                            "The president",
                            "A child",
                            "A scientist",
                        ]
                    ),
                    quote=self.rng.choice(
                        [
                            "We must work together to solve this.",
                            "The future is what we make it.",
                            "Knowledge is the key to everything.",
                            "Never give up on your dreams.",
                        ]
                    ),
                    reaction=self.rng.choice(
                        [
                            "nodded in agreement",
                            "applauded",
                            "began to think",
                            "felt inspired",
                        ]
                    ),
                )
            elif "Recipe" in template:
                sample = template.format(
                    food=self.rng.choice(
                        ["pasta", "soup", "cake", "stir-fry", "bread"]
                    ),
                    step1=self.rng.choice(
                        [
                            "Gather ingredients",
                            "Preheat oven",
                            "Prepare vegetables",
                            "Mix dry ingredients",
                        ]
                    ),
                    step2=self.rng.choice(
                        [
                            "Combine everything",
                            "Cook on medium heat",
                            "Let it rest",
                            "Add seasoning",
                        ]
                    ),
                    step3=self.rng.choice(
                        [
                            "Taste and adjust",
                            "Serve hot",
                            "Let cool",
                            "Garnish and serve",
                        ]
                    ),
                )
            else:
                sample = template.format(
                    topic=self.rng.choice(
                        [
                            "The Future of AI",
                            "Climate Change",
                            "Space Exploration",
                            "Education",
                        ]
                    ),
                    intro=self.rng.choice(
                        [
                            "This is a topic of great importance.",
                            "Recent developments have changed everything.",
                            "Experts are divided on this issue.",
                        ]
                    ),
                    body=self.rng.choice(
                        [
                            "Research shows mixed results.",
                            "The evidence is compelling.",
                            "More study is needed.",
                        ]
                    ),
                    conclusion=self.rng.choice(
                        [
                            "In conclusion, the future is bright.",
                            "Overall, challenges remain but progress continues.",
                            "To summarize, we must act now.",
                        ]
                    ),
                )

            samples.append({"text": sample, "type": "language"})

        return samples
