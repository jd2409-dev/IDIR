# IDIR - Infinite Depth Implicit Reasoners

This repository provides a complete implementation of the **Infinite Depth Implicit Reasoner (IDIR)**, a 70M parameter language model that achieves reasoning capabilities comparable to much larger models by leveraging compute instead of parameter scaling.

## Architecture

IDIR replaces fixed architectural depth with a learned fixed-point equation solved iteratively. This allows the model's effective depth to grow dynamically with test-time compute, without increasing the number of parameters.

Key components include:
- **Contractive Core**: A fixed-point operator with spectral normalization on the hidden-to-hidden weight matrix to guarantee convergence.
- **Sparse Expert Module**: Four expert MLPs with a top-2 routing mechanism to specialize computation.
- **Differentiable Internal Reasoning**: An internal state `z` updated via a learned transformation and a `SoftTopK` operation, approximating multi-step deductive reasoning.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers
- Datasets
- Numpy

Install the required packages:
```bash
pip install torch transformers datasets numpy
```

### 1. Training the Model

The entire training process is handled by `training_curriculum.py`. This script implements the four-phase curriculum described in the paper:

1.  **Language Pretraining**: General language understanding on the WikiText dataset.
2.  **Algorithmic Curriculum**: Training on synthetic tasks like arithmetic and symbol manipulation.
3.  **Reasoning Distillation**: Learning from a larger teacher model (`gpt2-medium`).
4.  **Self-Consistency Training**: Fine-tuning to produce consistent outputs across multiple reasoning paths.

To start the training, run:
```bash
python training_curriculum.py
```
Checkpoints for each phase will be saved in the `checkpoints/` directory.

### 2. Running Benchmarks

Evaluate the trained model on a suite of reasoning benchmarks using `benchmark_suite.py`. This script includes:

- Six-digit addition
- GSM8K math word problems
- Multi-hop logic chains
- Parentheses matching
- Symbol rewriting
- Simple code generation

To run the benchmarks on a trained checkpoint:
```bash
python benchmark_suite.py --checkpoint_path checkpoints/phase_4_epoch_0.pt
```

### 3. Chat with the Model

Interact with your trained IDIR model using the `chat.py` script.

```bash
python chat.py --checkpoint_path checkpoints/phase_4_epoch_0.pt
```

You can customize generation parameters like temperature, top-k, and number of beams.

## Performance

The `flop_analysis.py` script provides a detailed breakdown of the model's performance characteristics.

- **Parameter Count**: ~70M parameters, closely matching the paper's specification.
- **FLOPs**: The script calculates the theoretical FLOPs for the IDIR model and compares it to dense transformer baselines.
- **Wall-Clock Time**: The script provides an estimated wall-clock time comparison, showing IDIR's potential for efficient inference.

To generate a full performance report:
```bash
python flop_analysis.py
```

## Key Files

- `idir_model.py`: The core `IDIRModel` implementation.
- `training_curriculum.py`: The 4-phase training script.
- `benchmark_suite.py`: The reasoning benchmark suite.
- `flop_analysis.py`: The performance analysis script.
- `chat.py`: The interactive chat script.
- `teacher_model.py`: A placeholder for the teacher model, not used in the final implementation.
- `test_model.py`: A simple script to test the model's forward pass.