# IDIR-KS: Knowledge-Dense Implicit Reasoning with Inference-Time Scaling

A novel language modeling architecture that combines implicit fixed-point reasoning, knowledge-dense parameterization, internal memory retrieval, sparse mixture-of-experts (MoE), and inference-time compute scaling.

## Architecture Overview

```
Intelligence ≈ Structure × Compute × Data Quality
```

### Key Components

1. **Implicit Fixed-Point Reasoning**
   - Iterative solver: `h* = F_theta(h*, x, M)`
   - Replaces depth with iterative computation
   - Adaptive compute: `steps = base_steps + sigmoid(W h) × extra_steps`

2. **Knowledge-Dense Factorization**
   - Factorized transformations: `W ≈ A × B`
   - Reduces redundancy while maintaining expressiveness

3. **Internal Memory Retrieval**
   - Differentiable memory access: `Memory(h) = softmax(h M^T) M`
   - Multi-head attention over memory matrix

4. **Mixture-of-Experts**
   - Sparse MoE with top-k routing
   - Domain specialization: code, math, logic, language

5. **Inference-Time Scaling**
   - Multi-trajectory reasoning with self-consistency
   - `y = argmax_y Σ_i P(y | h_i*)`

6. **Hybrid Optimization**
   - AdamW for core layers: `lr=2e-4, betas=(0.9, 0.95)`
   - RMSProp for memory/routing: `lr=1e-4, alpha=0.99`

## Installation

```bash
git clone <repository>
cd idir_ks
pip install torch pyyaml  # Add other dependencies
```

## Quick Start

### Test the Implementation

```bash
python -m idir_ks.main test
```

### Train a Model

```bash
# Base model
python -m idir_ks.main train --size base --device cuda

# Small model for testing
python -m idir_ks.main train --size small --device cpu

# Large model
python -m idir_ks.main train --size large --device cuda
```

### Run Ablation Study

```bash
# Test all ablation variants
python -m idir_ks.main ablation

# Test specific variant
python -m idir_ks.main ablation --variant A
```

### Evaluate Model

```bash
python -m idir_ks.main evaluate --checkpoint checkpoints/final_model.pt
```

## Architecture Details

### Model Configuration

```python
from idir_ks.utils import get_base_config

config = get_base_config()
config.model.dim = 768
config.model.num_layers = 6
config.model.num_experts = 8
config.model.use_implicit_solver = True
config.model.use_memory = True
config.model.use_moe = True
config.model.use_factorization = True
```

### Creating the Model

```python
from idir_ks.model import IDIRKSModel

model = IDIRKSModel(
    vocab_size=50000,
    dim=768,
    num_layers=6,
    num_heads=12,
    num_experts=8,
    # ... other parameters
)
```

### Training

```python
from idir_ks.training import IDIRKSTrainer, create_composite_dataset

# Create dataset
dataset = create_composite_dataset(
    code_path='data/code.jsonl',
    math_path='data/math.jsonl',
    logic_path='data/logic.jsonl',
    language_path='data/language.jsonl',
    weights={'code': 0.40, 'math': 0.25, 'logic': 0.20, 'language': 0.15},
)

# Create trainer
trainer = IDIRKSTrainer(
    model=model,
    train_dataloader=dataloader,
    device='cuda',
)

# Train
trainer.train()
```

## Ablation Study

The paper defines 9 ablation variants:

| Variant | Description | Expected Impact |
|---------|-------------|-----------------|
| A | Full Model (baseline) | - |
| B | No Implicit Solver | Significant drop in reasoning depth |
| C | No Memory Module | Weaker generalization |
| D | No MoE | Reduced specialization |
| E | No Factorization | Increased redundancy |
| F | No Multi-Trajectory | Lower robustness |
| G | No Adaptive Compute | Inefficient inference |
| H | Adam Only | Unstable routing and memory |
| I | RMSProp Only | Slower convergence |

## Dataset Composition

From the paper:

| Domain | Weight | Purpose |
|--------|--------|---------|
| Code | 40% | Structured reasoning, syntax |
| Math | 25% | Mathematical reasoning |
| Logic | 20% | Logical inference |
| Language | 15% | General language understanding |

## Training Phases

### Phase 1: Stability Warm-up (5,000 steps)
- Focus: Language + syntax
- Optimizer: Adam only
- Learning rate: 1e-4

### Phase 2: Full System (45,000 steps)
- Focus: Reasoning + structured tasks
- Optimizer: Adam + RMSProp
- Learning rate: 2e-4 (Adam), 1e-4 (RMSProp)

### Phase 3: Convergence (5,000 steps)
- Focus: Multi-trajectory consistency
- Reduced learning rates

## Loss Function

```
L = CE + λ1 * consistency_loss + λ2 * entropy_regularization
```

- `CE`: Standard cross-entropy
- `consistency_loss`: Encourage trajectory agreement
- `entropy_regularization`: Maintain prediction diversity

## Evaluation Metrics

1. **Perplexity** - Language modeling quality
2. **GSM8K Accuracy** - Math reasoning
3. **MBPP Pass@k** - Code generation
4. **Logical Reasoning Accuracy** - Logical inference
5. **Training Stability** - Loss variance
6. **Throughput** - Tokens/sec

## Project Structure

```
idir_ks/
├── model/
│   ├── __init__.py
│   ├── idir_core.py          # Fixed-point solver
│   ├── memory_module.py        # Memory retrieval
│   ├── moe_layer.py           # Mixture-of-experts
│   ├── factorized_linear.py  # Knowledge-dense layers
│   └── idir_ks_model.py      # Complete model
├── training/
│   ├── __init__.py
│   ├── hybrid_optimizer.py  # AdamW + RMSProp
│   ├── trainer.py            # Training loop
│   └── data.py              # Data loading
├── evaluation/
│   ├── __init__.py
│   ├── ablations.py         # Ablation variants
│   └── metrics.py           # Evaluation metrics
├── utils/
│   ├── __init__.py
│   └── config.py            # Configuration management
└── main.py                  # Entry point
```

## Citation

```bibtex
@article{idir_ks_2024,
  title={IDIR-KS: Knowledge-Dense Implicit Reasoning with Inference-Time Scaling and Hybrid Optimization for Efficient Language Modeling},
  year={2024}
}
```

## License

MIT License

## Key Equations

### Fixed-Point Objective
```
h* = F_theta(h*, x, M)
```

### Iterative Solver
```
h_{k+1} = F_theta(h_k, x, M)
```

### Operator Definition
```
F_theta(h, x, M) = Norm(
    h
    + G1 ⊙ Phi(h, x)
    + G2 ⊙ Memory(h, M)
    + G3 ⊙ Experts(h)
)
```

### Memory Retrieval
```
Memory(h) = softmax(h M^T) M
```

### Mixture-of-Experts
```
Experts(h) = Σ_i g_i E_i(h), top-k routing
```

### Adaptive Compute
```
steps = base_steps + sigmoid(W h) × extra_steps
```

## Performance Characteristics

- **Parameter Efficiency**: Factorization reduces parameters while maintaining capacity
- **Compute Scaling**: Iterative reasoning increases capability without parameter growth
- **Training Stability**: Hybrid optimization stabilizes heterogeneous components
- **Inference**: Adaptive compute balances quality vs. cost
