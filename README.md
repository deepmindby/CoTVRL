# CoT Vectors Reproduction

Reproduction of "CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs" with extension for **Uncertainty-Aware CoT Vector (UA-Vector)**.

## Methods

### 1. Extracted CoT Vector (Baseline)
Standard extraction method that computes the mean of activation differences:
```
v = (1/N) Σ v^(i)
```

### 2. UA-Vector (Uncertainty-Aware) ⭐ New
Bayesian shrinkage-based method that uses variance to adaptively gate the mean vector:
```
μ = (1/N) Σ v^(i)           # Mean (first moment)
σ² = (1/(N-1)) Σ (v^(i)-μ)²  # Variance (second moment, unbiased)
λ = 1 / (1 + γ·σ²)           # Gating coefficient
v* = λ ⊙ μ                   # Final vector (Hadamard product)
```

**Key Insight:**
- Low variance dimensions → High confidence → λ ≈ 1 → Trust the mean
- High variance dimensions → Low confidence → λ ≈ 0 → Suppress to zero

## Quick Start

### Extracted CoT Vector (Baseline)
```bash
python main.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method extracted \
    --layer_idx 15 \
    --num_support_samples 100 \
    --num_test_samples 50 \
    --mode both
```

### UA-Vector (Recommended)
```bash
python main.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method ua_vector \
    --layer_idx 15 \
    --ua_gamma 1.0 \
    --ua_normalize_variance \
    --num_support_samples 100 \
    --num_test_samples 50 \
    --mode both
```

### Extract Only (No Evaluation)
```bash
python main.py \
    --method ua_vector \
    --ua_gamma 1.0 \
    --mode extract \
    --save_vector
```

### Evaluate Pre-extracted Vector
```bash
python main.py \
    --vector_path outputs/ua_vector_gsm8k_L15_xxx.pt \
    --layer_idx 15 \
    --mode eval
```

## UA-Vector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ua_gamma` | 1.0 | Noise penalty factor γ. When γ=0, degrades to standard Extracted Vector |
| `--ua_normalize_variance` | True | Normalize σ² by layer mean for cross-layer robustness |
| `--no_ua_normalize_variance` | - | Disable variance normalization |

### Choosing γ (Gamma)

- **γ = 0**: No shrinkage, equivalent to Extracted Vector
- **γ = 0.1-1.0**: Mild shrinkage, preserves most dimensions
- **γ = 1.0-10.0**: Moderate shrinkage, suppresses noisy dimensions
- **γ > 10**: Aggressive shrinkage, only preserves very consistent dimensions

## Project Structure

```
cot_vectors/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── config/
│   └── secrets.yaml         # WandB credentials
├── src/
│   ├── __init__.py
│   ├── args.py              # Argument parser
│   ├── data_utils.py        # Data loading & prompts
│   ├── eval.py              # Evaluation logic
│   ├── models.py            # Model wrapper with hooks
│   ├── utils.py             # Utilities
│   └── methods/
│       ├── __init__.py      # Method registry
│       ├── base.py          # Base class
│       ├── extracted.py     # Extracted CoT Vector
│       └── ua_vector.py     # UA-Vector (main contribution)
└── outputs/                 # Saved vectors
```

## Key Arguments

### General
- `--model_path`: Path to pretrained model
- `--model_name`: Model type (`qwen` or `llama`)
- `--data_path`: Path to data directory
- `--dataset`: Dataset (`gsm8k`, `math_easy`, `math_hard`, `mmlu_pro`)
- `--layer_idx`: Layer index for vector injection/extraction
- `--seed`: Random seed (default: 42)

### Data
- `--num_support_samples`: Support set size for extraction
- `--num_test_samples`: Test set size for evaluation

### Generation
- `--max_new_tokens`: Max tokens to generate (default: 512)
- `--num_beams`: Beam search width (default: 3)
- `--use_early_stopping`: Stop when answer detected

### Logging
- `--use_wandb`: Enable WandB logging
- `--wandb_project`: WandB project name

## Output Statistics (UA-Vector)

When using `--method ua_vector`, the following statistics are reported:

| Statistic | Description |
|-----------|-------------|
| `Original norm` | Norm of mean vector μ (standard Extracted Vector) |
| `Final norm` | Norm of UA-Vector v* = λ⊙μ |
| `Norm reduction` | Percentage reduction from shrinkage |
| `Lambda mean` | Average gating coefficient |
| `Suppressed dims` | % of dimensions with λ < 0.5 |
| `Preserved dims` | % of dimensions with λ > 0.9 |

## Data Format

Expected data structure:
```
data/
├── gsm8k/
│   ├── train.jsonl
│   └── test.jsonl
├── math/
│   ├── train.jsonl
│   └── test.jsonl
└── mmlu_pro/
    ├── validation.jsonl
    └── test.jsonl
```

Each JSONL file should have lines with:
```json
{"question": "...", "answer": "...", "cot": "..."}
```

## Citation

If you use UA-Vector in your research, please cite:

```bibtex
@article{cot_vectors,
  title={CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs},
  author={...},
  year={2024}
}
```