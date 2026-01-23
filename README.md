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

### Single-Layer Extraction and Evaluation
```bash
# Extracted CoT Vector (Baseline)
python main.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method extracted \
    --layer_idx 15 \
    --num_support_samples 100 \
    --num_test_samples 50 \
    --mode both

# UA-Vector (Recommended)
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

### Layer Sweep (Find Best Layer)
```bash
# Sweep all layers with extracted method
python run_layer_sweep.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method extracted \
    --layer_step 2 \
    --num_support_samples 100 \
    --num_test_samples 50

# Sweep all layers with UA-Vector
python run_layer_sweep.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method ua_vector \
    --ua_gamma 1.0 \
    --layer_step 2 \
    --num_support_samples 100 \
    --num_test_samples 50

# Sweep specific layers
python run_layer_sweep.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --method ua_vector \
    --layers 0,5,10,15,20,25 \
    --save_vectors
```

### Multi-Layer Injection (All Layers Simultaneously)
```bash
# Inject UA-Vectors at ALL layers
python run_multi_layer.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --ua_gamma 1.0 \
    --num_support_samples 100 \
    --num_test_samples 50

# Inject at specific layers
python run_multi_layer.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --layers 10,15,20,25 \
    --ua_gamma 1.0

# Inject at layer range (layers 10-25)
python run_multi_layer.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --layer_start 10 \
    --layer_end 26 \
    --ua_gamma 1.0
```

## Project Structure

```
cot_vectors/
├── main.py                  # Single-layer entry point
├── run_layer_sweep.py       # Layer sweep script
├── run_multi_layer.py       # Multi-layer injection script
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
│       ├── ua_vector.py     # UA-Vector (single layer)
│       └── multi_layer_ua.py # Multi-layer UA-Vector
└── outputs/                 # Saved vectors
```

## Key Arguments

### UA-Vector Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ua_gamma` | 1.0 | Noise penalty factor γ. When γ=0, degrades to standard Extracted Vector |
| `--ua_normalize_variance` | True | Normalize σ² by layer mean for cross-layer robustness |
| `--no_ua_normalize_variance` | - | Disable variance normalization |

### Layer Sweep Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layers` | None | Comma-separated layers (e.g., "0,5,10,15") |
| `--layer_step` | 2 | Step size when testing all layers |
| `--save_vectors` | False | Save vectors for each layer |
| `--skip_baseline` | False | Skip baseline evaluation |

### Multi-Layer Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layers` | None | Comma-separated layers (None = all) |
| `--layer_start` | None | Start layer index |
| `--layer_end` | None | End layer index |
| `--scaling_factor` | 1.0 | Global scaling for all vectors |

### Choosing γ (Gamma)

- **γ = 0**: No shrinkage, equivalent to Extracted Vector
- **γ = 0.1-1.0**: Mild shrinkage, preserves most dimensions
- **γ = 1.0-10.0**: Moderate shrinkage, suppresses noisy dimensions
- **γ > 10**: Aggressive shrinkage, only preserves very consistent dimensions

## Output Statistics

When using UA-Vector methods, the following statistics are reported:

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