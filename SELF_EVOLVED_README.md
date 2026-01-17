# Self-Evolved CoT Vector - Refactored RL Implementation

This document describes the refactored implementation of Self-Evolved CoT Vector using GRPO and DAPO algorithms.

## Overview

The implementation follows a clean, modular design:

```
src/
├── args.py                 # All hyperparameters
├── rl_solvers/
│   ├── __init__.py
│   ├── grpo.py            # GRPO algorithm implementation
│   └── dapo.py            # DAPO algorithm implementation
└── methods/
    └── self_evolved.py    # Main training class
```

## Quick Start

### Using GRPO (Default)

```bash
python main.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method self_evolved \
    --rl_method grpo \
    --layer_idx 0 \
    --num_rollouts 8 \
    --num_iterations 100 \
    --learning_rate_vector 1e-2 \
    --mode both
```

### Using DAPO

```bash
python main.py \
    --model_path /path/to/Qwen2.5-Math-7B \
    --data_path /path/to/data \
    --dataset gsm8k \
    --method self_evolved \
    --rl_method dapo \
    --layer_idx 0 \
    --num_rollouts 8 \
    --num_iterations 100 \
    --beta 0.1 \
    --mode both
```

### Initialize from Extracted Vector

```bash
python main.py \
    --method self_evolved \
    --rl_method grpo \
    --init_from_extracted \
    --extracted_vector_path outputs/extracted_gsm8k_L0.pt \
    ...
```

### Using Soft Rewards

```bash
python main.py \
    --method self_evolved \
    --soft_reward \
    ...
```

## Inspecting Rollouts

After training, use the inspection script to verify the vector's effect:

```bash
python inspect_rollouts.py \
    --vector_path outputs/self_evolved_gsm8k_L0_xxx.pt \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --num_samples 5 \
    --compare_baseline
```

## Key Arguments

### RL Method Selection
- `--rl_method`: Choose `grpo` or `dapo`

### Reward Configuration
- `--soft_reward`: Enable partial rewards for format/length (default: binary 0/1)

### Vector Initialization
- `--init_from_extracted`: Initialize from pre-extracted vector
- `--extracted_vector_path`: Path to extracted vector file
- `--init_std`: Std for random initialization (0.0 = zero init)

### GRPO/DAPO Parameters
- `--num_rollouts`: Group size G (default: 8)
- `--beta`: KL penalty (GRPO) or temperature (DAPO) (default: 0.0)
- `--learning_rate_vector`: Learning rate for vector (default: 1e-2)
- `--max_grad_norm`: Gradient clipping norm (default: 1.0)

### Training Configuration
- `--num_iterations`: Number of training iterations (default: 100)
- `--questions_per_iter`: Questions sampled per iteration (default: 4)

### Generation Parameters
- `--rl_max_new_tokens`: Max tokens for RL generation (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Nucleus sampling parameter (default: 0.9)

## Algorithm Details

### GRPO (Group Relative Policy Optimization)

1. **Sampling Phase** (No Gradient):
   - Generate G completions per question with vector injection
   
2. **Reward Computation**:
   - Binary: 1.0 if correct, 0.0 otherwise
   - Soft: 1.0 (correct) + 0.1 (valid format) + 0.05 (reasonable length)
   
3. **Advantage Normalization**:
   - Normalize rewards within group to mean=0, std=1
   
4. **Training Phase** (With Gradient):
   - Re-run forward pass on generated sequences
   - Compute GRPO loss: `-1/G * Σ(A_i * Σlog π(a_i|x))`
   
5. **Optimization**:
   - Update vector with gradient clipping

### DAPO (Direct Alignment Policy Optimization)

1. **Sampling Phase**:
   - Generate G completions per question
   
2. **Pair Construction**:
   - Identify Winner (correct) and Loser (incorrect)
   - Skip if no valid pair exists
   
3. **Training Phase**:
   - Compute log probabilities for winner and loser
   - DPO loss: `-log σ(β * (log π(y_w|x) - log π(y_l|x)))`

## Memory Optimization

The implementation uses several strategies to manage memory:

1. **Detached Generation**: Rollout generation uses detached vectors
2. **Per-Sample Gradients**: Processes one sample at a time
3. **Periodic Cleanup**: Regular `torch.cuda.empty_cache()` calls
4. **Gradient Accumulation**: Optional accumulation for larger effective batches

## Compatibility

- The `extracted` and `learnable` methods remain unchanged
- All existing evaluation code works with vectors from all methods
- Vector save format is consistent across methods