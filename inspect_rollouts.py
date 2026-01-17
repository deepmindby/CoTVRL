#!/usr/bin/env python3
"""
Inspect Rollouts - Utility script to debug RL performance.

This script loads a trained vector and runs inference on random samples
to visually verify if the vector is inducing Chain-of-Thought reasoning.

Usage:
    python inspect_rollouts.py --vector_path outputs/self_evolved_gsm8k_L0.pt \
                               --model_path /path/to/model \
                               --data_path /path/to/data \
                               --num_samples 5
"""

import os
import sys
import argparse
import random
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset, PROMPT_TEMPLATES
from src.utils import extract_answer_from_text, compare_answers, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect rollouts from a trained CoT vector"
    )
    
    parser.add_argument(
        "--vector_path",
        type=str,
        required=True,
        help="Path to the trained vector file (.pt)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen",
        choices=["qwen", "llama"],
        help="Model architecture name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Layer index for vector injection"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to inspect"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        default=False,
        help="Also show baseline (without vector) for comparison"
    )
    
    return parser.parse_args()


def load_vector(path: str, device: torch.device) -> torch.Tensor:
    """Load a saved vector from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found: {path}")
    
    loaded = torch.load(path, map_location=device)
    
    # Handle different save formats
    if isinstance(loaded, dict) and "vector" in loaded:
        vector = loaded["vector"]
    else:
        vector = loaded
    
    return vector.to(device)


def generate_with_vector(
    model_wrapper: CoTModelWrapper,
    tokenizer,
    prompt: str,
    vector: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text with vector injection."""
    device = model_wrapper.device
    
    # Tokenize
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    input_len = input_ids.shape[1]
    
    # Clear and register injection hook
    model_wrapper.clear_hooks()
    model_wrapper.register_injection_hook(
        layer_idx,
        vector,
        scaling_factor=1.0,
        requires_grad=False
    )
    
    # Generate
    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode generated part only
    generated_ids = outputs[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    model_wrapper.clear_hooks()
    
    return generated_text


def generate_baseline(
    model_wrapper: CoTModelWrapper,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text without vector injection (baseline)."""
    device = model_wrapper.device
    
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    input_len = input_ids.shape[1]
    
    model_wrapper.clear_hooks()
    
    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def main():
    """Main inspection routine."""
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 70)
    print("CoT Vector Rollout Inspector")
    print("=" * 70)
    print(f"Vector: {args.vector_path}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Layer: {args.layer_idx}")
    print(f"Samples: {args.num_samples}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    device = model_wrapper.device
    print(f"Model loaded, device: {device}")
    
    # Load vector
    print(f"\nLoading vector from {args.vector_path}...")
    vector = load_vector(args.vector_path, device)
    print(f"Vector shape: {vector.shape}, norm: {vector.norm().item():.4f}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} test set...")
    samples = load_dataset(args.data_path, args.dataset, "test", num_samples=100)
    print(f"Loaded {len(samples)} samples")
    
    # Get prompt template
    prompt_template = PROMPT_TEMPLATES.get(args.dataset, PROMPT_TEMPLATES["gsm8k"])
    
    # Randomly select samples
    selected = random.sample(samples, min(args.num_samples, len(samples)))
    
    print("\n" + "=" * 70)
    print("INSPECTION RESULTS")
    print("=" * 70)
    
    correct_with_vector = 0
    correct_baseline = 0
    
    for i, sample in enumerate(selected, 1):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i}/{args.num_samples}")
        print("=" * 70)
        
        # Build prompt
        if args.dataset == "mmlu_pro":
            prompt = prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = prompt_template["cot"].format(question=sample.question)
        
        # Display question
        print(f"\n📝 QUESTION:")
        print("-" * 50)
        print(sample.question)
        
        # Display ground truth
        print(f"\n✅ GROUND TRUTH: {sample.answer}")
        
        # Generate with vector
        print(f"\n🤖 GENERATED (with vector @ layer {args.layer_idx}):")
        print("-" * 50)
        generated = generate_with_vector(
            model_wrapper, tokenizer, prompt,
            vector, args.layer_idx,
            args.max_new_tokens, args.temperature
        )
        print(generated)
        
        # Extract and compare answer
        predicted = extract_answer_from_text(generated, args.dataset)
        is_correct = compare_answers(predicted, sample.answer, args.dataset)
        
        print("-" * 50)
        print(f"Extracted answer: {predicted}")
        print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")
        
        if is_correct:
            correct_with_vector += 1
        
        # Optionally show baseline
        if args.compare_baseline:
            print(f"\n🔄 BASELINE (without vector):")
            print("-" * 50)
            baseline_gen = generate_baseline(
                model_wrapper, tokenizer, prompt,
                args.max_new_tokens, args.temperature
            )
            print(baseline_gen)
            
            baseline_pred = extract_answer_from_text(baseline_gen, args.dataset)
            baseline_correct = compare_answers(baseline_pred, sample.answer, args.dataset)
            
            print("-" * 50)
            print(f"Extracted answer: {baseline_pred}")
            print(f"Correct: {'✓ YES' if baseline_correct else '✗ NO'}")
            
            if baseline_correct:
                correct_baseline += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"With vector:     {correct_with_vector}/{args.num_samples} correct "
          f"({correct_with_vector/args.num_samples*100:.1f}%)")
    
    if args.compare_baseline:
        print(f"Baseline:        {correct_baseline}/{args.num_samples} correct "
              f"({correct_baseline/args.num_samples*100:.1f}%)")
        diff = correct_with_vector - correct_baseline
        print(f"Improvement:     {'+' if diff >= 0 else ''}{diff} samples")
    
    print("=" * 70)
    print("Inspection complete!")


if __name__ == "__main__":
    main()