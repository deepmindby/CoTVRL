#!/usr/bin/env python3
"""
Multi-Layer UA-Vector: Extract and inject vectors at multiple (or all) layers.

This script extracts UA-Vectors at multiple layers simultaneously and
evaluates their combined effect when injected together.

Usage:
    # All layers
    python run_multi_layer.py \
        --model_path /path/to/model \
        --data_path /path/to/data \
        --dataset gsm8k \
        --ua_gamma 1.0

    # Specific layers
    python run_multi_layer.py \
        --model_path /path/to/model \
        --layers 0,5,10,15,20 \
        --ua_gamma 1.0

    # Layer range
    python run_multi_layer.py \
        --model_path /path/to/model \
        --layer_start 10 \
        --layer_end 20 \
        --ua_gamma 1.0
"""

import os
import argparse
import torch
from datetime import datetime

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.multi_layer_ua import MultiLayerUAVector, MultiLayerEvaluator
from src.eval import run_baseline_evaluation
from src.utils import set_seed, setup_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Layer UA-Vector Extraction and Evaluation")
    
    # Model & Data
    parser.add_argument("--model_path", type=str, default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--data_path", type=str, default="/home/haichao/TA/CoTVRL/data")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # Layer selection (mutually exclusive options)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (e.g., '0,5,10,15')")
    parser.add_argument("--layer_start", type=int, default=None,
                        help="Start layer index (inclusive)")
    parser.add_argument("--layer_end", type=int, default=None,
                        help="End layer index (exclusive)")
    parser.add_argument("--layer_step", type=int, default=1,
                        help="Step size for layer range")
    parser.add_argument("--all_layers", action="store_true", default=False,
                        help="Use all layers (default if no layer args specified)")
    
    # UA-Vector parameters
    parser.add_argument("--ua_gamma", type=float, default=1.0,
                        help="Noise penalty factor γ")
    parser.add_argument("--ua_normalize_variance", action="store_true", default=True)
    parser.add_argument("--no_ua_normalize_variance", action="store_true", default=False)
    parser.add_argument("--scaling_factor", type=float, default=1.0,
                        help="Global scaling factor for injection")
    
    # Data configuration
    parser.add_argument("--num_support_samples", type=int, default=100)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=3)
    
    # Mode
    parser.add_argument("--skip_baseline", action="store_true", default=False)
    parser.add_argument("--save_vectors", action="store_true", default=True)
    parser.add_argument("--load_vectors", type=str, default=None,
                        help="Load pre-extracted vectors from file")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="cot-vectors-multilayer")
    
    return parser.parse_args()


def determine_layers(args, num_layers: int):
    """Determine which layers to use based on arguments."""
    if args.layers:
        # Explicit layer list
        return [int(l.strip()) for l in args.layers.split(",")]
    elif args.layer_start is not None and args.layer_end is not None:
        # Layer range
        return list(range(args.layer_start, args.layer_end, args.layer_step))
    elif args.layer_start is not None:
        # From start to end
        return list(range(args.layer_start, num_layers, args.layer_step))
    elif args.layer_end is not None:
        # From 0 to end
        return list(range(0, args.layer_end, args.layer_step))
    else:
        # All layers (default)
        return list(range(0, num_layers, args.layer_step))


def main():
    args = parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("Multi-Layer UA-Vector Extraction and Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path.split('/')[-1]}")
    print(f"Dataset: {args.dataset}")
    print(f"Gamma (γ): {args.ua_gamma}")
    normalize_var = args.ua_normalize_variance and not args.no_ua_normalize_variance
    print(f"Normalize variance: {normalize_var}")
    print(f"Scaling factor: {args.scaling_factor}")
    print("=" * 70)
    
    # Setup WandB
    wandb_run = None
    if args.use_wandb:
        wandb_run = setup_wandb(args)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"Model loaded: {model_wrapper.num_layers} layers, hidden_size={model_wrapper.hidden_size}")
    
    # Determine layers
    layer_indices = determine_layers(args, model_wrapper.num_layers)
    print(f"Target layers: {layer_indices} ({len(layer_indices)} layers)")
    
    # Load data
    print("\nLoading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support: {len(support_samples)}, Test: {len(test_samples)}")
    
    # Extract or load vectors
    if args.load_vectors:
        print(f"\nLoading vectors from {args.load_vectors}...")
        multi_layer = MultiLayerUAVector.load(args.load_vectors, model_wrapper, tokenizer)
    else:
        print("\n" + "=" * 70)
        print("Extracting Multi-Layer UA-Vectors...")
        print("=" * 70)
        
        multi_layer = MultiLayerUAVector(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            dataset_type=args.dataset,
            gamma=args.ua_gamma,
            normalize_variance=normalize_var,
        )
        
        multi_layer.extract(support_samples, wandb_run)
        
        # Save vectors
        if args.save_vectors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                args.output_dir,
                f"multi_layer_ua_{args.dataset}_g{args.ua_gamma}_{timestamp}.pt"
            )
            multi_layer.save(save_path)
    
    # Baseline evaluation
    baseline_accuracy = None
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("Baseline Evaluation (no injection)")
        print("=" * 70)
        
        baseline_results = run_baseline_evaluation(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=test_samples,
            dataset_type=args.dataset,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        baseline_accuracy = baseline_results['accuracy']
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Multi-layer injection evaluation
    print("\n" + "=" * 70)
    print(f"Multi-Layer Injection Evaluation ({len(layer_indices)} layers)")
    print("=" * 70)
    
    evaluator = MultiLayerEvaluator(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        multi_layer_vector=multi_layer,
        dataset_type=args.dataset,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    
    injection_results = evaluator.evaluate_dataset(
        test_samples,
        scaling_factor=args.scaling_factor,
        desc="Multi-Layer Eval"
    )
    
    # Print final results
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    print(f"Dataset:         {args.dataset}")
    print(f"Layers used:     {len(layer_indices)} layers")
    print(f"Gamma (γ):       {args.ua_gamma}")
    print(f"Scaling factor:  {args.scaling_factor}")
    print("-" * 70)
    
    if baseline_accuracy is not None:
        print(f"Baseline:        {baseline_accuracy:.2f}%")
        diff = injection_results['accuracy'] - baseline_accuracy
        sign = "+" if diff >= 0 else ""
        print(f"Multi-Layer:     {injection_results['accuracy']:.2f}% [{sign}{diff:.2f}%]")
    else:
        print(f"Multi-Layer:     {injection_results['accuracy']:.2f}%")
    
    # Per-layer statistics summary
    stats = multi_layer.get_statistics()
    total_norm = sum(s['final_vector_norm'] for s in stats.values())
    avg_lambda = sum(s['lambda_mean'] for s in stats.values()) / len(stats)
    
    print("-" * 70)
    print(f"Total vector norm: {total_norm:.4f}")
    print(f"Average λ:         {avg_lambda:.4f}")
    print("=" * 70)
    
    # Log to WandB
    if wandb_run:
        wandb_run.log({
            "eval/baseline_accuracy": baseline_accuracy,
            "eval/multilayer_accuracy": injection_results['accuracy'],
            "eval/improvement": injection_results['accuracy'] - (baseline_accuracy or 0),
            "eval/num_layers": len(layer_indices),
            "eval/total_vector_norm": total_norm,
            "eval/avg_lambda": avg_lambda,
        })
        wandb_run.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()