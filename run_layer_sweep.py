#!/usr/bin/env python3
"""
Layer sweep script for CoT Vectors.
Evaluates injection at different layers to find optimal performance.

Supports all methods: extracted, ua_vector, learnable, self_evolved

Usage:
    # Extracted method
    python run_layer_sweep.py \
        --model_path /path/to/model \
        --data_path /path/to/data \
        --dataset gsm8k \
        --method extracted

    # UA-Vector method
    python run_layer_sweep.py \
        --model_path /path/to/model \
        --data_path /path/to/data \
        --dataset gsm8k \
        --method ua_vector \
        --ua_gamma 1.0

    # Specific layers
    python run_layer_sweep.py \
        --layers 0,5,10,15,20,25 \
        --method ua_vector
"""

import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.ua_vector import UncertaintyAwareCoTVector
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed, save_vector


def parse_args():
    parser = argparse.ArgumentParser(description="Layer sweep for CoT vectors")
    
    # ==================== Model & Data ====================
    parser.add_argument("--model_path", type=str, default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--data_path", type=str, default="/home/haichao/TA/CoTVRL/data")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_support_samples", type=int, default=100)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # ==================== Method Selection ====================
    parser.add_argument("--method", type=str, default="extracted",
                        choices=["extracted", "ua_vector", "learnable", "self_evolved"],
                        help="CoT Vector acquisition method")
    
    # ==================== Layer Selection ====================
    parser.add_argument("--layers", type=str, default=None, 
                        help="Comma-separated layers to test (e.g., '0,5,10'). Default: all layers")
    parser.add_argument("--layer_step", type=int, default=2,
                        help="Step size when testing all layers (e.g., 2 = test every 2nd layer)")
    
    # ==================== Generation Config ====================
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--use_early_stopping", action="store_true", default=False)
    parser.add_argument("--scaling_factor", type=float, default=1.0)
    
    # ==================== Evaluation Options ====================
    parser.add_argument("--skip_baseline", action="store_true", default=False,
                        help="Skip baseline evaluation")
    parser.add_argument("--baseline_accuracy", type=float, default=None,
                        help="Pre-computed baseline accuracy (use with --skip_baseline)")
    
    # ==================== UA-Vector Config ====================
    parser.add_argument("--ua_gamma", type=float, default=1.0,
                        help="Noise penalty factor γ for UA-Vector")
    parser.add_argument("--ua_normalize_variance", action="store_true", default=True,
                        help="Normalize variance by layer mean")
    parser.add_argument("--no_ua_normalize_variance", action="store_true", default=False,
                        help="Disable variance normalization")
    
    # ==================== Learnable Vector Config ====================
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=1024)
    
    # ==================== Self-Evolved (GRPO) Config ====================
    parser.add_argument("--rl_method", type=str, default="grpo", choices=["grpo", "dapo"])
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--grpo_lr", type=float, default=5e-3)
    parser.add_argument("--questions_per_iter", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--init_std", type=float, default=0.35)
    parser.add_argument("--soft_reward", action="store_true", default=False)
    
    # ==================== Saving Options ====================
    parser.add_argument("--save_vectors", action="store_true", default=False,
                        help="Save vectors for each layer")
    parser.add_argument("--load_vectors_dir", type=str, default=None,
                        help="Load pre-trained vectors from directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("CoT Vector Layer Sweep")
    print("=" * 70)
    print(f"Model:    {args.model_path.split('/')[-1]}")
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    print(f"Support:  {args.num_support_samples}, Test: {args.num_test_samples}")
    
    if args.method == "ua_vector":
        normalize_var = args.ua_normalize_variance and not args.no_ua_normalize_variance
        print(f"UA-Vector Config: γ={args.ua_gamma}, normalize_var={normalize_var}")
    elif args.method == "learnable":
        print(f"Learnable Config: epochs={args.num_epochs}, batch={args.batch_size}, "
              f"lr={args.learning_rate}, λ={args.lambda_val}")
    elif args.method == "self_evolved":
        print(f"GRPO Config: G={args.group_size}, iters={args.num_iterations}, "
              f"Q/iter={args.questions_per_iter}, lr={args.grpo_lr}")
    
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    num_layers = model_wrapper.num_layers
    print(f"Model has {num_layers} layers")
    
    # Load data
    print("\nLoading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support: {len(support_samples)}, Test: {len(test_samples)}")
    
    # Determine layers to test
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = list(range(0, num_layers, args.layer_step))
    
    print(f"\nLayers to test: {layers}")
    print(f"Total: {len(layers)} layers")
    
    # Baseline evaluation
    baseline_accuracy = args.baseline_accuracy
    
    if not args.skip_baseline:
        print("\n" + "-" * 70)
        print("Running baseline evaluation...")
        baseline = run_baseline_evaluation(
            model_wrapper, tokenizer, test_samples, args.dataset,
            args.max_new_tokens, args.num_beams, args.use_early_stopping
        )
        baseline_accuracy = baseline['accuracy']
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    elif baseline_accuracy is not None:
        print(f"\nUsing provided baseline accuracy: {baseline_accuracy:.2f}%")
    else:
        print("\nSkipping baseline (no accuracy provided)")
        baseline_accuracy = 0.0
    
    # Layer sweep
    print("\n" + "-" * 70)
    print(f"Testing {len(layers)} layers with method: {args.method}")
    print("-" * 70)
    
    results = []
    vectors_dict = {}
    
    for layer_idx in tqdm(layers, desc="Layer sweep", ncols=100):
        print(f"\n>>> Layer {layer_idx}")
        
        vector = None
        method_stats = None
        
        # Check if we should load a pre-trained vector
        if args.load_vectors_dir:
            vector_path = os.path.join(
                args.load_vectors_dir,
                f"{args.method}_{args.dataset}_L{layer_idx}.pt"
            )
            if os.path.exists(vector_path):
                print(f"  Loading vector from {vector_path}")
                loaded = torch.load(vector_path, map_location="cpu")
                if isinstance(loaded, dict) and "vector" in loaded:
                    vector = loaded["vector"]
                else:
                    vector = loaded
        
        # Extract/Train vector if not loaded
        if vector is None:
            try:
                if args.method == "extracted":
                    method = ExtractedCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                    )
                    vector = method.extract(support_samples)
                    
                elif args.method == "ua_vector":
                    normalize_var = args.ua_normalize_variance and not args.no_ua_normalize_variance
                    method = UncertaintyAwareCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        gamma=args.ua_gamma,
                        normalize_variance=normalize_var,
                    )
                    vector = method.extract(support_samples)
                    method_stats = method.get_statistics()
                    
                elif args.method == "learnable":
                    from src.methods.learnable import LearnableCoTVector
                    method = LearnableCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        lambda_val=args.lambda_val,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        warmup_ratio=args.warmup_ratio,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        max_length=args.max_length,
                    )
                    vector = method.train(support_samples)
                    
                elif args.method == "self_evolved":
                    from src.methods.self_evolved import SelfEvolvedCoTVector
                    method = SelfEvolvedCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        rl_method=args.rl_method,
                        soft_reward=args.soft_reward,
                        init_std=args.init_std,
                        num_rollouts=args.group_size,
                        beta=args.beta,
                        learning_rate_vector=args.grpo_lr,
                        num_iterations=args.num_iterations,
                        questions_per_iter=args.questions_per_iter,
                        temperature=args.temperature,
                    )
                    vector = method.train(support_samples)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Layer {layer_idx}: CUDA OOM")
                    torch.cuda.empty_cache()
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy,
                        'error': "CUDA OOM",
                    })
                    continue
                raise
            except Exception as e:
                print(f"  Layer {layer_idx}: Error - {e}")
                results.append({
                    'layer': layer_idx,
                    'accuracy': 0,
                    'diff': -baseline_accuracy,
                    'error': str(e),
                })
                continue
        
        # Save vector if requested
        if args.save_vectors and vector is not None:
            vector_path = os.path.join(
                args.output_dir,
                f"{args.method}_{args.dataset}_L{layer_idx}.pt"
            )
            metadata = {"layer_idx": layer_idx, "method": args.method}
            if method_stats:
                metadata["stats"] = method_stats
            save_vector(vector, vector_path, metadata)
            vectors_dict[layer_idx] = vector_path
        
        # Evaluate
        try:
            layer_results = run_injection_evaluation(
                model_wrapper, tokenizer, test_samples, vector,
                layer_idx, args.dataset, args.scaling_factor, 
                args.max_new_tokens, args.num_beams, args.use_early_stopping
            )
            
            diff = layer_results['accuracy'] - baseline_accuracy
            vec_norm = vector.norm().item() if vector is not None else 0
            
            result_entry = {
                'layer': layer_idx,
                'accuracy': layer_results['accuracy'],
                'diff': diff,
                'correct': layer_results['correct'],
                'total': layer_results['total'],
                'vector_norm': vec_norm,
            }
            
            # Add UA-Vector specific stats
            if method_stats:
                result_entry['lambda_mean'] = method_stats.get('lambda_mean', 0)
                result_entry['suppressed_ratio'] = method_stats.get('suppressed_ratio', 0)
            
            results.append(result_entry)
            
            print(f"  Layer {layer_idx:2d}: {layer_results['accuracy']:.2f}% "
                  f"({layer_results['correct']}/{layer_results['total']}) "
                  f"[{diff:+.2f}% vs baseline] norm={vec_norm:.2f}")
            
        except Exception as e:
            print(f"  Layer {layer_idx:2d}: Eval Error - {e}")
            results.append({
                'layer': layer_idx,
                'accuracy': 0,
                'diff': -baseline_accuracy,
                'error': str(e),
            })
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Method:   {args.method}")
    if args.method == "ua_vector":
        print(f"Gamma:    {args.ua_gamma}")
    print(f"Dataset:  {args.dataset}")
    print(f"Baseline: {baseline_accuracy:.2f}%")
    print("-" * 70)
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        # Sort by accuracy
        valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        avg_norm = sum(r.get('vector_norm', 0) for r in valid_results) / len(valid_results)
        
        print(f"\nLayer-wise Average: {avg_accuracy:.2f}%")
        print(f"Average Vector Norm: {avg_norm:.2f}")
        
        print("\nTop 5 layers:")
        for r in valid_results[:5]:
            diff_str = f"({r['diff']:+.2f}%)"
            norm_str = f"norm={r.get('vector_norm', 0):.1f}"
            extra = ""
            if 'lambda_mean' in r:
                extra = f" λ={r['lambda_mean']:.3f}"
            print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str} {norm_str}{extra}")
        
        if len(valid_results) > 5:
            print("\nBottom 5 layers:")
            for r in valid_results[-5:]:
                diff_str = f"({r['diff']:+.2f}%)"
                norm_str = f"norm={r.get('vector_norm', 0):.1f}"
                print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str} {norm_str}")
        
        # Best layer
        best = valid_results[0]
        print(f"\n★ Best Layer: {best['layer']} with {best['accuracy']:.2f}%")
        print(f"  Improvement over baseline: {best['diff']:+.2f}%")
        print(f"  Vector norm: {best.get('vector_norm', 0):.2f}")
    
    # Error summary
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nErrors in {len(error_results)} layers:")
        for r in error_results:
            print(f"  Layer {r['layer']}: {r['error'][:50]}...")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir, 
        f"layer_sweep_{args.method}_{args.dataset}_{timestamp}.txt"
    )
    
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Method: {args.method}\n")
        if args.method == "ua_vector":
            f.write(f"Gamma: {args.ua_gamma}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Baseline: {baseline_accuracy:.2f}%\n")
        f.write(f"Support samples: {args.num_support_samples}\n")
        f.write(f"Test samples: {args.num_test_samples}\n\n")
        
        f.write(f"Layer\tAccuracy\tDiff\tCorrect\tTotal\tNorm\n")
        f.write("-" * 60 + "\n")
        
        for r in sorted(results, key=lambda x: x['layer']):
            if 'error' in r:
                f.write(f"{r['layer']}\tERROR\t-\t-\t-\t-\t{r['error'][:30]}\n")
            else:
                f.write(f"{r['layer']}\t{r['accuracy']:.2f}\t{r['diff']:+.2f}\t"
                       f"{r['correct']}\t{r['total']}\t{r.get('vector_norm', 0):.2f}\n")
        
        if valid_results:
            f.write(f"\n" + "=" * 60 + "\n")
            f.write(f"Best Layer: {valid_results[0]['layer']} ({valid_results[0]['accuracy']:.2f}%)\n")
            f.write(f"Average: {avg_accuracy:.2f}%\n")
    
    print(f"\nResults saved to {result_file}")
    
    # Save vectors paths
    if args.save_vectors and vectors_dict:
        vectors_file = os.path.join(
            args.output_dir,
            f"vectors_paths_{args.method}_{args.dataset}_{timestamp}.txt"
        )
        with open(vectors_file, "w") as f:
            for layer, path in sorted(vectors_dict.items()):
                f.write(f"Layer {layer}: {path}\n")
        print(f"Vector paths saved to {vectors_file}")
    
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()