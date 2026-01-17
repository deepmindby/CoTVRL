"""
Transfer Evaluation Script for CoT Vectors.

This script evaluates a trained CoT Vector on different datasets (Transfer Learning).
It performs a comparative analysis:
1. Baseline Evaluation (Original Model)
2. Injection Evaluation (Model + CoT Vector)

Supported Datasets: gsm8k, math, mmlu_pro
"""

import argparse
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Tuple

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoT Vector Transfer Performance")
    
    # Model Configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B",
        help="Path to the pretrained model directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen",
        choices=["qwen", "llama"],
        help="Model architecture name"
    )
    
    # Vector Configuration
    parser.add_argument(
        "--vector_path",
        type=str,
        required=True,
        help="Path to the trained vector file (.pt)"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        help="Target layer index for injection. If None, attempts to read from vector file metadata."
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=1.0,
        help="Scaling factor for the vector injection (strength)"
    )
    
    # Dataset Configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "math", "math_easy", "math_hard", "mmlu_pro"],
        help="Target dataset for evaluation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/data",
        help="Path to the data directory containing gsm8k/math/mmlu_pro folders"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (set to -1 for full test set)"
    )
    
    # Generation Configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for generation"
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Enable early stopping for beam search"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def load_vector(path: str, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    Load the vector and extract metadata if available.
    Returns: (vector_tensor, metadata_dict)
    """
    print(f"Loading vector from {path}...")
    data = torch.load(path, map_location="cpu") # Load to CPU first
    
    metadata = {}
    vector = None
    
    if isinstance(data, dict) and "vector" in data:
        vector = data["vector"]
        # Try to extract args from saved metadata
        if "args" in data:
            metadata = data["args"]
            print(f"Found metadata in vector file: Layer {metadata.get('layer_idx', 'Unknown')}")
    elif isinstance(data, torch.Tensor):
        vector = data
    else:
        raise ValueError(f"Unknown vector file format. Expected dict with 'vector' key or raw Tensor.")
    
    return vector.to(device), metadata

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("CoT Vector Transfer Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Vector: {args.vector_path}")
    
    # 1. Load Model and Tokenizer
    print("\n[1/4] Loading Model and Tokenizer...")
    try:
        # Initialize Wrapper (handles model loading internally)
        model_wrapper = CoTModelWrapper(args.model_path, model_name=args.model_name)
        tokenizer = load_tokenizer(args.model_path)
        print(f"Model loaded successfully. Device: {model_wrapper.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Vector
    print("\n[2/4] Loading Task Vector...")
    try:
        vector, metadata = load_vector(args.vector_path, model_wrapper.device)
        vector = vector.to(dtype=model_wrapper.dtype)
        print(f"Vector Shape: {vector.shape}, Norm: {vector.norm().item():.4f}")
        
        # Determine Layer Index
        # Priority: Command line arg > Metadata in file > Error
        if args.layer_idx is not None:
            layer_idx = args.layer_idx
            print(f"Using provided layer index: {layer_idx}")
        elif "layer_idx" in metadata:
            layer_idx = metadata["layer_idx"]
            print(f"Using layer index from file metadata: {layer_idx}")
        else:
            # Try to infer from filename
            import re
            match = re.search(r"_L(\d+)_", args.vector_path)
            if match:
                layer_idx = int(match.group(1))
                print(f"Inferred layer index from filename: {layer_idx}")
            else:
                raise ValueError("Layer index not specified and could not be inferred. Please use --layer_idx.")
                
    except Exception as e:
        print(f"Error loading vector: {e}")
        return

    # 3. Load Dataset
    print(f"\n[3/4] Loading {args.dataset} Dataset...")
    try:
        # Determine split based on dataset type
        # For evaluation, we typically use 'test' or 'validation'
        split = "test"
        if args.dataset == "mmlu_pro":
            # MMLU-Pro usually uses 'test' or 'validation'
            split = "test" 
        
        samples = load_dataset(
            data_path=args.data_path,
            dataset_name=args.dataset,
            split=split,
            num_samples=args.num_samples if args.num_samples > 0 else None
        )
        print(f"Loaded {len(samples)} samples from {split} set.")
        
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        print("Tip: Check if the jsonl files exist in the data directory.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. Evaluation Loop
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    
    # --- Phase A: Baseline (No Vector) ---
    print("\n>>> Phase A: Baseline Evaluation (Original Model)")
    baseline_acc = 0.0
    try:
        baseline_results = run_baseline_evaluation(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=samples,
            dataset_type=args.dataset,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            use_early_stopping=args.use_early_stopping,
            # Removed batch_size
        )
        baseline_acc = baseline_results['accuracy']
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Phase B: Injection (With Vector) ---
    print(f"\n>>> Phase B: Injection Evaluation (Layer {layer_idx}, Scale {args.scaling_factor})")
    injection_acc = 0.0
    try:
        injection_results = run_injection_evaluation(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=samples,
            vector=vector,
            layer_idx=layer_idx,
            dataset_type=args.dataset,
            scaling_factor=args.scaling_factor,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            use_early_stopping=args.use_early_stopping,
            # Removed batch_size
        )
        injection_acc = injection_results['accuracy']
        print(f"Injection Accuracy: {injection_acc:.2f}%")
    except Exception as e:
        print(f"Injection evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Final Report
    print("\n" + "=" * 60)
    print("FINAL TRANSFER RESULTS")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Vector:     {os.path.basename(args.vector_path)}")
    print(f"Layer:      {layer_idx}")
    print("-" * 60)
    print(f"Baseline:   {baseline_acc:.2f}%")
    print(f"Injection:  {injection_acc:.2f}%")
    
    delta = injection_acc - baseline_acc
    sign = "+" if delta >= 0 else ""
    print(f"Difference: {sign}{delta:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
