"""
Inspect Rollouts with CoT Vector Injection.

This script allows loading a trained vector and inspecting the generated
Chain-of-Thought trajectories. It supports generating multiple rollouts
per question to analyze stability and diversity.

Usage:
    python inspect_rollouts.py --vector_path <path> --num_samples 5 --num_rollouts 4
"""

import argparse
import torch
import random
import re
from transformers import AutoTokenizer

# Import project modules
# Ensure src.models is accessible
from src.models import CoTModelWrapper
from src.data_utils import load_dataset, PROMPT_TEMPLATES
from src.utils import extract_answer_from_text, compare_answers

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect CoT Vector Rollouts")
    
    # Model and Data
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
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
        help="Dataset to inspect"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/data",
        help="Path to data directory"
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
        help="Layer index (if None, tries to infer from filename)"
    )
    
    # Inspection Configuration
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of different questions to inspect"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Number of trajectories to generate per question (to check stability)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum length of generation"
    )
    
    # Sampling Parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more diversity)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("CoT Vector Rollout Inspector (Multi-Trajectory)")
    print("=" * 70)
    print(f"Vector: {args.vector_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples (Questions): {args.num_samples}")
    print(f"Rollouts per Question: {args.num_rollouts}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)

    # 1. Load Tokenizer (for decoding)
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Load Model Wrapper
    # FIXED: Initialize wrapper with path, not model object
    print("Loading model wrapper (this loads the model)...")
    try:
        model_wrapper = CoTModelWrapper(args.model_path, model_name=args.model_name)
    except Exception as e:
        print(f"Error loading model wrapper: {e}")
        return
        
    print(f"Model loaded, device: {model_wrapper.device}")

    # 3. Load Vector
    print(f"\nLoading vector from {args.vector_path}...")
    try:
        loaded_data = torch.load(args.vector_path, map_location=model_wrapper.device)
        
        # Handle dict or raw tensor
        if isinstance(loaded_data, dict):
            vector = loaded_data.get("vector", loaded_data)
        else:
            vector = loaded_data
            
        vector = vector.to(dtype=model_wrapper.dtype, device=model_wrapper.device)
        print(f"Vector shape: {vector.shape}, norm: {vector.norm().item():.4f}")
        
    except Exception as e:
        print(f"Failed to load vector: {e}")
        return

    # Infer layer index if not provided
    layer_idx = args.layer_idx
    if layer_idx is None:
        match = re.search(r"_L(\d+)_", args.vector_path)
        if match:
            layer_idx = int(match.group(1))
            print(f"Inferred layer index from filename: {layer_idx}")
        else:
            layer_idx = 0
            print(f"Could not infer layer index, defaulting to {layer_idx}")
    
    # 4. Load Data
    print(f"\nLoading {args.dataset} test set...")
    try:
        samples = load_dataset(
            data_path=args.data_path,
            dataset_name=args.dataset,
            split="test",
            num_samples=None 
        )
        print(f"Loaded {len(samples)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Detailed error for debugging
        import traceback
        traceback.print_exc()
        return
    
    # Select random samples
    if len(samples) > args.num_samples:
        selected_samples = random.sample(samples, args.num_samples)
    else:
        selected_samples = samples
    
    # Prompt Template
    prompt_template = PROMPT_TEMPLATES.get(args.dataset, PROMPT_TEMPLATES["gsm8k"])

    # 5. Inspection Loop
    print("\n" + "=" * 70)
    print("INSPECTION START")
    print("=" * 70)

    for i, sample in enumerate(selected_samples):
        question = sample.question
        answer = sample.answer
        
        # Build prompt
        if args.dataset == "mmlu_pro" and sample.choices:
            prompt = prompt_template["cot"].format(question=question, choices=sample.choices)
        else:
            prompt = prompt_template["cot"].format(question=question)

        print(f"\n\n>>> SAMPLE {i+1}/{len(selected_samples)}")
        print("-" * 50)
        print(f"📝 QUESTION:\n{question.strip()}")
        print("-" * 50)
        print(f"✅ GROUND TRUTH: {answer}")
        print("-" * 50)
        
        correct_count = 0
        
        # Multiple Rollouts Loop
        for r in range(args.num_rollouts):
            # Clear previous hooks
            model_wrapper.clear_hooks()
            
            # Register injection hook
            model_wrapper.register_injection_hook(
                layer_idx=layer_idx,
                vector=vector,
                scaling_factor=1.0
            )
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model_wrapper.device)
            
            # Generate
            # Using model_wrapper.generate which delegates to self.model.generate
            with torch.no_grad():
                outputs = model_wrapper.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Evaluate
            extracted = extract_answer_from_text(generated_text, args.dataset)
            is_correct = compare_answers(extracted, answer, args.dataset)
            
            if is_correct:
                correct_count += 1
                status_icon = "✓"
            else:
                status_icon = "✗"
                
            print(f"\n--- Trajectory {r+1}/{args.num_rollouts} [{status_icon}] ---")
            print(generated_text.strip())
            print(f"\n[Extracted: {extracted}]")
        
        # Summary for this question
        consistency = (correct_count / args.num_rollouts) * 100
        print("-" * 50)
        print(f"📊 Consistency for this question: {consistency:.1f}% ({correct_count}/{args.num_rollouts})")
        
        # Cleanup
        model_wrapper.clear_hooks()

    print("\n" + "=" * 70)
    print("Inspection Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()