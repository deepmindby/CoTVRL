"""
Multi-Layer Uncertainty-Aware CoT Vector implementation.

Extracts and applies UA-Vectors at multiple (or all) layers simultaneously.
Each layer has its own independent vector with layer-specific Bayesian shrinkage.

Key equations (per layer l):
    μ_l = (1/N) Σ v^(i)_l           # Layer-specific mean
    σ²_l = (1/(N-1)) Σ (v^(i)_l - μ_l)²  # Layer-specific variance
    λ_l = 1 / (1 + γ · σ²_l)        # Layer-specific gating
    v*_l = λ_l ⊙ μ_l                # Layer-specific final vector
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class MultiLayerUAVector(BaseCoTVectorMethod):
    """
    Multi-Layer Uncertainty-Aware CoT Vector.
    
    Extracts and applies UA-Vectors at multiple layers simultaneously.
    Can be configured to use all layers or a subset of layers.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_indices: Optional[List[int]] = None,  # None = all layers
        dataset_type: str = "gsm8k",
        gamma: float = 1.0,
        normalize_variance: bool = True,
        layer_weights: Optional[List[float]] = None,  # Optional per-layer scaling
    ):
        """
        Initialize Multi-Layer UA-Vector.
        
        Args:
            model_wrapper: The model wrapper with hook mechanisms
            tokenizer: Tokenizer for the model
            layer_indices: List of layer indices to use (None = all layers)
            dataset_type: Type of dataset for prompt templates
            gamma: Noise penalty factor γ for Bayesian shrinkage
            normalize_variance: Whether to normalize variance by layer mean
            layer_weights: Optional per-layer scaling factors (default: all 1.0)
        """
        # Use layer 0 as placeholder for parent class
        super().__init__(model_wrapper, tokenizer, 0, dataset_type)
        
        self.num_layers = model_wrapper.num_layers
        self.hidden_size = model_wrapper.hidden_size
        
        # Determine which layers to use
        if layer_indices is None:
            self.layer_indices = list(range(self.num_layers))
        else:
            self.layer_indices = sorted(layer_indices)
        
        self.gamma = gamma
        self.normalize_variance = normalize_variance
        
        # Per-layer weights (for scaling each layer's contribution)
        if layer_weights is not None:
            assert len(layer_weights) == len(self.layer_indices), \
                f"layer_weights length ({len(layer_weights)}) must match layer_indices ({len(self.layer_indices)})"
            self.layer_weights = layer_weights
        else:
            self.layer_weights = [1.0] * len(self.layer_indices)
        
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Storage for vectors and statistics
        self.vectors: Dict[int, torch.Tensor] = {}  # layer_idx -> vector
        self.layer_stats: Dict[int, Dict[str, Any]] = {}  # layer_idx -> stats
        
    def extract_single(self, sample) -> Dict[int, torch.Tensor]:
        """
        Extract activation differences from a single sample for all target layers.
        
        Returns:
            Dict mapping layer_idx to difference vector
        """
        device = self.model_wrapper.device
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        # Find answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        layer_diffs = {}
        
        # Extract CoT activations for all layers
        self.model_wrapper.clear_hooks()
        for layer_idx in self.layer_indices:
            self.model_wrapper.register_extraction_hook(layer_idx)
        
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        
        cot_activations = {}
        for layer_idx in self.layer_indices:
            act = self.model_wrapper.get_activations(layer_idx)
            cot_activations[layer_idx] = act[:, cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Extract non-CoT activations for all layers
        self.model_wrapper.clear_hooks()
        for layer_idx in self.layer_indices:
            self.model_wrapper.register_extraction_hook(layer_idx)
        
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        
        for layer_idx in self.layer_indices:
            act = self.model_wrapper.get_activations(layer_idx)
            non_cot_act = act[:, non_cot_answer_pos, :].mean(dim=1)  # [1, hidden]
            
            # Compute difference
            diff = cot_activations[layer_idx] - non_cot_act
            layer_diffs[layer_idx] = diff.squeeze(0)  # [hidden]
        
        self.model_wrapper.clear_hooks()
        
        return layer_diffs
    
    def extract(self, support_samples: List, wandb_run=None) -> Dict[int, torch.Tensor]:
        """
        Extract Multi-Layer UA-Vectors from support set.
        
        Args:
            support_samples: List of training samples
            wandb_run: Optional WandB run for logging
            
        Returns:
            Dict mapping layer_idx to UA-Vector
        """
        print(f"Extracting Multi-Layer UA-Vectors...")
        print(f"  Layers: {self.layer_indices} ({len(self.layer_indices)} layers)")
        print(f"  Samples: {len(support_samples)}")
        print(f"  Gamma (γ): {self.gamma}")
        print(f"  Normalize variance: {self.normalize_variance}")
        print("-" * 50)
        
        # Collect all difference vectors per layer
        layer_vectors: Dict[int, List[torch.Tensor]] = {l: [] for l in self.layer_indices}
        
        for sample in tqdm(support_samples, desc="Extracting", ncols=100):
            try:
                layer_diffs = self.extract_single(sample)
                for layer_idx, diff in layer_diffs.items():
                    layer_vectors[layer_idx].append(diff)
            except Exception as e:
                continue
        
        # Apply UA-Vector formula to each layer
        print("\nApplying Bayesian shrinkage per layer...")
        
        for layer_idx in tqdm(self.layer_indices, desc="Processing layers", ncols=100):
            vectors = layer_vectors[layer_idx]
            
            if len(vectors) < 2:
                print(f"  Layer {layer_idx}: Skipped (insufficient samples)")
                continue
            
            # Stack: [N, hidden_size]
            V = torch.stack(vectors, dim=0)
            N = V.shape[0]
            
            # Compute mean (first moment)
            mu = V.mean(dim=0)  # [hidden_size]
            
            # Compute unbiased variance (second moment)
            var = V.var(dim=0, unbiased=True)  # [hidden_size]
            
            # Normalize variance if requested
            if self.normalize_variance:
                var_mean = var.mean()
                if var_mean > 1e-8:
                    var_normalized = var / var_mean
                else:
                    var_normalized = var
            else:
                var_normalized = var
            
            # Compute lambda (gating coefficient)
            # λ = 1 / (1 + γ · σ²)
            lambda_gate = 1.0 / (1.0 + self.gamma * var_normalized)
            
            # Apply shrinkage: v* = λ ⊙ μ
            ua_vector = lambda_gate * mu
            
            # Get layer weight
            weight_idx = self.layer_indices.index(layer_idx)
            layer_weight = self.layer_weights[weight_idx]
            
            # Apply layer weight
            ua_vector = layer_weight * ua_vector
            
            self.vectors[layer_idx] = ua_vector
            
            # Store statistics
            original_norm = mu.norm().item()
            final_norm = ua_vector.norm().item()
            
            self.layer_stats[layer_idx] = {
                "num_samples": N,
                "original_vector_norm": original_norm,
                "final_vector_norm": final_norm,
                "norm_reduction_percent": (1 - final_norm / (original_norm + 1e-8)) * 100,
                "lambda_mean": lambda_gate.mean().item(),
                "lambda_std": lambda_gate.std().item(),
                "lambda_min": lambda_gate.min().item(),
                "lambda_max": lambda_gate.max().item(),
                "suppressed_ratio": (lambda_gate < 0.5).float().mean().item(),
                "preserved_ratio": (lambda_gate > 0.9).float().mean().item(),
                "variance_mean": var.mean().item(),
                "layer_weight": layer_weight,
            }
        
        # Print summary
        print("\n" + "=" * 60)
        print("Multi-Layer UA-Vector Summary")
        print("=" * 60)
        print(f"{'Layer':<8}{'Orig Norm':<12}{'Final Norm':<12}{'Reduction':<12}{'λ Mean':<10}")
        print("-" * 60)
        
        total_final_norm = 0
        for layer_idx in self.layer_indices:
            if layer_idx in self.layer_stats:
                s = self.layer_stats[layer_idx]
                print(f"{layer_idx:<8}{s['original_vector_norm']:<12.4f}{s['final_vector_norm']:<12.4f}"
                      f"{s['norm_reduction_percent']:<12.1f}%{s['lambda_mean']:<10.4f}")
                total_final_norm += s['final_vector_norm']
        
        print("-" * 60)
        print(f"Total final norm (sum): {total_final_norm:.4f}")
        print("=" * 60)
        
        # Log to WandB
        if wandb_run:
            for layer_idx, stats in self.layer_stats.items():
                for key, value in stats.items():
                    wandb_run.log({f"layer_{layer_idx}/{key}": value})
        
        return self.vectors
    
    def register_all_injection_hooks(self, scaling_factor: float = 1.0):
        """
        Register injection hooks for all extracted layers.
        
        Args:
            scaling_factor: Global scaling factor applied to all vectors
        """
        self.model_wrapper.clear_hooks()
        
        for layer_idx, vector in self.vectors.items():
            self.model_wrapper.register_injection_hook(
                layer_idx=layer_idx,
                vector=vector,
                scaling_factor=scaling_factor,
                requires_grad=False
            )
    
    def get_vector(self, layer_idx: Optional[int] = None) -> Optional[Union[torch.Tensor, Dict[int, torch.Tensor]]]:
        """
        Get extracted vector(s).
        
        Args:
            layer_idx: Specific layer index, or None for all vectors
            
        Returns:
            Single vector if layer_idx specified, else dict of all vectors
        """
        if layer_idx is not None:
            return self.vectors.get(layer_idx)
        return self.vectors
    
    def get_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get per-layer statistics."""
        return self.layer_stats
    
    def save(self, path: str):
        """Save all vectors and statistics to file."""
        save_dict = {
            "vectors": {k: v.cpu() for k, v in self.vectors.items()},
            "layer_stats": self.layer_stats,
            "config": {
                "layer_indices": self.layer_indices,
                "gamma": self.gamma,
                "normalize_variance": self.normalize_variance,
                "layer_weights": self.layer_weights,
                "dataset_type": self.dataset_type,
            }
        }
        torch.save(save_dict, path)
        print(f"Multi-layer vectors saved to {path}")
    
    @classmethod
    def load(cls, path: str, model_wrapper: CoTModelWrapper, tokenizer) -> "MultiLayerUAVector":
        """Load vectors from file."""
        data = torch.load(path, map_location="cpu")
        
        config = data["config"]
        instance = cls(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_indices=config["layer_indices"],
            dataset_type=config["dataset_type"],
            gamma=config["gamma"],
            normalize_variance=config["normalize_variance"],
            layer_weights=config.get("layer_weights"),
        )
        
        device = model_wrapper.device
        instance.vectors = {k: v.to(device) for k, v in data["vectors"].items()}
        instance.layer_stats = data["layer_stats"]
        
        return instance


class MultiLayerEvaluator:
    """
    Evaluator for multi-layer vector injection.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        multi_layer_vector: MultiLayerUAVector,
        dataset_type: str = "gsm8k",
        max_new_tokens: int = 512,
        num_beams: int = 3,
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.multi_layer_vector = multi_layer_vector
        self.dataset_type = dataset_type
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        from transformers import GenerationConfig
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if num_beams > 1:
            gen_kwargs["length_penalty"] = 0.0
        
        self.generation_config = GenerationConfig(**gen_kwargs)
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def evaluate_sample(self, sample, scaling_factor: float = 1.0) -> Dict[str, Any]:
        """Evaluate a single sample with multi-layer injection."""
        from ..utils import extract_answer_from_text, compare_answers
        
        # Build prompt
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        
        # Tokenize
        device = self.model_wrapper.device
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        input_len = input_ids.shape[1]
        
        # Register all injection hooks
        self.multi_layer_vector.register_all_injection_hooks(scaling_factor)
        
        # Generate
        with torch.no_grad():
            outputs = self.model_wrapper.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )
        
        # Decode
        generated_ids = outputs[0, input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract answer
        predicted = extract_answer_from_text(generated_text, self.dataset_type)
        is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        return {
            "predicted": predicted,
            "ground_truth": sample.answer,
            "correct": is_correct,
            "generated_text": generated_text,
            "num_tokens": len(generated_ids),
        }
    
    def evaluate_dataset(
        self,
        samples: List,
        scaling_factor: float = 1.0,
        desc: str = "Multi-Layer Eval"
    ) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        correct = 0
        total = len(samples)
        results = []
        
        pbar = tqdm(samples, desc=desc, ncols=100)
        for sample in pbar:
            result = self.evaluate_sample(sample, scaling_factor)
            results.append(result)
            
            if result["correct"]:
                correct += 1
            
            acc = correct / len(results) * 100
            pbar.set_postfix({"acc": f"{acc:.1f}%"})
        
        accuracy = correct / total * 100
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }