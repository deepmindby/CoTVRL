"""
Uncertainty-Aware CoT Vector (UA-Vector) Implementation.

This module implements the UA-Vector method based on the VI derivation document.
It extends the Extracted CoT Vector by incorporating second-order statistics (variance)
to perform Bayesian shrinkage on the mean vector.

Key Formulas:
    - Mean (First Moment): μ = (1/N) Σ v^(i)
    - Variance (Second Moment): σ² = (1/(N-1)) Σ (v^(i) - μ)² (unbiased)
    - Gating Coefficient: λ = 1 / (1 + γ · σ²)
    - Final Vector: v* = λ ⊙ μ

The core idea is:
    - Low variance dimensions: High confidence → λ ≈ 1 → Trust the mean
    - High variance dimensions: Low confidence → λ ≈ 0 → Suppress to zero (no intervention)
"""

import torch
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .extracted import ExtractedCoTVector
from ..models import CoTModelWrapper


class UncertaintyAwareCoTVector(ExtractedCoTVector):
    """
    Uncertainty-Aware CoT Vector using Bayesian shrinkage.
    
    This method computes an adaptive gating coefficient based on the variance
    of activation differences across the support set. High-variance (noisy)
    dimensions are suppressed, while low-variance (consistent) dimensions
    are preserved.
    
    Inherits from ExtractedCoTVector to reuse:
        - __init__ method (with additional parameters)
        - extract_single method (for computing individual sample vectors)
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        gamma: float = 1.0,
        normalize_variance: bool = True,
    ):
        """
        Initialize Uncertainty-Aware CoT Vector extractor.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index to extract CoT Vector from
            dataset_type: Type of dataset (gsm8k, math, mmlu_pro)
            gamma: Noise penalty factor γ. When γ=0, degrades to standard Extracted Vector.
                   Larger values penalize high-variance dimensions more aggressively.
            normalize_variance: Whether to normalize variance within the layer.
                               This makes γ more robust across different layers.
        """
        # Call parent class constructor
        super().__init__(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            dataset_type=dataset_type,
        )
        
        # UA-Vector specific parameters
        self.gamma = gamma
        self.normalize_variance = normalize_variance
        
        # Statistics storage for analysis
        self.stats: Dict[str, Any] = {}
    
    def extract(self, support_samples: List, wandb_run=None) -> torch.Tensor:
        """
        Extract Uncertainty-Aware CoT vector from support set.
        
        Implements the full UA-Vector pipeline:
        1. Collect all sample activation differences
        2. Compute mean (first moment) and variance (second moment)
        3. Optionally normalize variance for numerical stability
        4. Compute adaptive gating coefficient λ
        5. Apply Bayesian shrinkage: v* = λ ⊙ μ
        
        Args:
            support_samples: List of CoTSample objects from the support set
            wandb_run: Optional WandB run for logging statistics
            
        Returns:
            The uncertainty-aware CoT vector (torch.Tensor)
        """
        print(f"Extracting UA-Vector from {len(support_samples)} samples at layer {self.layer_idx}...")
        print(f"  Gamma (γ): {self.gamma}")
        print(f"  Normalize variance: {self.normalize_variance}")
        print("-" * 50)
        
        # ==================== Stage 1: Collect All Sample Vectors ====================
        print("[Stage 1/3] Collecting activation differences...")
        
        vectors = []
        failed_count = 0
        
        for sample in tqdm(support_samples, desc="Extracting samples", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                # Skip problematic samples but count failures
                failed_count += 1
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted! All samples failed.")
        
        if failed_count > 0:
            print(f"  Warning: {failed_count} samples failed during extraction")
        
        # Stack all vectors: V ∈ R^{N × D}
        V = torch.stack(vectors, dim=0)  # [N, hidden_dim]
        N, D = V.shape
        print(f"  Collected {N} vectors with dimension {D}")
        
        # ==================== Stage 2: Compute Statistics ====================
        print("[Stage 2/3] Computing statistics (mean and variance)...")
        
        # Compute first moment (mean): μ = (1/N) Σ v^(i)
        mu = V.mean(dim=0)  # [hidden_dim]
        
        # Compute second moment (unbiased variance): σ² = (1/(N-1)) Σ (v^(i) - μ)²
        # Using unbiased=True for unbiased estimation
        if N > 1:
            sigma_sq = V.var(dim=0, unbiased=True)  # [hidden_dim]
        else:
            # If only one sample, variance is undefined; use zero (no shrinkage)
            sigma_sq = torch.zeros_like(mu)
            print("  Warning: Only 1 sample, variance set to 0 (no shrinkage)")
        
        # Store raw statistics
        self.stats["raw_mean_norm"] = mu.norm().item()
        self.stats["raw_variance_mean"] = sigma_sq.mean().item()
        self.stats["raw_variance_max"] = sigma_sq.max().item()
        self.stats["raw_variance_min"] = sigma_sq.min().item()
        self.stats["raw_variance_std"] = sigma_sq.std().item()
        
        print(f"  Mean vector norm: {self.stats['raw_mean_norm']:.4f}")
        print(f"  Variance stats - mean: {self.stats['raw_variance_mean']:.6f}, "
              f"max: {self.stats['raw_variance_max']:.6f}, min: {self.stats['raw_variance_min']:.6f}")
        
        # Optionally normalize variance for numerical stability
        if self.normalize_variance:
            variance_mean = sigma_sq.mean()
            if variance_mean > 1e-10:  # Avoid division by zero
                sigma_sq_normalized = sigma_sq / variance_mean
                print(f"  Variance normalized by mean ({variance_mean:.6f})")
            else:
                sigma_sq_normalized = sigma_sq
                print("  Warning: Variance mean too small, skipping normalization")
        else:
            sigma_sq_normalized = sigma_sq
        
        self.stats["normalized_variance_mean"] = sigma_sq_normalized.mean().item()
        self.stats["normalized_variance_max"] = sigma_sq_normalized.max().item()
        
        # ==================== Stage 3: Compute Gating and Final Vector ====================
        print("[Stage 3/3] Computing gating coefficient and final vector...")
        
        # Compute gating coefficient: λ = 1 / (1 + γ · σ²)
        # When γ = 0: λ = 1 (no shrinkage, degrades to standard Extracted Vector)
        # When σ² → ∞: λ → 0 (full suppression)
        # When σ² → 0: λ → 1 (full trust)
        
        lambda_gate = 1.0 / (1.0 + self.gamma * sigma_sq_normalized)  # [hidden_dim]
        
        # Compute gating statistics
        self.stats["lambda_mean"] = lambda_gate.mean().item()
        self.stats["lambda_std"] = lambda_gate.std().item()
        self.stats["lambda_min"] = lambda_gate.min().item()
        self.stats["lambda_max"] = lambda_gate.max().item()
        
        # Count how many dimensions are effectively suppressed (λ < 0.5)
        suppressed_ratio = (lambda_gate < 0.5).float().mean().item()
        self.stats["suppressed_ratio"] = suppressed_ratio
        
        # Count how many dimensions are effectively preserved (λ > 0.9)
        preserved_ratio = (lambda_gate > 0.9).float().mean().item()
        self.stats["preserved_ratio"] = preserved_ratio
        
        print(f"  Gating λ stats - mean: {self.stats['lambda_mean']:.4f}, "
              f"min: {self.stats['lambda_min']:.4f}, max: {self.stats['lambda_max']:.4f}")
        print(f"  Suppressed dimensions (λ < 0.5): {suppressed_ratio * 100:.1f}%")
        print(f"  Preserved dimensions (λ > 0.9): {preserved_ratio * 100:.1f}%")
        
        # Compute final vector: v* = λ ⊙ μ (Hadamard product)
        v_star = lambda_gate * mu  # [hidden_dim]
        
        # Store final vector
        self.vector = v_star
        
        # Compute final statistics
        final_norm = v_star.norm().item()
        original_norm = mu.norm().item()
        norm_reduction = (1 - final_norm / (original_norm + 1e-10)) * 100
        
        self.stats["final_vector_norm"] = final_norm
        self.stats["original_vector_norm"] = original_norm
        self.stats["norm_reduction_percent"] = norm_reduction
        
        print("-" * 50)
        print(f"UA-Vector extraction complete!")
        print(f"  Original (Extracted) norm: {original_norm:.4f}")
        print(f"  Final (UA-Vector) norm: {final_norm:.4f}")
        print(f"  Norm reduction: {norm_reduction:.1f}%")
        
        # ==================== WandB Logging ====================
        if wandb_run is not None:
            self._log_to_wandb(wandb_run, lambda_gate, sigma_sq_normalized)
        
        return v_star
    
    def _log_to_wandb(self, wandb_run, lambda_gate: torch.Tensor, sigma_sq: torch.Tensor):
        """
        Log statistics and histograms to WandB for analysis.
        
        Args:
            wandb_run: The WandB run object
            lambda_gate: The gating coefficient vector
            sigma_sq: The (normalized) variance vector
        """
        try:
            import wandb
            
            # Log scalar statistics
            log_dict = {
                f"ua_vector/layer_{self.layer_idx}/gamma": self.gamma,
                f"ua_vector/layer_{self.layer_idx}/mean_norm": self.stats["raw_mean_norm"],
                f"ua_vector/layer_{self.layer_idx}/final_norm": self.stats["final_vector_norm"],
                f"ua_vector/layer_{self.layer_idx}/norm_reduction_percent": self.stats["norm_reduction_percent"],
                f"ua_vector/layer_{self.layer_idx}/lambda_mean": self.stats["lambda_mean"],
                f"ua_vector/layer_{self.layer_idx}/lambda_std": self.stats["lambda_std"],
                f"ua_vector/layer_{self.layer_idx}/suppressed_ratio": self.stats["suppressed_ratio"],
                f"ua_vector/layer_{self.layer_idx}/preserved_ratio": self.stats["preserved_ratio"],
                f"ua_vector/layer_{self.layer_idx}/variance_mean": self.stats["raw_variance_mean"],
                f"ua_vector/layer_{self.layer_idx}/variance_max": self.stats["raw_variance_max"],
            }
            
            # Log histograms for detailed analysis
            # Lambda distribution histogram
            lambda_np = lambda_gate.cpu().numpy()
            log_dict[f"ua_vector/layer_{self.layer_idx}/lambda_histogram"] = wandb.Histogram(lambda_np)
            
            # Variance distribution histogram (log scale for better visualization)
            sigma_np = sigma_sq.cpu().numpy()
            log_dict[f"ua_vector/layer_{self.layer_idx}/variance_histogram"] = wandb.Histogram(sigma_np)
            
            wandb_run.log(log_dict)
            print("  Statistics logged to WandB")
            
        except ImportError:
            print("  Warning: WandB not installed, skipping logging")
        except Exception as e:
            print(f"  Warning: Failed to log to WandB: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get the computed statistics for analysis.
        
        Returns:
            Dictionary containing all computed statistics
        """
        return self.stats.copy()
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the extracted UA-Vector."""
        return self.vector