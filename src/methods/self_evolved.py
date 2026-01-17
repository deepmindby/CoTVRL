"""
Self-Evolved CoT Vector via RL-based Task Vector Search.

This module implements the SelfEvolvedCoTVector class that uses either
GRPO or DAPO to learn a static activation vector inducing CoT reasoning.

The LLM parameters are FROZEN; only the vector v is trainable.

Key features:
- Clean, modular design with decoupled RL solvers
- Support for both GRPO and DAPO algorithms
- Optional initialization from extracted vectors
- Memory-efficient training with gradient accumulation
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import random
import gc

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES
from ..utils import extract_answer_from_text, compare_answers
from ..rl_solvers import GRPOSolver, DAPOSolver


class SelfEvolvedCoTVector(BaseCoTVectorMethod):
    """
    Self-Evolved CoT Vector via RL-based optimization.
    
    Searches for a static activation vector v that, when added to a
    specific layer's hidden states, induces the LLM to perform
    Chain-of-Thought reasoning and output correct answers.
    
    Supports two RL algorithms:
    - GRPO: Group Relative Policy Optimization
    - DAPO: Direct Alignment Policy Optimization (DPO-like)
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        # RL method selection
        rl_method: str = "grpo",
        # Reward configuration
        soft_reward: bool = False,
        # Vector initialization
        init_from_extracted: bool = False,
        extracted_vector_path: Optional[str] = None,
        init_std: float = 0.0,
        # RL hyperparameters
        num_rollouts: int = 8,
        beta: float = 0.0,
        learning_rate_vector: float = 1e-2,
        max_grad_norm: float = 1.0,
        # Training configuration
        num_iterations: int = 100,
        questions_per_iter: int = 4,
        # Generation parameters
        rl_max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize Self-Evolved CoT Vector trainer.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for vector injection
            dataset_type: Dataset type (gsm8k, math, mmlu_pro)
            rl_method: RL algorithm ('grpo' or 'dapo')
            soft_reward: Use partial rewards if True, binary if False
            init_from_extracted: Initialize from pre-extracted vector
            extracted_vector_path: Path to extracted vector file
            init_std: Std for random initialization (0.0 = zero init)
            num_rollouts: Group size G for RL
            beta: KL penalty (GRPO) or temperature (DAPO)
            learning_rate_vector: Learning rate for vector
            max_grad_norm: Max gradient norm for clipping
            num_iterations: Number of training iterations
            questions_per_iter: Questions per iteration
            rl_max_new_tokens: Max tokens for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        # Store configuration
        self.rl_method = rl_method.lower()
        self.soft_reward = soft_reward
        self.num_rollouts = num_rollouts
        self.beta = beta
        self.learning_rate_vector = learning_rate_vector
        self.max_grad_norm = max_grad_norm
        self.num_iterations = num_iterations
        self.questions_per_iter = questions_per_iter
        self.rl_max_new_tokens = rl_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Get target device from model
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        self.target_device = next(target_layer.parameters()).device
        
        # Initialize vector
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        
        if init_from_extracted and extracted_vector_path:
            self._load_extracted_vector(extracted_vector_path)
        elif init_std > 0:
            nn.init.normal_(self.vector_param, std=init_std)
        # else: zero initialization (default)
        
        self.vector_param.data = self.vector_param.data.to(
            device=self.target_device,
            dtype=torch.float32
        )
        
        # Initialize RL solver
        self.solver = self._create_solver()
    
    def _load_extracted_vector(self, path: str) -> None:
        """Load pre-extracted vector for initialization."""
        if not os.path.exists(path):
            print(f"Warning: Extracted vector not found at {path}, using zero init")
            return
        
        loaded = torch.load(path, map_location="cpu")
        
        # Handle different save formats
        if isinstance(loaded, dict) and "vector" in loaded:
            vec = loaded["vector"]
        else:
            vec = loaded
        
        self.vector_param.data = vec.float()
        print(f"Loaded extracted vector from {path}, norm={vec.norm().item():.4f}")
    
    def _create_solver(self):
        """Create the appropriate RL solver based on configuration."""
        if self.rl_method == "grpo":
            return GRPOSolver(
                model_wrapper=self.model_wrapper,
                tokenizer=self.tokenizer,
                layer_idx=self.layer_idx,
                soft_reward=self.soft_reward,
                beta=self.beta,
            )
        elif self.rl_method == "dapo":
            return DAPOSolver(
                model_wrapper=self.model_wrapper,
                tokenizer=self.tokenizer,
                layer_idx=self.layer_idx,
                beta=self.beta if self.beta > 0 else 0.1,
            )
        else:
            raise ValueError(f"Unknown RL method: {self.rl_method}")
    
    def _build_prompt(self, sample) -> str:
        """Build prompt from sample."""
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        return prompt
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """
        Train the self-evolved CoT vector using RL.
        
        Args:
            support_samples: List of training samples
            wandb_run: Optional WandB run for logging
            
        Returns:
            Trained vector tensor
        """
        print(f"Training Self-Evolved CoT Vector at layer {self.layer_idx}")
        print(f"  RL Method: {self.rl_method.upper()}")
        print(f"  Samples: {len(support_samples)}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Rollouts per question (G): {self.num_rollouts}")
        print(f"  Questions per iteration: {self.questions_per_iter}")
        print(f"  Learning rate: {self.learning_rate_vector}")
        print(f"  Soft reward: {self.soft_reward}")
        print(f"  Initial vector norm: {self.vector_param.norm().item():.4f}")
        print("-" * 50)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate_vector,
            weight_decay=1e-4,
        )
        
        # Training metrics
        best_avg_reward = -float('inf')
        best_vector = None
        reward_history = []
        
        # Progress bar
        pbar = tqdm(range(self.num_iterations), desc=f"{self.rl_method.upper()}", ncols=100)
        
        for iteration in pbar:
            iter_losses = []
            iter_rewards = []
            skipped = 0
            
            # Sample questions for this iteration
            batch_samples = random.sample(
                support_samples,
                min(self.questions_per_iter, len(support_samples))
            )
            
            for sample in batch_samples:
                prompt = self._build_prompt(sample)
                ground_truth = sample.answer
                
                try:
                    # Call solver step
                    metrics = self.solver.step(
                        prompt=prompt,
                        ground_truth=ground_truth,
                        vector=self.vector_param,
                        optimizer=optimizer,
                        max_new_tokens=self.rl_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        num_rollouts=self.num_rollouts,
                        extract_answer_fn=extract_answer_from_text,
                        compare_answers_fn=compare_answers,
                        dataset_type=self.dataset_type,
                        max_grad_norm=self.max_grad_norm,
                    )
                    
                    if metrics.get("skipped", False):
                        skipped += 1
                    else:
                        iter_losses.append(metrics.get("loss", 0.0))
                    
                    # Track rewards
                    if "mean_reward" in metrics:
                        iter_rewards.append(metrics["mean_reward"])
                    elif "accuracy" in metrics:
                        iter_rewards.append(metrics["accuracy"])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        gc.collect()
                        skipped += 1
                        continue
                    raise
            
            # Compute iteration statistics
            avg_loss = sum(iter_losses) / max(len(iter_losses), 1)
            avg_reward = sum(iter_rewards) / max(len(iter_rewards), 1)
            current_norm = self.vector_param.norm().item()
            
            reward_history.append(avg_reward)
            
            # Track best
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_vector = self.vector_param.detach().clone()
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{avg_loss:.3f}",
                "Reward": f"{avg_reward:.2f}",
                "Norm": f"{current_norm:.1f}",
                "Skip": skipped,
            })
            
            # WandB logging
            if wandb_run and iteration % 5 == 0:
                wandb_run.log({
                    "iteration": iteration,
                    "train/loss": avg_loss,
                    "train/reward": avg_reward,
                    "train/best_reward": best_avg_reward,
                    "train/vector_norm": current_norm,
                    "train/skipped": skipped,
                })
            
            # Periodic memory cleanup
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Use best vector
        self.vector = best_vector if best_vector is not None else self.vector_param.detach().clone()
        
        # Final summary
        print("\n" + "=" * 50)
        print("Training Complete")
        print("=" * 50)
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Final vector norm: {self.vector.norm().item():.4f}")
        print("=" * 50)
        
        return self.vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the trained CoT vector."""
        return self.vector