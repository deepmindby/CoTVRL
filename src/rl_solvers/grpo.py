"""
GRPO (Group Relative Policy Optimization) Solver for Task Vector Training.

This module implements the GRPO algorithm for learning a static activation vector
that induces Chain-of-Thought reasoning in LLMs.

Key Algorithm:
1. Sample G completions per question (no grad)
2. Compute rewards based on answer correctness
3. Normalize rewards within group to get advantages
4. Re-run forward pass with gradients to compute log probabilities
5. Compute GRPO loss: -1/G * sum(A_i * sum(log π(a_i|x)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import gc


class GRPOSolver:
    """
    GRPO Solver for training task vectors via group relative policy optimization.
    
    The LLM parameters are FROZEN; only the vector v is trainable.
    """
    
    def __init__(
        self,
        model_wrapper,
        tokenizer,
        layer_idx: int,
        soft_reward: bool = False,
        beta: float = 0.0,
        epsilon: float = 1e-8,
    ):
        """
        Initialize GRPO Solver.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for vector injection
            soft_reward: If True, use partial rewards; if False, binary 0/1
            beta: KL penalty coefficient (default 0, disabled)
            epsilon: Small value for numerical stability
        """
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.soft_reward = soft_reward
        self.beta = beta
        self.epsilon = epsilon
        
        # Get target device from the model layer
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        self.device = next(target_layer.parameters()).device
    
    def generate_rollouts(
        self,
        prompt: str,
        vector: torch.Tensor,
        num_rollouts: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Tuple[str, torch.Tensor, int]]:
        """
        Generate multiple rollouts for a single prompt (no gradient).
        
        Args:
            prompt: The input prompt string
            vector: The task vector to inject
            num_rollouts: Number of completions to generate (G)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of (generated_text, generated_ids, num_tokens) tuples
        """
        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]
        
        rollouts = []
        
        for _ in range(num_rollouts):
            # Clear hooks and register injection (no grad for generation)
            self.model_wrapper.clear_hooks()
            self.model_wrapper.register_injection_hook(
                self.layer_idx,
                vector.detach(),  # Detach to avoid building computation graph
                scaling_factor=1.0,
                requires_grad=False
            )
            
            # Generate completion
            with torch.no_grad():
                outputs = self.model_wrapper.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated portion
            generated_ids = outputs[0, input_len:].clone()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_tokens = len(generated_ids)
            
            rollouts.append((generated_text, generated_ids, num_tokens))
            
            self.model_wrapper.clear_hooks()
        
        return rollouts
    
    def compute_rewards(
        self,
        rollouts: List[Tuple[str, torch.Tensor, int]],
        ground_truth: str,
        extract_answer_fn,
        compare_answers_fn,
        dataset_type: str,
    ) -> torch.Tensor:
        """
        Compute rewards for each rollout.
        
        Args:
            rollouts: List of (generated_text, generated_ids, num_tokens)
            ground_truth: The correct answer
            extract_answer_fn: Function to extract answer from text
            compare_answers_fn: Function to compare predicted vs ground truth
            dataset_type: Type of dataset for answer extraction
            
        Returns:
            Tensor of rewards [num_rollouts]
        """
        rewards = []
        
        for generated_text, _, num_tokens in rollouts:
            predicted = extract_answer_fn(generated_text, dataset_type)
            is_correct = compare_answers_fn(predicted, ground_truth, dataset_type)
            
            if self.soft_reward:
                # Partial rewards for format/length
                reward = 0.0
                if is_correct:
                    reward = 1.0
                elif predicted is not None:
                    # Valid format but wrong answer
                    reward = 0.1
                # Additional small reward for reasonable length
                if 50 <= num_tokens <= 400:
                    reward += 0.05
            else:
                # Binary reward
                reward = 1.0 if is_correct else 0.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)
    
    def compute_advantages(self, rewards: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Compute normalized advantages within the group.
        
        Args:
            rewards: Tensor of rewards [num_rollouts]
            
        Returns:
            Tuple of (advantages tensor, is_valid flag)
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        # Check if std is too small (all rewards are the same)
        if std_reward < self.epsilon:
            # Cannot compute meaningful advantages
            return torch.zeros_like(rewards), False
        
        # Normalize to mean=0, std=1
        advantages = (rewards - mean_reward) / (std_reward + self.epsilon)
        
        return advantages, True
    
    def compute_log_probs(
        self,
        prompt: str,
        generated_ids: torch.Tensor,
        vector: nn.Parameter,
    ) -> torch.Tensor:
        """
        Compute log probabilities of generated tokens (with gradients).
        
        This is the crucial step where we re-run the forward pass to get gradients.
        
        Args:
            prompt: The input prompt
            generated_ids: The generated token ids
            vector: The trainable task vector (nn.Parameter)
            
        Returns:
            Sum of log probabilities for the generated sequence
        """
        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        input_ids = encoding["input_ids"].to(self.device)
        input_len = input_ids.shape[1]
        
        # Construct full sequence: [prompt_ids, generated_ids]
        full_ids = torch.cat([input_ids.squeeze(0), generated_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        
        # Clear hooks and register injection WITH gradient
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_injection_hook(
            self.layer_idx,
            vector,  # The nn.Parameter, requires grad
            scaling_factor=1.0,
            requires_grad=True
        )
        
        # Forward pass
        outputs = self.model_wrapper(full_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute log probabilities for the generated part only
        # Shift: logits[..., input_len-1:-1, :] predicts tokens at positions [input_len, end]
        shift_logits = logits[:, input_len-1:-1, :].contiguous()
        shift_labels = full_ids[:, input_len:].contiguous()
        
        # Log softmax to get log probabilities
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        
        # Gather log probs for actual generated tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum of log probabilities (sequence level)
        total_log_prob = token_log_probs.sum()
        
        self.model_wrapper.clear_hooks()
        
        return total_log_prob
    
    def compute_grpo_loss(
        self,
        prompt: str,
        rollouts: List[Tuple[str, torch.Tensor, int]],
        advantages: torch.Tensor,
        vector: nn.Parameter,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with gradient accumulation to save memory.
        
        Loss = -1/G * sum(A_i * sum(log π(a_i|x)))
        
        Args:
            prompt: The input prompt
            rollouts: List of (generated_text, generated_ids, num_tokens)
            advantages: Normalized advantages for each rollout
            vector: The trainable task vector
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        G = len(rollouts)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        metrics = {
            "num_valid_rollouts": 0,
            "avg_log_prob": 0.0,
        }
        
        for i, (_, generated_ids, _) in enumerate(rollouts):
            # Skip if advantage is too small
            if abs(advantages[i].item()) < 0.01:
                continue
            
            # Compute log probability for this rollout
            log_prob = self.compute_log_probs(prompt, generated_ids, vector)
            
            # GRPO loss component: -A_i * log π
            sample_loss = -advantages[i] * log_prob
            total_loss = total_loss + sample_loss / G
            
            metrics["num_valid_rollouts"] += 1
            metrics["avg_log_prob"] += log_prob.item()
        
        if metrics["num_valid_rollouts"] > 0:
            metrics["avg_log_prob"] /= metrics["num_valid_rollouts"]
        
        return total_loss, metrics
    
    def step(
        self,
        prompt: str,
        ground_truth: str,
        vector: nn.Parameter,
        optimizer: torch.optim.Optimizer,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_rollouts: int,
        extract_answer_fn,
        compare_answers_fn,
        dataset_type: str,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform one GRPO training step for a single question.
        
        This method handles the full pipeline:
        1. Generate rollouts
        2. Compute rewards
        3. Compute advantages
        4. Compute loss and update vector
        
        Args:
            prompt: Input prompt
            ground_truth: Correct answer
            vector: Trainable task vector
            optimizer: Optimizer for vector
            max_new_tokens: Max tokens for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_rollouts: Group size G
            extract_answer_fn: Function to extract answers
            compare_answers_fn: Function to compare answers
            dataset_type: Dataset type
            max_grad_norm: Max gradient norm for clipping
            
        Returns:
            Dict containing metrics (loss, reward, etc.)
        """
        # Step 1: Generate rollouts (no grad)
        rollouts = self.generate_rollouts(
            prompt, vector, num_rollouts,
            max_new_tokens, temperature, top_p
        )
        
        # Step 2: Compute rewards
        rewards = self.compute_rewards(
            rollouts, ground_truth,
            extract_answer_fn, compare_answers_fn, dataset_type
        )
        
        # Step 3: Compute advantages
        advantages, is_valid = self.compute_advantages(rewards)
        
        # Metrics to return
        step_metrics = {
            "rewards": rewards.tolist(),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "loss": 0.0,
            "skipped": not is_valid,
        }
        
        if not is_valid:
            # All rewards are the same, skip this question
            return step_metrics
        
        # Step 4: Compute GRPO loss
        optimizer.zero_grad()
        
        loss, loss_metrics = self.compute_grpo_loss(
            prompt, rollouts, advantages, vector
        )
        
        # Step 5: Backward and optimize
        if loss.requires_grad:
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_([vector], max_norm=max_grad_norm)
            
            optimizer.step()
            
            step_metrics["loss"] = loss.item()
            step_metrics["grad_norm"] = grad_norm.item()
            step_metrics.update(loss_metrics)
        
        # Clean up
        self.model_wrapper.clear_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        
        return step_metrics