"""
DAPO (Direct Alignment Policy Optimization) Solver for Task Vector Training.

This module implements a DPO-like algorithm for learning a static activation vector
that induces Chain-of-Thought reasoning in LLMs.

Key Algorithm:
1. Sample G completions per question
2. Identify Winner (correct) and Loser (incorrect) pairs
3. If no valid pair exists, skip this batch
4. Compute DPO-style loss to maximize margin between chosen/rejected

Note: Since we don't have a reference model, we use a simplified formulation
that focuses on maximizing the margin between chosen and rejected responses
given the task vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import gc


class DAPOSolver:
    """
    DAPO Solver for training task vectors via direct alignment optimization.
    
    The LLM parameters are FROZEN; only the vector v is trainable.
    """
    
    def __init__(
        self,
        model_wrapper,
        tokenizer,
        layer_idx: int,
        beta: float = 0.1,
        epsilon: float = 1e-8,
    ):
        """
        Initialize DAPO Solver.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for vector injection
            beta: Temperature parameter for DPO loss
            epsilon: Small value for numerical stability
        """
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.beta = beta if beta > 0 else 0.1  # Ensure beta is positive
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
    ) -> List[Tuple[str, torch.Tensor, int, bool]]:
        """
        Generate multiple rollouts and evaluate correctness.
        
        Args:
            prompt: The input prompt string
            vector: The task vector to inject
            num_rollouts: Number of completions to generate (G)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of (generated_text, generated_ids, num_tokens, is_correct) tuples
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
                vector.detach(),
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
            
            # is_correct will be filled later
            rollouts.append((generated_text, generated_ids, num_tokens, None))
            
            self.model_wrapper.clear_hooks()
        
        return rollouts
    
    def construct_pairs(
        self,
        rollouts: List[Tuple[str, torch.Tensor, int, bool]],
        ground_truth: str,
        extract_answer_fn,
        compare_answers_fn,
        dataset_type: str,
    ) -> Optional[Tuple[Tuple, Tuple]]:
        """
        Construct Winner-Loser pairs from rollouts.
        
        Args:
            rollouts: List of rollout tuples
            ground_truth: Correct answer
            extract_answer_fn: Function to extract answers
            compare_answers_fn: Function to compare answers
            dataset_type: Dataset type
            
        Returns:
            Tuple of (winner_tuple, loser_tuple) or None if no valid pair
        """
        winners = []
        losers = []
        
        for generated_text, generated_ids, num_tokens, _ in rollouts:
            predicted = extract_answer_fn(generated_text, dataset_type)
            is_correct = compare_answers_fn(predicted, ground_truth, dataset_type)
            
            if is_correct:
                winners.append((generated_text, generated_ids, num_tokens, True))
            else:
                losers.append((generated_text, generated_ids, num_tokens, False))
        
        # Need at least one winner and one loser
        if len(winners) == 0 or len(losers) == 0:
            return None
        
        # Select the first winner and first loser (could also randomly select)
        return winners[0], losers[0]
    
    def compute_log_probs(
        self,
        prompt: str,
        generated_ids: torch.Tensor,
        vector: nn.Parameter,
    ) -> torch.Tensor:
        """
        Compute log probabilities of generated tokens (with gradients).
        
        Args:
            prompt: The input prompt
            generated_ids: The generated token ids
            vector: The trainable task vector
            
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
        
        # Construct full sequence
        full_ids = torch.cat([input_ids.squeeze(0), generated_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        
        # Register injection WITH gradient
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_injection_hook(
            self.layer_idx,
            vector,
            scaling_factor=1.0,
            requires_grad=True
        )
        
        # Forward pass
        outputs = self.model_wrapper(full_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute log probs for generated part
        shift_logits = logits[:, input_len-1:-1, :].contiguous()
        shift_labels = full_ids[:, input_len:].contiguous()
        
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mean log probability (per-token, for better comparison)
        total_log_prob = token_log_probs.mean()
        
        self.model_wrapper.clear_hooks()
        
        return total_log_prob
    
    def compute_dpo_loss(
        self,
        prompt: str,
        winner: Tuple,
        loser: Tuple,
        vector: nn.Parameter,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO-style loss.
        
        Loss = -log σ(β * (log π(y_w|x) - log π(y_l|x)))
        
        Note: Without a reference model, we simply maximize the margin
        between chosen (winner) and rejected (loser) responses.
        
        Args:
            prompt: Input prompt
            winner: Winner (correct) rollout tuple
            loser: Loser (incorrect) rollout tuple
            vector: Trainable task vector
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        _, winner_ids, _, _ = winner
        _, loser_ids, _, _ = loser
        
        # Compute log probs for winner and loser
        log_prob_winner = self.compute_log_probs(prompt, winner_ids, vector)
        log_prob_loser = self.compute_log_probs(prompt, loser_ids, vector)
        
        # DPO loss: -log σ(β * (log π_w - log π_l))
        log_ratio = log_prob_winner - log_prob_loser
        loss = -F.logsigmoid(self.beta * log_ratio)
        
        metrics = {
            "log_prob_winner": log_prob_winner.item(),
            "log_prob_loser": log_prob_loser.item(),
            "log_ratio": log_ratio.item(),
            "margin": (log_prob_winner - log_prob_loser).item(),
        }
        
        return loss, metrics
    
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
        Perform one DAPO training step for a single question.
        
        Pipeline:
        1. Generate rollouts
        2. Construct winner/loser pairs
        3. Compute DPO loss and update vector
        
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
            Dict containing metrics
        """
        # Step 1: Generate rollouts
        rollouts = self.generate_rollouts(
            prompt, vector, num_rollouts,
            max_new_tokens, temperature, top_p
        )
        
        # Step 2: Construct winner/loser pairs
        pair = self.construct_pairs(
            rollouts, ground_truth,
            extract_answer_fn, compare_answers_fn, dataset_type
        )
        
        # Count correct answers for metrics
        correct_count = sum(
            1 for text, ids, ntok, _ in rollouts
            if compare_answers_fn(
                extract_answer_fn(text, dataset_type),
                ground_truth,
                dataset_type
            )
        )
        
        step_metrics = {
            "num_correct": correct_count,
            "num_rollouts": num_rollouts,
            "accuracy": correct_count / num_rollouts,
            "loss": 0.0,
            "skipped": pair is None,
        }
        
        if pair is None:
            # No valid pair (all correct or all incorrect)
            return step_metrics
        
        winner, loser = pair
        
        # Step 3: Compute DPO loss
        optimizer.zero_grad()
        
        loss, loss_metrics = self.compute_dpo_loss(prompt, winner, loser, vector)
        
        # Step 4: Backward and optimize
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