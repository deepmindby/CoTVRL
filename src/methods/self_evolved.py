"""
Self-Evolved CoT Vector via GRPO (Group Relative Policy Optimization).

优化版本V3 - 解决显存问题：
1. 减小默认batch参数
2. 每个样本独立backward，不累积计算图
3. 截断生成序列长度以减少log_prob计算
4. 更频繁的显存清理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
import random
import math
import gc

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES
from ..utils import extract_answer_from_text, compare_answers


class SelfEvolvedCoTVector(BaseCoTVectorMethod):
    """
    Self-Evolved CoT Vector optimized via GRPO.
    显存优化版本。
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        group_size: int = 4,  # 减小: 8 -> 4
        num_iterations: int = 100,
        learning_rate: float = 5e-3,
        beta: float = 0.01,
        epsilon: float = 1e-8,
        max_new_tokens: int = 512,
        questions_per_iter: int = 2,  # 减小: 4 -> 2
        temperature: float = 0.7,
        init_std: float = 0.35,
        init_from_extracted: Optional[torch.Tensor] = None,
        use_soft_reward: bool = True,
        exploration_noise: float = 0.1,
        min_reward_std: float = 0.05,
        gradient_accumulation: int = 1,  # 减小: 2 -> 1 (立即更新)
        max_log_prob_tokens: int = 128,  # 新增: 只用最后N个token计算log_prob
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.group_size = group_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.questions_per_iter = questions_per_iter
        self.temperature = temperature
        self.init_std = init_std
        self.use_soft_reward = use_soft_reward
        self.exploration_noise = exploration_noise
        self.min_reward_std = min_reward_std
        self.gradient_accumulation = gradient_accumulation
        self.max_log_prob_tokens = max_log_prob_tokens  # 截断长度
        
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Get target device
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        self.target_device = next(target_layer.parameters()).device
        self.target_dtype = next(target_layer.parameters()).dtype
        
        # Initialize vector
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        
        if init_from_extracted is not None:
            self.vector_param.data = init_from_extracted.clone().float()
            print(f"  Initialized from extracted vector, norm={self.vector_param.norm().item():.4f}")
        else:
            nn.init.normal_(self.vector_param, std=init_std)
            print(f"  Initialized with std={init_std}, norm={self.vector_param.norm().item():.4f}")
        
        self.vector_param.data = self.vector_param.data.to(
            device=self.target_device,
            dtype=torch.float32
        )
    
    def _build_prompt(self, sample) -> str:
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        return prompt
    
    def _add_exploration_noise(self) -> torch.Tensor:
        if self.exploration_noise > 0:
            noise = torch.randn_like(self.vector_param) * self.exploration_noise
            return self.vector_param + noise
        return self.vector_param
    
    def _generate_samples(
        self,
        prompt: str,
        num_samples: int,
        add_noise: bool = True,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor, int]]:
        """生成样本，返回(text, input_ids, generated_ids, num_tokens)"""
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # 限制输入长度
        )
        input_ids = encoding["input_ids"].to(self.target_device)
        attention_mask = encoding["attention_mask"].to(self.target_device)
        input_len = input_ids.shape[1]
        
        results = []
        
        for _ in range(num_samples):
            self.model_wrapper.clear_hooks()
            
            if add_noise:
                inject_vector = self._add_exploration_noise()
            else:
                inject_vector = self.vector_param
            
            self.model_wrapper.register_injection_hook(
                self.layer_idx,
                inject_vector.detach(),  # detach避免在生成时建立计算图
                scaling_factor=1.0,
                requires_grad=False
            )
            
            with torch.no_grad():
                outputs = self.model_wrapper.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0, input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_tokens = len(generated_ids)
            
            results.append((generated_text, input_ids.clone(), generated_ids.clone(), num_tokens))
            
            self.model_wrapper.clear_hooks()
        
        return results
    
    def _compute_soft_reward(
        self,
        generated_text: str,
        ground_truth: str,
        num_tokens: int,
    ) -> float:
        """计算连续reward"""
        reward = 0.0
        
        predicted = extract_answer_from_text(generated_text, self.dataset_type)
        is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
        
        if is_correct:
            reward += 1.0
        elif predicted is not None:
            reward += 0.1
        
        # 简化的长度惩罚
        if num_tokens > 400:
            reward -= 0.1
        elif num_tokens < 50:
            reward -= 0.05
        
        if "\\boxed" in generated_text:
            reward += 0.05
        
        return reward
    
    def _compute_rewards(
        self,
        generated_samples: List[Tuple[str, torch.Tensor, torch.Tensor, int]],
        ground_truth: str,
    ) -> torch.Tensor:
        rewards = []
        for text, _, _, num_tokens in generated_samples:
            if self.use_soft_reward:
                reward = self._compute_soft_reward(text, ground_truth, num_tokens)
            else:
                predicted = extract_answer_from_text(text, self.dataset_type)
                is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
                reward = 1.0 if is_correct else 0.0
            rewards.append(reward)
        return torch.tensor(rewards, device=self.target_device, dtype=torch.float32)
    
    def _compute_advantages(self, rewards: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        if std_reward < self.min_reward_std:
            n = len(rewards)
            if rewards.max() == rewards.min():
                return torch.zeros_like(rewards), False
            ranks = torch.argsort(torch.argsort(rewards, descending=True)).float()
            advantages = 1.0 - 2.0 * ranks / (n - 1) if n > 1 else torch.zeros_like(rewards)
            return advantages, True
        
        advantages = (rewards - mean_reward) / (std_reward + self.epsilon)
        return advantages, True
    
    def _compute_log_prob_efficient(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        高效计算log_prob - 只使用最后N个token
        大幅减少显存占用
        """
        # 截断生成序列，只保留最后max_log_prob_tokens个token
        if len(generated_ids) > self.max_log_prob_tokens:
            generated_ids = generated_ids[-self.max_log_prob_tokens:]
        
        # 构建输入：只需要生成部分前面的一小段上下文
        context_len = min(64, input_ids.shape[1])  # 只保留64个token作为上下文
        context_ids = input_ids[0, -context_len:]
        
        full_ids = torch.cat([context_ids, generated_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_injection_hook(
            self.layer_idx,
            self.vector_param,
            scaling_factor=1.0,
            requires_grad=True
        )
        
        # Forward pass
        outputs = self.model_wrapper(full_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 只计算生成部分的log_prob
        gen_start = context_len
        shift_logits = logits[:, gen_start-1:-1, :].contiguous()
        shift_labels = full_ids[:, gen_start:].contiguous()
        
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 使用mean而非sum，更稳定
        total_log_prob = token_log_probs.mean()
        
        self.model_wrapper.clear_hooks()
        
        return total_log_prob
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """训练 - 显存优化版本"""
        print(f"Training Self-Evolved CoT Vector (Memory Optimized) at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Iterations: {self.num_iterations}")
        print(f"  Group size: {self.group_size}, Questions/iter: {self.questions_per_iter}")
        print(f"  LR: {self.learning_rate}, Temperature: {self.temperature}")
        print(f"  Init norm: {self.vector_param.norm().item():.4f}")
        print(f"  Max log_prob tokens: {self.max_log_prob_tokens}")
        
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_iterations,
            eta_min=self.learning_rate * 0.1
        )
        
        total_rewards = []
        best_avg_reward = -float('inf')
        best_vector = None
        
        pbar = tqdm(range(self.num_iterations), desc="GRPO Training", ncols=100)
        
        for iteration in pbar:
            iter_rewards = []
            iter_grad_norms = []
            skipped = 0
            
            sampled_questions = random.sample(
                support_samples,
                min(self.questions_per_iter, len(support_samples))
            )
            
            for sample in sampled_questions:
                prompt = self._build_prompt(sample)
                ground_truth = sample.answer
                
                # 生成样本
                try:
                    generated_samples = self._generate_samples(prompt, self.group_size, add_noise=True)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise
                
                rewards = self._compute_rewards(generated_samples, ground_truth)
                iter_rewards.extend(rewards.tolist())
                
                advantages, is_valid = self._compute_advantages(rewards)
                
                if not is_valid:
                    skipped += 1
                    continue
                
                # === 关键优化: 每个样本独立计算梯度并立即更新 ===
                for i, (text, input_ids, generated_ids, _) in enumerate(generated_samples):
                    if abs(advantages[i].item()) < 0.1:
                        continue
                    
                    try:
                        # 计算单个样本的loss
                        log_prob = self._compute_log_prob_efficient(input_ids, generated_ids)
                        loss = -advantages[i] * log_prob
                        
                        # 立即backward
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_([self.vector_param], max_norm=1.0)
                        iter_grad_norms.append(grad_norm.item())
                        
                        # 立即更新
                        optimizer.step()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        raise
                    finally:
                        # 清理
                        self.model_wrapper.clear_hooks()
                
                # 每个问题后清理显存
                torch.cuda.empty_cache()
            
            # 更新scheduler
            scheduler.step()
            
            # 统计
            avg_reward = sum(iter_rewards) / max(len(iter_rewards), 1)
            avg_grad = sum(iter_grad_norms) / max(len(iter_grad_norms), 1) if iter_grad_norms else 0
            current_norm = self.vector_param.norm().item()
            
            total_rewards.append(avg_reward)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_vector = self.vector_param.detach().clone()
            
            pbar.set_postfix({
                "R": f"{avg_reward:.2f}",
                "norm": f"{current_norm:.1f}",
                "grad": f"{avg_grad:.3f}",
                "skip": skipped,
            })
            
            if wandb_run and iteration % 10 == 0:
                wandb_run.log({
                    "iteration": iteration,
                    "train/avg_reward": avg_reward,
                    "train/best_reward": best_avg_reward,
                    "train/vector_norm": current_norm,
                    "train/grad_norm": avg_grad,
                })
            
            # 每10轮清理一次显存
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        self.vector = best_vector if best_vector is not None else self.vector_param.detach().clone()
        
        print(f"\nTraining complete!")
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Final vector norm: {self.vector.norm().item():.4f}")
        
        return self.vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        return self.vector


class SelfEvolvedCoTVectorV2(SelfEvolvedCoTVector):
    """
    V2版本 - 继承显存优化，添加自适应温度
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        group_size: int = 4,
        num_iterations: int = 100,
        learning_rate: float = 5e-3,
        beta: float = 0.01,
        epsilon: float = 1e-8,
        max_new_tokens: int = 512,
        questions_per_iter: int = 2,
        temperature: float = 0.7,
        init_std: float = 0.35,
        init_from_extracted: Optional[torch.Tensor] = None,
        use_soft_reward: bool = True,
        exploration_noise: float = 0.1,
        min_reward_std: float = 0.02,
        gradient_accumulation: int = 1,
        max_log_prob_tokens: int = 128,
        adaptive_temp: bool = True,
    ):
        super().__init__(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            dataset_type=dataset_type,
            group_size=group_size,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            beta=beta,
            epsilon=epsilon,
            max_new_tokens=max_new_tokens,
            questions_per_iter=questions_per_iter,
            temperature=temperature,
            init_std=init_std,
            init_from_extracted=init_from_extracted,
            use_soft_reward=use_soft_reward,
            exploration_noise=exploration_noise,
            min_reward_std=min_reward_std,
            gradient_accumulation=gradient_accumulation,
            max_log_prob_tokens=max_log_prob_tokens,
        )
        
        self.adaptive_temp = adaptive_temp
        self.base_temperature = temperature
        self.recent_reward_stds = []
    
    def _get_adaptive_temperature(self) -> float:
        if not self.adaptive_temp or len(self.recent_reward_stds) < 5:
            return self.temperature
        
        avg_std = sum(self.recent_reward_stds[-10:]) / len(self.recent_reward_stds[-10:])
        
        if avg_std < 0.1:
            return min(self.base_temperature * 1.2, 1.0)
        elif avg_std > 0.3:
            return max(self.base_temperature * 0.9, 0.5)
        
        return self.temperature
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """V2训练 - 显存优化 + 自适应温度"""
        print(f"Training Self-Evolved CoT Vector V2 (Optimized) at layer {self.layer_idx}...")
        print(f"  Total samples: {len(support_samples)}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Group size G: {self.group_size}")
        print(f"  Questions per iteration: {self.questions_per_iter}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Temperature: {self.temperature} (adaptive: {self.adaptive_temp})")
        print(f"  Init vector norm: {self.vector_param.norm().item():.4f}")
        print(f"  Soft reward: {self.use_soft_reward}")
        print(f"  Exploration noise: {self.exploration_noise}")
        print(f"  Max log_prob tokens: {self.max_log_prob_tokens}")
        print("-" * 50)
        
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        def lr_lambda(step):
            warmup_steps = self.num_iterations // 10
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, self.num_iterations - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        reward_history = []
        norm_history = [self.vector_param.norm().item()]
        correct_count = 0
        total_count = 0
        best_avg_reward = -float('inf')
        best_vector = None
        
        pbar = tqdm(range(self.num_iterations), desc="GRPO V2", ncols=120)
        
        for iteration in pbar:
            iter_rewards = []
            iter_grad_norms = []
            skipped = 0
            
            # 自适应温度
            current_temp = self._get_adaptive_temperature()
            self.temperature = current_temp
            
            batch_samples = random.sample(
                support_samples,
                min(self.questions_per_iter, len(support_samples))
            )
            
            for sample in batch_samples:
                prompt = self._build_prompt(sample)
                ground_truth = sample.answer
                
                try:
                    generated_samples = self._generate_samples(
                        prompt, 
                        self.group_size,
                        add_noise=True
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise
                
                rewards = self._compute_rewards(generated_samples, ground_truth)
                iter_rewards.extend(rewards.tolist())
                
                self.recent_reward_stds.append(rewards.std().item())
                if len(self.recent_reward_stds) > 50:
                    self.recent_reward_stds = self.recent_reward_stds[-50:]
                
                for r in rewards:
                    total_count += 1
                    if r > 0.8:
                        correct_count += 1
                
                advantages, is_valid = self._compute_advantages(rewards)
                
                if not is_valid:
                    skipped += 1
                    continue
                
                # 每个样本独立更新
                for i, (text, input_ids, generated_ids, _) in enumerate(generated_samples):
                    if abs(advantages[i].item()) < 0.05:
                        continue
                    
                    try:
                        log_prob = self._compute_log_prob_efficient(input_ids, generated_ids)
                        loss = -advantages[i] * log_prob
                        
                        optimizer.zero_grad()
                        loss.backward()
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_([self.vector_param], max_norm=1.0)
                        iter_grad_norms.append(grad_norm.item())
                        
                        optimizer.step()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        raise
                    finally:
                        self.model_wrapper.clear_hooks()
                
                torch.cuda.empty_cache()
            
            scheduler.step()
            
            # 统计
            avg_reward = sum(iter_rewards) / max(len(iter_rewards), 1)
            avg_grad = sum(iter_grad_norms) / max(len(iter_grad_norms), 1) if iter_grad_norms else 0
            current_norm = self.vector_param.norm().item()
            
            reward_history.append(avg_reward)
            norm_history.append(current_norm)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_vector = self.vector_param.detach().clone()
            
            accuracy = correct_count / max(total_count, 1) * 100
            pbar.set_postfix({
                "R": f"{avg_reward:.2f}",
                "norm": f"{current_norm:.1f}",
                "Acc": f"{accuracy:.0f}%",
                "T": f"{current_temp:.2f}",
            })
            
            if wandb_run and iteration % 5 == 0:
                wandb_run.log({
                    "iteration": iteration,
                    "train/reward": avg_reward,
                    "train/accuracy": accuracy,
                    "train/best_reward": best_avg_reward,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/vector_norm": current_norm,
                    "train/temperature": current_temp,
                    "train/skipped": skipped,
                    "train/grad_norm": avg_grad,
                })
            
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        self.vector = best_vector if best_vector is not None else self.vector_param.detach().clone()
        
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"  Initial vector norm: {norm_history[0]:.4f}")
        print(f"  Final vector norm: {self.vector.norm().item():.4f}")
        print(f"  Norm change: {self.vector.norm().item() - norm_history[0]:+.4f}")
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Final accuracy: {correct_count/max(total_count,1)*100:.1f}%")
        print("=" * 50)
        
        return self.vector
