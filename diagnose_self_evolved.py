#!/usr/bin/env python3
"""
Self-Evolved CoT Vector 诊断脚本
用于快速定位为什么学到的vector L2范数极小的问题

诊断项目：
1. 初始化检查 - vector初始范数是否合理
2. 采样检查 - 能否采样到正确答案
3. Reward分布 - 奖励信号是否有效
4. Advantage计算 - 优势函数是否有意义
5. 梯度检查 - 梯度是否消失/爆炸
6. 更新检查 - 每步vector变化量
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset, PROMPT_TEMPLATES
from src.utils import extract_answer_from_text, compare_answers, set_seed


class SelfEvolvedDiagnostic:
    """Self-Evolved方法诊断类"""
    
    def __init__(self, model_wrapper, tokenizer, layer_idx, dataset_type="gsm8k"):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.dataset_type = dataset_type
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # 获取目标设备
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        self.target_device = next(target_layer.parameters()).device
        self.target_dtype = next(target_layer.parameters()).dtype
        self.hidden_size = model_wrapper.hidden_size
        
        # 诊断结果存储
        self.diagnostics = {
            "init": {},
            "sampling": [],
            "rewards": [],
            "advantages": [],
            "gradients": [],
            "updates": [],
        }
    
    def _build_prompt(self, sample) -> str:
        """构建prompt"""
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        return prompt
    
    def diagnose_initialization(self, std_values=[0.02, 0.1, 0.5, 1.0]):
        """诊断1: 检查不同初始化标准差下的vector范数"""
        print("\n" + "="*60)
        print("诊断1: 初始化检查")
        print("="*60)
        
        results = []
        for std in std_values:
            vec = torch.zeros(self.hidden_size)
            nn.init.normal_(vec, std=std)
            norm = vec.norm().item()
            expected_norm = std * np.sqrt(self.hidden_size)
            
            result = {
                "std": std,
                "actual_norm": norm,
                "expected_norm": expected_norm,
                "hidden_size": self.hidden_size,
            }
            results.append(result)
            print(f"  std={std:.2f}: 实际范数={norm:.4f}, 期望范数≈{expected_norm:.4f}")
        
        self.diagnostics["init"]["std_sweep"] = results
        
        # 建议
        print("\n建议:")
        print(f"  - 当前hidden_size={self.hidden_size}")
        print(f"  - 若目标范数≈20, 建议std≈{20/np.sqrt(self.hidden_size):.4f}")
        print(f"  - 或者使用extracted方法得到的vector范数作为参考")
        
        return results
    
    def diagnose_sampling(self, samples, num_samples_per_question=4, temperature=0.7, max_new_tokens=512):
        """诊断2: 检查采样是否能得到正确答案"""
        print("\n" + "="*60)
        print("诊断2: 采样检查 (无vector注入)")
        print("="*60)
        
        total_correct = 0
        total_generated = 0
        sample_results = []
        
        for i, sample in enumerate(tqdm(samples, desc="采样测试")):
            prompt = self._build_prompt(sample)
            ground_truth = sample.answer
            
            # Tokenize
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            input_ids = encoding["input_ids"].to(self.target_device)
            attention_mask = encoding["attention_mask"].to(self.target_device)
            input_len = input_ids.shape[1]
            
            # 清除hooks
            self.model_wrapper.clear_hooks()
            
            question_results = {
                "question_idx": i,
                "question": sample.question[:100] + "...",
                "ground_truth": ground_truth,
                "generations": [],
                "num_correct": 0,
            }
            
            # 生成多个样本
            for j in range(num_samples_per_question):
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_ids = outputs[0, input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 提取答案
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
                
                question_results["generations"].append({
                    "text_preview": generated_text[-200:],  # 只保留末尾
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "num_tokens": len(generated_ids),
                })
                
                if is_correct:
                    question_results["num_correct"] += 1
                    total_correct += 1
                total_generated += 1
            
            sample_results.append(question_results)
            
            # 打印前几个样本的详细信息
            if i < 3:
                print(f"\n问题 {i+1}: {sample.question[:80]}...")
                print(f"  正确答案: {ground_truth}")
                print(f"  正确采样数: {question_results['num_correct']}/{num_samples_per_question}")
                for j, gen in enumerate(question_results["generations"]):
                    status = "✓" if gen["is_correct"] else "✗"
                    print(f"    样本{j+1} [{status}]: pred={gen['predicted']}, tokens={gen['num_tokens']}")
        
        # 统计
        accuracy = total_correct / total_generated * 100 if total_generated > 0 else 0
        questions_with_correct = sum(1 for r in sample_results if r["num_correct"] > 0)
        
        print(f"\n采样统计:")
        print(f"  总采样数: {total_generated}")
        print(f"  正确数: {total_correct}")
        print(f"  正确率: {accuracy:.2f}%")
        print(f"  有正确答案的问题数: {questions_with_correct}/{len(samples)}")
        
        self.diagnostics["sampling"] = {
            "total_generated": total_generated,
            "total_correct": total_correct,
            "accuracy": accuracy,
            "questions_with_correct": questions_with_correct,
            "details": sample_results,
        }
        
        # 诊断结论
        print("\n诊断结论:")
        if accuracy < 5:
            print("  ⚠️ 采样正确率极低 (<5%), 这是主要问题!")
            print("  建议: ")
            print("    1. 降低temperature (如0.3)")
            print("    2. 增加num_beams")
            print("    3. 使用更简单的数据集")
            print("    4. 检查answer extraction逻辑")
        elif accuracy < 20:
            print("  ⚠️ 采样正确率较低 (<20%), 信号稀疏")
            print("  建议: 增加group_size或questions_per_iter")
        else:
            print("  ✓ 采样正确率尚可, 问题可能在其他环节")
        
        return self.diagnostics["sampling"]
    
    def diagnose_reward_and_advantage(self, samples, group_size=4, temperature=0.7, max_new_tokens=512):
        """诊断3: 检查reward和advantage计算"""
        print("\n" + "="*60)
        print("诊断3: Reward和Advantage计算检查")
        print("="*60)
        
        epsilon = 1e-8
        reward_batches = []
        advantage_batches = []
        skipped_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="Reward/Advantage测试")):
            prompt = self._build_prompt(sample)
            ground_truth = sample.answer
            
            # Tokenize
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            input_ids = encoding["input_ids"].to(self.target_device)
            attention_mask = encoding["attention_mask"].to(self.target_device)
            input_len = input_ids.shape[1]
            
            self.model_wrapper.clear_hooks()
            
            # 生成样本并计算reward
            rewards = []
            for _ in range(group_size):
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
                rewards.append(1.0 if is_correct else 0.0)
            
            rewards_tensor = torch.tensor(rewards)
            reward_batches.append(rewards)
            
            # 计算advantage
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std()
            
            if std_r < epsilon:
                # 所有reward相同，跳过
                skipped_count += 1
                advantage_batches.append(None)
            else:
                advantages = (rewards_tensor - mean_r) / (std_r + epsilon)
                advantage_batches.append(advantages.tolist())
            
            if i < 3:
                print(f"\n问题 {i+1}:")
                print(f"  Rewards: {rewards}")
                print(f"  Mean: {mean_r:.4f}, Std: {std_r:.4f}")
                if std_r >= epsilon:
                    print(f"  Advantages: {advantages.tolist()}")
                else:
                    print(f"  Advantages: SKIPPED (std < epsilon)")
        
        # 统计
        all_rewards = [r for batch in reward_batches for r in batch]
        valid_advantages = [a for batch in advantage_batches if batch is not None for a in batch]
        
        print(f"\n统计:")
        print(f"  问题总数: {len(samples)}")
        print(f"  跳过的问题数 (std<epsilon): {skipped_count} ({skipped_count/len(samples)*100:.1f}%)")
        print(f"  Reward均值: {np.mean(all_rewards):.4f}")
        print(f"  Reward标准差: {np.std(all_rewards):.4f}")
        if valid_advantages:
            print(f"  Advantage均值: {np.mean(valid_advantages):.4f}")
            print(f"  Advantage标准差: {np.std(valid_advantages):.4f}")
            print(f"  Advantage范围: [{min(valid_advantages):.4f}, {max(valid_advantages):.4f}]")
        
        self.diagnostics["rewards"] = {
            "all_rewards": all_rewards,
            "mean": np.mean(all_rewards),
            "std": np.std(all_rewards),
        }
        self.diagnostics["advantages"] = {
            "valid_advantages": valid_advantages,
            "skipped_count": skipped_count,
            "skip_ratio": skipped_count / len(samples),
        }
        
        # 诊断结论
        print("\n诊断结论:")
        if skipped_count / len(samples) > 0.8:
            print("  ⚠️ 超过80%的问题被跳过，因为reward全相同")
            print("  原因: 采样正确率太低(全0)或太高(全1)")
            print("  建议: 参考诊断2的建议提高采样多样性")
        if np.mean(all_rewards) < 0.1:
            print("  ⚠️ Reward均值极低 (<0.1), 正向信号太弱")
        
        return self.diagnostics["rewards"], self.diagnostics["advantages"]
    
    def diagnose_gradient_flow(self, samples, std=0.02, learning_rate=1e-3, num_steps=5, group_size=4, temperature=0.7, max_new_tokens=256):
        """诊断4: 检查梯度流动"""
        print("\n" + "="*60)
        print("诊断4: 梯度流动检查")
        print("="*60)
        
        # 初始化vector
        vector_param = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(vector_param, std=std)
        vector_param.data = vector_param.data.to(device=self.target_device, dtype=torch.float32)
        
        print(f"初始vector范数: {vector_param.norm().item():.4f}")
        
        optimizer = torch.optim.AdamW([vector_param], lr=learning_rate)
        epsilon = 1e-8
        
        gradient_history = []
        norm_history = [vector_param.norm().item()]
        
        for step in range(num_steps):
            sample = random.choice(samples)
            prompt = self._build_prompt(sample)
            ground_truth = sample.answer
            
            # Tokenize
            encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encoding["input_ids"].to(self.target_device)
            attention_mask = encoding["attention_mask"].to(self.target_device)
            input_len = input_ids.shape[1]
            
            # 生成样本 (不注入vector, 只用于获取生成结果)
            self.model_wrapper.clear_hooks()
            
            generated_samples = []
            rewards = []
            
            for _ in range(group_size):
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_ids = outputs[0, input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_samples.append((input_ids.clone(), generated_ids.clone()))
                
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
                rewards.append(1.0 if is_correct else 0.0)
            
            rewards_tensor = torch.tensor(rewards, device=self.target_device, dtype=torch.float32)
            
            # 检查是否可以计算advantage
            if rewards_tensor.std() < epsilon:
                print(f"  Step {step+1}: 跳过 (rewards全相同: {rewards})")
                gradient_history.append({"step": step+1, "skipped": True, "rewards": rewards})
                continue
            
            # 计算advantage
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + epsilon)
            
            # 计算log_prob和梯度
            loss = torch.tensor(0.0, device=self.target_device, requires_grad=True)
            
            for i, (inp_ids, gen_ids) in enumerate(generated_samples):
                if abs(advantages[i].item()) < epsilon:
                    continue
                
                # 计算log_prob (需要注入vector)
                full_ids = torch.cat([inp_ids.squeeze(0), gen_ids], dim=0).unsqueeze(0)
                full_mask = torch.ones_like(full_ids)
                
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx,
                    vector_param,
                    scaling_factor=1.0,
                    requires_grad=True
                )
                
                outputs = self.model_wrapper(full_ids, attention_mask=full_mask)
                logits = outputs.logits
                
                # 计算生成部分的log_prob
                gen_len = gen_ids.shape[0]
                shift_logits = logits[:, inp_ids.shape[1]-1:-1, :].contiguous()
                shift_labels = full_ids[:, inp_ids.shape[1]:].contiguous()
                
                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                total_log_prob = token_log_probs.sum()
                
                sample_loss = -advantages[i] * total_log_prob
                loss = loss + sample_loss
            
            loss = loss / group_size
            
            # Backward
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                
                grad_norm = vector_param.grad.norm().item() if vector_param.grad is not None else 0.0
                
                # Clip gradient
                torch.nn.utils.clip_grad_norm_([vector_param], max_norm=1.0)
                grad_norm_clipped = vector_param.grad.norm().item() if vector_param.grad is not None else 0.0
                
                optimizer.step()
                
                new_norm = vector_param.norm().item()
                delta_norm = new_norm - norm_history[-1]
                
                gradient_history.append({
                    "step": step+1,
                    "skipped": False,
                    "rewards": rewards,
                    "advantages": advantages.tolist(),
                    "loss": loss.item(),
                    "grad_norm_before_clip": grad_norm,
                    "grad_norm_after_clip": grad_norm_clipped,
                    "vector_norm": new_norm,
                    "delta_norm": delta_norm,
                })
                norm_history.append(new_norm)
                
                print(f"  Step {step+1}: rewards={rewards}, loss={loss.item():.4f}, "
                      f"grad_norm={grad_norm:.6f}, vec_norm={new_norm:.4f}, Δ={delta_norm:+.6f}")
            else:
                print(f"  Step {step+1}: loss不需要梯度")
                gradient_history.append({"step": step+1, "skipped": True, "reason": "no grad"})
            
            self.model_wrapper.clear_hooks()
        
        # 统计
        valid_steps = [g for g in gradient_history if not g.get("skipped", False)]
        
        print(f"\n统计:")
        print(f"  总步数: {num_steps}")
        print(f"  有效步数: {len(valid_steps)}")
        print(f"  初始范数: {norm_history[0]:.4f}")
        print(f"  最终范数: {norm_history[-1]:.4f}")
        print(f"  范数变化: {norm_history[-1] - norm_history[0]:+.6f}")
        
        if valid_steps:
            avg_grad = np.mean([g["grad_norm_before_clip"] for g in valid_steps])
            avg_delta = np.mean([g["delta_norm"] for g in valid_steps])
            print(f"  平均梯度范数: {avg_grad:.6f}")
            print(f"  平均每步范数变化: {avg_delta:.6f}")
        
        self.diagnostics["gradients"] = {
            "history": gradient_history,
            "norm_history": norm_history,
        }
        
        # 诊断结论
        print("\n诊断结论:")
        if len(valid_steps) == 0:
            print("  ⚠️ 没有有效的梯度更新步!")
            print("  原因: 所有步都被跳过(reward全相同)")
        elif valid_steps:
            avg_grad = np.mean([g["grad_norm_before_clip"] for g in valid_steps])
            if avg_grad < 1e-6:
                print("  ⚠️ 梯度极小 (<1e-6), 梯度消失!")
                print("  可能原因: log_prob计算问题或advantage太小")
            elif avg_grad > 100:
                print("  ⚠️ 梯度较大 (>100), 需要更强的clip")
            else:
                print(f"  ✓ 梯度范数正常: {avg_grad:.4f}")
            
            total_change = norm_history[-1] - norm_history[0]
            if abs(total_change) < 0.01:
                print(f"  ⚠️ Vector几乎没有变化 (Δ={total_change:.6f})")
                print("  可能原因: 学习率太小或有效更新太少")
        
        return self.diagnostics["gradients"]
    
    def run_full_diagnosis(self, samples, args):
        """运行完整诊断"""
        print("\n" + "="*60)
        print("Self-Evolved CoT Vector 完整诊断")
        print("="*60)
        print(f"模型: {self.model_wrapper.model_path}")
        print(f"层: {self.layer_idx}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"样本数: {len(samples)}")
        
        # 1. 初始化检查
        self.diagnose_initialization()
        
        # 2. 采样检查
        sampling_samples = samples[:args.num_sampling_samples]
        self.diagnose_sampling(
            sampling_samples, 
            num_samples_per_question=args.group_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        # 3. Reward/Advantage检查
        reward_samples = samples[:args.num_reward_samples]
        self.diagnose_reward_and_advantage(
            reward_samples,
            group_size=args.group_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        # 4. 梯度流检查
        self.diagnose_gradient_flow(
            samples,
            std=args.init_std,
            learning_rate=args.learning_rate,
            num_steps=args.num_grad_steps,
            group_size=args.group_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens // 2  # 减少tokens加速
        )
        
        # 最终总结
        print("\n" + "="*60)
        print("诊断总结")
        print("="*60)
        
        issues = []
        
        # 检查初始化
        if args.init_std == 0.02:
            expected_norm = 0.02 * np.sqrt(self.hidden_size)
            if expected_norm < 5:
                issues.append(f"初始化std=0.02太小, 初始范数仅≈{expected_norm:.2f}")
        
        # 检查采样
        if "sampling" in self.diagnostics and self.diagnostics["sampling"]:
            acc = self.diagnostics["sampling"].get("accuracy", 0)
            if acc < 5:
                issues.append(f"采样正确率极低 ({acc:.1f}%), 无法产生有效的reward信号")
        
        # 检查advantage
        if "advantages" in self.diagnostics and self.diagnostics["advantages"]:
            skip_ratio = self.diagnostics["advantages"].get("skip_ratio", 0)
            if skip_ratio > 0.7:
                issues.append(f"大部分问题被跳过 ({skip_ratio*100:.0f}%), advantage信号稀疏")
        
        # 检查梯度
        if "gradients" in self.diagnostics and self.diagnostics["gradients"]:
            history = self.diagnostics["gradients"].get("history", [])
            valid_steps = [g for g in history if not g.get("skipped", False)]
            if len(valid_steps) == 0:
                issues.append("没有有效的梯度更新步")
            elif valid_steps:
                avg_grad = np.mean([g["grad_norm_before_clip"] for g in valid_steps])
                if avg_grad < 1e-5:
                    issues.append(f"梯度极小 ({avg_grad:.2e})")
        
        if issues:
            print("发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n建议的修复方案:")
            print("  1. 增大初始化std (如0.1或0.5), 或用extracted vector初始化")
            print("  2. 降低temperature (如0.3-0.5)以提高正确率")
            print("  3. 增加group_size (如8-16)以增加reward多样性")
            print("  4. 增加questions_per_iter")
            print("  5. 使用更简单的数据集进行测试")
            print("  6. 检查answer extraction逻辑是否正确")
        else:
            print("  ✓ 未发现明显问题")
        
        return self.diagnostics


def main():
    parser = argparse.ArgumentParser(description="Self-Evolved CoT Vector诊断")
    
    # 模型和数据
    parser.add_argument("--model_path", type=str, default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--data_path", type=str, default="/home/haichao/TA/CoTVRL/data")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=20, help="总样本数")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # 诊断参数
    parser.add_argument("--num_sampling_samples", type=int, default=5, help="采样诊断的问题数")
    parser.add_argument("--num_reward_samples", type=int, default=5, help="reward诊断的问题数")
    parser.add_argument("--num_grad_steps", type=int, default=5, help="梯度诊断的步数")
    
    # Self-evolved参数
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 加载模型
    print("加载模型...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"模型加载完成: {model_wrapper.num_layers}层, hidden_size={model_wrapper.hidden_size}")
    
    # 加载数据
    print(f"加载{args.dataset}数据...")
    samples = load_dataset(args.data_path, args.dataset, "train", args.num_samples)
    print(f"加载了{len(samples)}个样本")
    
    # 运行诊断
    diagnostic = SelfEvolvedDiagnostic(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        dataset_type=args.dataset,
    )
    
    diagnostic.run_full_diagnosis(samples, args)
    
    print("\n诊断完成!")


if __name__ == "__main__":
    main()
