"""
Learnable CoT Vector implementation.
Implements the teacher-student framework from Section 3.2.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class CoTDataset(Dataset):
    """Dataset for CoT vector training."""
    
    def __init__(self, samples: List, tokenizer, dataset_type: str, max_length: int = 2048):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize
        teacher_enc = self.tokenizer(
            teacher_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding="max_length"
        )
        student_enc = self.tokenizer(
            student_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        # Get answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        answer_len = len(answer_ids)
        
        # Find actual sequence lengths (excluding padding)
        teacher_len = (teacher_enc["attention_mask"].squeeze() == 1).sum().item()
        student_len = (student_enc["attention_mask"].squeeze() == 1).sum().item()
        
        return {
            "teacher_ids": teacher_enc["input_ids"].squeeze(),
            "teacher_mask": teacher_enc["attention_mask"].squeeze(),
            "student_ids": student_enc["input_ids"].squeeze(),
            "student_mask": student_enc["attention_mask"].squeeze(),
            "teacher_len": teacher_len,
            "student_len": student_len,
            "answer_len": answer_len,
        }


class LearnableCoTVector(BaseCoTVectorMethod):
    """
    Learnable CoT Vector optimized via teacher-student framework.
    
    Loss = L_align + λ * L_CE
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        lambda_val: float = 0.5,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize learnable vector
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.vector_param, std=0.02)
    
    def _compute_alignment_loss(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence alignment loss."""
        # Normalize for stable KL computation
        teacher_norm = F.softmax(teacher_hidden.float(), dim=-1)
        student_norm = F.log_softmax(student_hidden.float(), dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(student_norm, teacher_norm, reduction='batchmean')
        
        return kl_loss
    
    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on answer tokens."""
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        # Flatten
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1)
        
        # Compute loss only on masked positions
        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        masked_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
        
        return masked_loss
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """Train the learnable CoT vector."""
        print(f"Training learnable vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  LR: {self.learning_rate}, λ: {self.lambda_val}")
        
        # Create dataset and dataloader
        dataset = CoTDataset(support_samples, self.tokenizer, self.dataset_type)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Get target device for vector
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        self.vector_param.data = self.vector_param.data.to(target_device)
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_ce = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                teacher_ids = batch["teacher_ids"].to(target_device)
                teacher_mask = batch["teacher_mask"].to(target_device)
                student_ids = batch["student_ids"].to(target_device)
                student_mask = batch["student_mask"].to(target_device)
                
                # Teacher forward (frozen)
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_extraction_hook(self.layer_idx)
                
                with torch.no_grad():
                    teacher_outputs = self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
                teacher_hidden = self.model_wrapper.get_activations(self.layer_idx)
                
                # Student forward with injection
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx, 
                    self.vector_param,
                    scaling_factor=1.0,
                    requires_grad=True
                )
                self.model_wrapper.register_extraction_hook(self.layer_idx)
                
                student_outputs = self.model_wrapper(student_ids, attention_mask=student_mask)
                student_hidden = self.model_wrapper.get_activations(self.layer_idx)
                student_logits = student_outputs.logits
                
                # Get answer positions
                bs = teacher_ids.size(0)
                align_losses = []
                ce_losses = []
                
                for i in range(bs):
                    t_len = batch["teacher_len"][i].item()
                    s_len = batch["student_len"][i].item()
                    a_len = batch["answer_len"][i].item()
                    
                    # Answer token positions
                    t_ans_pos = list(range(max(0, t_len - a_len), t_len))
                    s_ans_pos = list(range(max(0, s_len - a_len), s_len))
                    
                    if len(t_ans_pos) > 0 and len(s_ans_pos) > 0:
                        t_hidden = teacher_hidden[i, t_ans_pos, :].mean(dim=0)
                        s_hidden = student_hidden[i, s_ans_pos, :].mean(dim=0)
                        align_losses.append(self._compute_alignment_loss(
                            t_hidden.unsqueeze(0), s_hidden.unsqueeze(0)
                        ))
                        
                        # CE loss
                        ans_mask = torch.zeros_like(student_mask[i])
                        ans_mask[s_ans_pos] = 1
                        ce_loss = self._compute_ce_loss(
                            student_logits[i:i+1],
                            student_ids[i:i+1],
                            ans_mask.unsqueeze(0)
                        )
                        ce_losses.append(ce_loss)
                
                if not align_losses:
                    continue
                
                # Combine losses
                align_loss = torch.stack(align_losses).mean()
                ce_loss = torch.stack(ce_losses).mean() if ce_losses else torch.tensor(0.0, device=target_device)
                
                loss = align_loss + self.lambda_val * ce_loss
                loss = loss / self.gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_([self.vector_param], 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Track losses
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_align += align_loss.item()
                epoch_ce += ce_loss.item()
                num_batches += 1
                
                # Update progress
                pbar.set_postfix({
                    "loss": f"{epoch_loss/num_batches:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                self.model_wrapper.clear_hooks()
            
            # Epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_align = epoch_align / max(num_batches, 1)
            avg_ce = epoch_ce / max(num_batches, 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, align={avg_align:.4f}, ce={avg_ce:.4f}")
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/align_loss": avg_align,
                    "train/ce_loss": avg_ce,
                    "train/lr": scheduler.get_last_lr()[0],
                })
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.vector = self.vector_param.detach().clone()
        
        # Final vector
        if self.vector is None:
            self.vector = self.vector_param.detach().clone()
        
        print(f"Training complete. Vector norm: {self.vector.norm().item():.4f}")
        
        return self.vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        return self.vector
