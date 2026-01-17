"""
Argument parser for CoT Vectors reproduction.
All hyperparameters follow the paper's Appendix A.2.
Extended with Self-Evolved CoT Vector (GRPO) configuration.

优化版本 - 更新了self-evolved方法的默认参数
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs"
    )
    
    # ==================== General Configuration ====================
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
        help="Model type for architecture-specific handling"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/haichao/TA/CoTVRL/data",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # ==================== Method Selection ====================
    parser.add_argument(
        "--method",
        type=str,
        default="extracted",
        choices=["extracted", "learnable", "self_evolved"],
        help="CoT Vector acquisition method"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["extract", "train", "eval", "both"],
        help="Operation mode"
    )
    
    # ==================== Dataset Configuration ====================
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_support_samples",
        type=int,
        default=3000,
        help="Number of support samples (None=use all)"
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=100,
        help="Number of test samples (None=use all)"
    )
    
    # ==================== CoT Vector Configuration ====================
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Layer index to inject/extract CoT Vector"
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=1.0,
        help="Scaling factor μ for extracted vectors (Eq. 7)"
    )
    
    # ==================== Learnable Vector Configuration ====================
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=0.5,
        help="Balance factor λ between alignment and CE loss (Eq. 6)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate for vector optimization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (reduced for memory efficiency)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for LR scheduler"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for learnable method"
    )
    
    # ==================== Self-Evolved Vector Configuration (GRPO) ====================
    # 显存优化后的默认参数
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,  # 减小以节省显存
        help="Number of samples generated per question (Group size G in GRPO)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations for self-evolved method"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="KL penalty coefficient for GRPO (reserved for stability)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small value for numerical stability in advantage normalization"
    )
    parser.add_argument(
        "--grpo_lr",
        type=float,
        default=5e-3,
        help="Learning rate specifically for GRPO training"
    )
    parser.add_argument(
        "--questions_per_iter",
        type=int,
        default=2,  # 减小以节省显存
        help="Number of questions to sample per iteration (batch size for GRPO)"
    )
    parser.add_argument(
        "--grpo_max_new_tokens",
        type=int,
        default=256,  # 减小以节省显存
        help="Max new tokens for GRPO generation (smaller than eval)"
    )
    
    # === 新增的Self-Evolved参数 ===
    parser.add_argument(
        "--init_std",
        type=float,
        default=0.35,
        help="Initialization std for self-evolved vector (0.35 -> norm≈21 for hidden_size=3584)"
    )
    parser.add_argument(
        "--use_soft_reward",
        action="store_true",
        default=True,
        help="Use continuous soft reward instead of binary reward"
    )
    parser.add_argument(
        "--no_soft_reward",
        action="store_true",
        default=False,
        help="Disable soft reward, use binary reward"
    )
    parser.add_argument(
        "--exploration_noise",
        type=float,
        default=0.1,  # 减小: 0.15 -> 0.1
        help="Exploration noise std added to vector during generation"
    )
    parser.add_argument(
        "--grpo_gradient_accumulation",
        type=int,
        default=1,  # 减小以节省显存: 2 -> 1
        help="Gradient accumulation steps for GRPO"
    )
    parser.add_argument(
        "--init_from_extracted",
        type=str,
        default=None,
        help="Path to extracted vector to initialize self-evolved training"
    )
    parser.add_argument(
        "--max_log_prob_tokens",
        type=int,
        default=128,
        help="Max tokens used for log_prob calculation (memory optimization)"
    )
    
    # ==================== Generation Configuration ====================
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="Number of beams (1=greedy, faster)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Use sampling"
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        default=False,
        help="Stop when answer pattern detected"
    )
    
    # ==================== Logging Configuration ====================
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cot-vectors-reproduction",
        help="WandB project name"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        default=False,
        help="Skip baseline evaluation (use cached result if available)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps"
    )
    
    # ==================== Vector I/O ====================
    parser.add_argument(
        "--vector_path",
        type=str,
        default=None,
        help="Path to load pre-computed vector"
    )
    parser.add_argument(
        "--save_vector",
        action="store_true",
        default=True,
        help="Save extracted/learned vector"
    )
    
    return parser.parse_args()
