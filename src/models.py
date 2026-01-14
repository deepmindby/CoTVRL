"""
Model wrapper with hook mechanisms for CoT Vector injection and extraction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Callable


class CoTModelWrapper(nn.Module):
    """
    Wrapper around HuggingFace models that provides:
    1. Forward hooks for extracting activations
    2. Injection hooks for adding CoT vectors
    """
    
    def __init__(self, model_path: str, model_name: str = "qwen"):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        
        # Load model with multi-GPU support
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        
        # Hook management
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, torch.Tensor] = {}
        self._injection_vector_cached: Optional[torch.Tensor] = None
        
    def _get_num_layers(self) -> int:
        if self.model_name == "qwen":
            return len(self.model.model.layers)
        elif self.model_name == "llama":
            return len(self.model.model.layers)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _get_hidden_size(self) -> int:
        return self.model.config.hidden_size
    
    def _get_layer(self, layer_idx: int) -> nn.Module:
        if self.model_name in ["qwen", "llama"]:
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def register_extraction_hook(self, layer_idx: int, position_ids: Optional[torch.Tensor] = None):
        """Register hook to extract activations at specified layer."""
        layer = self._get_layer(layer_idx)
        
        def hook_fn(module, input, output):
            # output is tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store activation (detached, float32 for numerical stability)
            if position_ids is not None:
                # Extract only at specified positions
                extracted = hidden_states[:, position_ids, :].detach().float()
            else:
                extracted = hidden_states.detach().float()
            
            self._activations[layer_idx] = extracted
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def register_injection_hook(
        self, 
        layer_idx: int, 
        vector: torch.Tensor, 
        scaling_factor: float = 1.0,
        requires_grad: bool = False
    ):
        """
        Register hook to inject CoT vector at specified layer.
        Pre-caches the vector in correct dtype/device for efficiency.
        """
        layer = self._get_layer(layer_idx)
        
        # Pre-convert vector to target layer's device and dtype
        target_device = next(layer.parameters()).device
        target_dtype = next(layer.parameters()).dtype
        
        if requires_grad:
            # For training: keep as parameter, will be converted in hook
            vector_scaled = scaling_factor * vector
        else:
            # For inference: pre-convert and cache
            vector_scaled = scaling_factor * vector.to(device=target_device, dtype=target_dtype)
            if vector_scaled.dim() == 1:
                vector_scaled = vector_scaled.unsqueeze(0).unsqueeze(0)
            elif vector_scaled.dim() == 2:
                vector_scaled = vector_scaled.unsqueeze(0)
        
        # Store cached vector
        object.__setattr__(self, '_injection_vector_cached', vector_scaled)
        object.__setattr__(self, '_injection_requires_grad', requires_grad)
        object.__setattr__(self, '_injection_target_device', target_device)
        object.__setattr__(self, '_injection_target_dtype', target_dtype)
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            cached = self._injection_vector_cached
            
            if self._injection_requires_grad:
                # Training: convert on-the-fly to maintain gradient
                vec = cached.to(device=hidden_states.device, dtype=hidden_states.dtype)
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0).unsqueeze(0)
                elif vec.dim() == 2:
                    vec = vec.unsqueeze(0)
                modified = hidden_states + vec.expand_as(hidden_states)
            else:
                # Inference: use pre-cached vector
                modified = hidden_states + cached.expand_as(hidden_states)
            
            if rest is not None:
                return (modified,) + rest
            return modified
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get extracted activations for a layer."""
        return self._activations.get(layer_idx)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._activations.clear()
        object.__setattr__(self, '_injection_vector_cached', None)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate text."""
        return self.model.generate(input_ids=input_ids, **kwargs)
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property 
    def dtype(self):
        return next(self.model.parameters()).dtype


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
