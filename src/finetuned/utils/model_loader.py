"""Helper functions untuk load model dengan LoRA configuration."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import torch
from typing import Tuple, Dict, Optional


def load_model_with_lora(
    model_name: str = 'Wikidepia/IndoT5-base',
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = None,
    device: str = 'cuda'
) -> Tuple:
    """
    Load model dengan LoRA configuration.
    
    Args:
        model_name: HuggingFace model name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout rate
        target_modules: List of modules to apply LoRA (default: ['q', 'v'])
        device: Device untuk load model
        
    Returns:
        (peft_model, tokenizer)
    """
    if target_modules is None:
        target_modules = ['q', 'v']
    
    print(f'Loading base model: {model_name}')
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('✓ Base model loaded')
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias='none',
        task_type='SEQ_2_SEQ_LM'
    )
    
    # Apply LoRA
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print parameter statistics
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f'✓ LoRA applied: r={lora_r}, alpha={lora_alpha}, target={target_modules}')
    print(f'  Trainable: {trainable:,} ({100*trainable/total:.2f}%)')
    print(f'  Total:     {total:,}')
    
    # Move to device and enable gradients
    if torch.cuda.is_available() and device == 'cuda':
        peft_model = peft_model.to(device)
        peft_model.enable_input_require_grads()
        print(f'✓ Model device: {next(peft_model.parameters()).device}')
        print(f'  GPU allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
    else:
        print('⚠ CUDA not available, using CPU')
    
    return peft_model, tokenizer


def print_model_info(model, tokenizer):
    """Print model and tokenizer information."""
    print('\n=== Model Information ===')
    print(f'Model type: {type(model).__name__}')
    print(f'Tokenizer: {type(tokenizer).__name__}')
    print(f'Vocab size: {tokenizer.vocab_size}')
    print(f'Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')
    print(f'EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\nParameters:')
    print(f'  Total: {total_params:,}')
    print(f'  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')
    print(f'  Frozen: {total_params - trainable_params:,}')
