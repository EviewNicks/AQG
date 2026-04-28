"""
Adapter Model Loader for IndoNanoT5
Handles loading base model and adding adapter layers for parameter-efficient fine-tuning.

IMPORTANT: This module uses the NEW 'adapters' library (not 'adapter-transformers').
- adapter-transformers is DEPRECATED and has compatibility issues
- adapters is the official successor with full backward compatibility
- Install: pip install adapters (NOT adapter-transformers)
"""

import torch
import adapters
from adapters import AutoAdapterModel, AdapterConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Tuple


def load_model_with_adapter(
    model_name: str = 'LazarusNLP/IndoNanoT5-base',
    adapter_name: str = 'mcq_generation',
    adapter_config: str = 'pfeiffer',
    reduction_factor: int = 12,
    non_linearity: str = 'relu',
    device: str = 'cuda'
) -> Tuple[AutoAdapterModel, AutoTokenizer]:
    """
    Load IndoNanoT5 model with adapter layers.
    
    Args:
        model_name: HuggingFace model identifier
        adapter_name: Name for the adapter
        adapter_config: Adapter architecture ('pfeiffer', 'houlsby', etc.)
        reduction_factor: Bottleneck dimension factor (768/reduction_factor)
        non_linearity: Activation function ('relu', 'gelu')
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model_with_adapter, tokenizer)
    
    Example:
        >>> model, tokenizer = load_model_with_adapter(
        ...     adapter_name='mcq_generation',
        ...     reduction_factor=12  # d=64 for 768-dim model
        ... )
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL WITH ADAPTER LAYERS")
    print(f"{'='*60}")
    
    # Load base model using NEW adapters library approach
    print(f'Loading base model: {model_name}')
    print('  Using NEW adapters library (not adapter-transformers)')
    
    # Method 1: Use AutoAdapterModel (recommended for T5)
    try:
        model = AutoAdapterModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('✓ Base model loaded with AutoAdapterModel')
    except Exception as e:
        # Method 2: Load with transformers, then initialize adapters
        print(f'⚠ AutoAdapterModel failed: {str(e)[:80]}...')
        print('  Trying alternative: Load with transformers + adapters.init()')
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        adapters.init(model)  # Initialize adapter support
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('✓ Base model loaded with transformers + adapters.init()')
    
    # Configure adapter (using new config names)
    # Note: 'pfeiffer' is now 'seq_bn' in new library (but old names still work)
    config = AdapterConfig.load(
        adapter_config,  # 'pfeiffer' or 'seq_bn' both work
        reduction_factor=reduction_factor,
        non_linearity=non_linearity
    )
    
    # Add adapter to model
    model.add_adapter(adapter_name, config=config)
    print(f'✓ Adapter added: {adapter_config} config, d={768//reduction_factor}')
    
    # Activate adapter for training
    model.train_adapter(adapter_name)
    print('✓ Adapter activated for training')
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        print(f'✓ Model moved to GPU')
        print(f'  GPU allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
    elif device == 'cuda' and not torch.cuda.is_available():
        print('⚠ CUDA not available, using CPU')
        device = 'cpu'
    
    return model, tokenizer


def print_adapter_info(model, tokenizer):
    """
    Print detailed information about model with adapter.
    
    Args:
        model: Model with adapter layers
        tokenizer: Tokenizer for the model
    """
    # Count parameters
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n{'='*60}")
    print("MODEL INFORMATION")
    print(f"{'='*60}")
    
    print(f"\nParameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"  Total:     {all_params:,}")
    print(f"  Frozen:    {all_params - trainable_params:,}")
    
    print(f"\nTokenizer:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token:  {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token:  {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    return trainable_params, all_params
