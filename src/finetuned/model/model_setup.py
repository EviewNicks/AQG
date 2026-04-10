"""Model setup untuk IndoNanoT5 dengan LoRA adapters."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Any


class ModelSetup:
    """Class untuk setup IndoNanoT5 model dengan LoRA."""
    
    def __init__(self):
        """Initialize ModelSetup."""
        pass
    
    def load_base_model(
        self, 
        model_name: str = "LazarusNLP/IndoNanoT5-base"
    ) -> AutoModelForSeq2SeqLM:
        """
        Load pre-trained IndoNanoT5 base model.
        
        Uses AutoModelForSeq2SeqLM per official IndoNanoT5 documentation.
        
        Args:
            model_name: Model name dari HuggingFace Hub
            
        Returns:
            AutoModelForSeq2SeqLM model (~248M parameters)
        """
        print(f"Loading base model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✓ Model loaded successfully")
        
        return model
    
    def load_tokenizer(
        self,
        model_name: str = "LazarusNLP/IndoNanoT5-base"
    ) -> AutoTokenizer:
        """
        Load tokenizer untuk IndoNanoT5.
        
        Uses AutoTokenizer per official IndoNanoT5 documentation.
        
        Args:
            model_name: Model name dari HuggingFace Hub
            
        Returns:
            AutoTokenizer instance
        """
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded successfully")
        
        return tokenizer
    
    def apply_lora(
        self, 
        model: T5ForConditionalGeneration,
        lora_config: LoraConfig = None
    ) -> PeftModel:
        """
        Apply LoRA adapters ke model.
        
        Args:
            model: Base T5 model
            lora_config: LoraConfig (optional, uses default if None)
                - r (rank): 8
                - lora_alpha: 16
                - lora_dropout: 0.1
                - target_modules: ["q", "v"]
                
        Returns:
            PeftModel dengan ~1.24M trainable parameters (~0.5%)
        """
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q", "v"],
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        
        print("Applying LoRA adapters...")
        print(f"  Rank: {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Dropout: {lora_config.lora_dropout}")
        print(f"  Target modules: {lora_config.target_modules}")
        
        peft_model = get_peft_model(model, lora_config)
        print("✓ LoRA adapters applied successfully")
        
        return peft_model
    
    def print_trainable_parameters(self, model: PeftModel) -> Dict[str, Any]:
        """
        Print summary trainable vs total parameters.
        
        Args:
            model: PeftModel dengan LoRA adapters
            
        Returns:
            Dict dengan parameter statistics
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        print("\n=== Model Parameter Summary ===")
        print(f"Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        print(f"Total params:     {total_params:,}")
        print(f"Non-trainable:    {total_params - trainable_params:,}")
        
        # Verify parameter efficiency
        if trainable_pct > 1.0:
            print(f"⚠ Warning: Trainable parameters ({trainable_pct:.2f}%) exceed 1%")
        else:
            print(f"✓ Parameter efficiency verified: {trainable_pct:.2f}% trainable")
        
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": trainable_pct
        }
    
    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Check GPU memory availability.
        
        Returns:
            Dict dengan:
            - total_memory_gb: float
            - allocated_memory_gb: float
            - free_memory_gb: float
        """
        if not torch.cuda.is_available():
            print("⚠ GPU not available")
            return {
                "total_memory_gb": 0.0,
                "allocated_memory_gb": 0.0,
                "free_memory_gb": 0.0
            }
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        free_memory = total_memory - allocated_memory
        
        print("\n=== GPU Memory Status ===")
        print(f"Total Memory:     {total_memory:.2f} GB")
        print(f"Allocated Memory: {allocated_memory:.2f} GB")
        print(f"Free Memory:      {free_memory:.2f} GB")
        
        if free_memory < 2.0:
            print("⚠ Warning: Low GPU memory available")
        else:
            print("✓ Sufficient GPU memory available")
        
        return {
            "total_memory_gb": total_memory,
            "allocated_memory_gb": allocated_memory,
            "free_memory_gb": free_memory
        }
    
    def setup_model_for_training(
        self,
        model_name: str = "LazarusNLP/IndoNanoT5-base",
        lora_config: LoraConfig = None
    ) -> tuple:
        """
        Complete setup: load model, tokenizer, dan apply LoRA.
        
        Args:
            model_name: Model name dari HuggingFace Hub
            lora_config: LoraConfig (optional)
            
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        print("\n" + "=" * 60)
        print("SETTING UP MODEL FOR TRAINING")
        print("=" * 60 + "\n")
        
        # Load base model and tokenizer
        base_model = self.load_base_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        
        # Apply LoRA
        peft_model = self.apply_lora(base_model, lora_config)
        
        # Print parameter summary
        self.print_trainable_parameters(peft_model)
        
        # Check GPU memory
        self.check_gpu_memory()
        
        print("\n✓ Model setup complete!")
        
        return peft_model, tokenizer
