"""Unit tests untuk ModelSetup."""

import pytest
import torch
from src.finetuned.model.model_setup import ModelSetup
from peft import LoraConfig


class TestModelSetup:
    """Test cases untuk ModelSetup class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_setup = ModelSetup()
    
    @pytest.mark.slow
    def test_load_base_model(self):
        """Test loading base model."""
        model = self.model_setup.load_base_model("LazarusNLP/IndoNanoT5-base")
        
        assert model is not None
        assert hasattr(model, "config")
    
    @pytest.mark.slow
    def test_load_tokenizer(self):
        """Test loading tokenizer."""
        tokenizer = self.model_setup.load_tokenizer("LazarusNLP/IndoNanoT5-base")
        
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")    
    @pytest.mark.slow
    def test_apply_lora(self):
        """Test applying LoRA adapters."""
        model = self.model_setup.load_base_model("LazarusNLP/IndoNanoT5-base")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q", "v"],
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        
        peft_model = self.model_setup.apply_lora(model, lora_config)
        
        assert peft_model is not None
        assert hasattr(peft_model, "print_trainable_parameters")
    
    @pytest.mark.slow
    def test_print_trainable_parameters(self):
        """Test parameter counting."""
        model = self.model_setup.load_base_model("LazarusNLP/IndoNanoT5-base")
        peft_model = self.model_setup.apply_lora(model)
        
        stats = self.model_setup.print_trainable_parameters(peft_model)
        
        assert "trainable_params" in stats
        assert "total_params" in stats
        assert "trainable_pct" in stats
        assert stats["trainable_pct"] <= 1.0  # Should be <= 1%
    
    def test_check_gpu_memory(self):
        """Test GPU memory check."""
        memory_info = self.model_setup.check_gpu_memory()
        
        assert "total_memory_gb" in memory_info
        assert "allocated_memory_gb" in memory_info
        assert "free_memory_gb" in memory_info
        
        if torch.cuda.is_available():
            assert memory_info["total_memory_gb"] > 0
