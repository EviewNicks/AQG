"""Unit tests untuk DomainAdaptationTrainer."""

import pytest
import torch
from pathlib import Path
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock


class TestDomainAdaptationTrainer:
    """Test cases untuk DomainAdaptationTrainer class."""
    
    def test_init(self):
        """Test initialization."""
        from src.finetuned.training.domain_trainer import DomainAdaptationTrainer
        
        # Mock model and tokenizer
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = DomainAdaptationTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
        assert trainer.max_length == 512
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing."""
        from src.finetuned.training.domain_trainer import DomainAdaptationTrainer
        
        # Mock model and tokenizer
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        tokenizer.as_target_tokenizer = MagicMock()
        tokenizer.__enter__ = Mock(return_value=tokenizer)
        tokenizer.__exit__ = Mock(return_value=None)
        
        trainer = DomainAdaptationTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        # Create dummy dataset
        data = {
            "input": ["test input 1", "test input 2"],
            "target": ["test output 1", "test output 2"]
        }
        dataset = Dataset.from_dict(data)
        
        # Mock the map function
        with patch.object(dataset, 'map') as mock_map:
            mock_map.return_value = dataset
            processed = trainer.preprocess_dataset(dataset)
            
            assert mock_map.called
    
    def test_get_training_args(self):
        """Test training arguments generation."""
        from src.finetuned.training.domain_trainer import DomainAdaptationTrainer
        from transformers import Seq2SeqTrainingArguments
        
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = DomainAdaptationTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        args = trainer.get_training_args()
        
        assert isinstance(args, Seq2SeqTrainingArguments)
        assert args.num_train_epochs == 6
        assert args.per_device_train_batch_size == 8
        assert args.learning_rate == 2e-4
    
    def test_get_training_args_custom(self):
        """Test custom training arguments."""
        from src.finetuned.training.domain_trainer import DomainAdaptationTrainer
        
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = DomainAdaptationTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        args = trainer.get_training_args(
            num_train_epochs=10,
            learning_rate=1e-4,
            per_device_train_batch_size=4
        )
        
        assert args.num_train_epochs == 10
        assert args.learning_rate == 1e-4
        assert args.per_device_train_batch_size == 4