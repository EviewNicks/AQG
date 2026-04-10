"""Unit tests untuk TaskSpecificTrainer."""

import pytest
import torch
from pathlib import Path
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestTaskSpecificTrainer:
    """Test cases untuk TaskSpecificTrainer class."""
    
    def test_init(self):
        """Test initialization."""
        from src.finetuned.training.task_trainer import TaskSpecificTrainer
        
        # Mock model and tokenizer
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = TaskSpecificTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
        assert trainer.max_length == 512
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        from src.finetuned.training.task_trainer import TaskSpecificTrainer
        
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.batch_decode = Mock(return_value=["prediction 1", "prediction 2"])
        
        trainer = TaskSpecificTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        # Mock eval_preds
        eval_preds = (
            np.array([[1, 2, 3], [4, 5, 6]]),  # predictions
            np.array([[1, 2, 3], [4, 5, 6]])   # labels
        )
        
        metrics = trainer.compute_metrics(eval_preds)
        
        assert "bleu_4" in metrics
        assert "rouge_l" in metrics
    
    def test_get_training_args(self):
        """Test training arguments generation."""
        from src.finetuned.training.task_trainer import TaskSpecificTrainer
        from transformers import Seq2SeqTrainingArguments
        
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = TaskSpecificTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        args = trainer.get_training_args()
        
        assert isinstance(args, Seq2SeqTrainingArguments)
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 8
        assert args.learning_rate == 1e-4
        assert args.metric_for_best_model == "eval_bleu_4"
    
    def test_get_training_args_custom(self):
        """Test custom training arguments."""
        from src.finetuned.training.task_trainer import TaskSpecificTrainer
        
        model = Mock()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        trainer = TaskSpecificTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./test_checkpoints"
        )
        
        args = trainer.get_training_args(
            num_train_epochs=5,
            learning_rate=5e-5,
            per_device_train_batch_size=4
        )
        
        assert args.num_train_epochs == 5
        assert args.learning_rate == 5e-5
        assert args.per_device_train_batch_size == 4