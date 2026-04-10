"""Unit tests untuk ModelEvaluator."""

import pytest
import torch
from pathlib import Path
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock


class TestModelEvaluator:
    """Test cases untuk ModelEvaluator class."""
    
    def test_init(self):
        """Test initialization."""
        from src.finetuned.evaluation.model_evaluator import ModelEvaluator
        from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
        
        # Mock model and tokenizer
        model = Mock()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        
        tokenizer = Mock()
        metrics = MetricsCalculator()
        
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            metrics_calculator=metrics
        )
        
        assert evaluator.model == model
        assert evaluator.tokenizer == tokenizer
        assert evaluator.max_length == 512
    
    def test_generate_prediction(self):
        """Test single prediction generation."""
        from src.finetuned.evaluation.model_evaluator import ModelEvaluator
        from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
        
        # Mock model
        model = Mock()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        model.__call__ = Mock(return_value=Mock(logits=torch.randn(1, 10, 1000)))
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        tokenizer.decode = Mock(return_value="generated text")
        
        metrics = MetricsCalculator()
        
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            metrics_calculator=metrics
        )
        
        prediction = evaluator.generate_prediction("test input")
        
        assert isinstance(prediction, str)
        assert prediction == "generated text"
    
    def test_compare_with_baseline(self):
        """Test baseline comparison."""
        from src.finetuned.evaluation.model_evaluator import ModelEvaluator
        from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
        
        model = Mock()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        
        tokenizer = Mock()
        metrics = MetricsCalculator()
        
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            metrics_calculator=metrics
        )
        
        finetuned_metrics = {
            "bleu": 0.5,
            "rouge_1": 0.6
        }
        
        baseline_metrics = {
            "bleu": 0.3,
            "rouge_1": 0.4
        }
        
        comparison = evaluator.compare_with_baseline(
            finetuned_metrics,
            baseline_metrics
        )
        
        assert "bleu_improvement_pct" in comparison
        assert "rouge_1_improvement_pct" in comparison
        assert comparison["bleu_improvement_pct"] > 0  # Should show improvement
    
    def test_compare_with_baseline_no_improvement(self):
        """Test baseline comparison dengan no improvement."""
        from src.finetuned.evaluation.model_evaluator import ModelEvaluator
        from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
        
        model = Mock()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        
        tokenizer = Mock()
        metrics = MetricsCalculator()
        
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            metrics_calculator=metrics
        )
        
        finetuned_metrics = {
            "bleu": 0.3,
            "rouge_1": 0.4
        }
        
        baseline_metrics = {
            "bleu": 0.5,
            "rouge_1": 0.6
        }
        
        comparison = evaluator.compare_with_baseline(
            finetuned_metrics,
            baseline_metrics
        )
        
        assert comparison["bleu_improvement_pct"] < 0  # Should show degradation