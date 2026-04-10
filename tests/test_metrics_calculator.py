"""Unit tests untuk MetricsCalculator."""

import pytest
from src.finetuned.evaluation.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Test cases untuk MetricsCalculator class."""
    
    def test_init(self):
        """Test initialization."""
        calculator = MetricsCalculator()
        
        assert calculator.lang == "id"
        assert calculator._bleu is None
        assert calculator._rouge is None
        assert calculator._bertscore is None
    
    def test_init_custom_lang(self):
        """Test initialization dengan custom language."""
        calculator = MetricsCalculator(lang="en")
        
        assert calculator.lang == "en"
    
    def test_compute_bleu(self):
        """Test BLEU computation."""
        calculator = MetricsCalculator()
        
        predictions = ["the cat is on the mat", "there is a cat on the mat"]
        references = ["the cat is on the mat", "the cat is on the mat"]
        
        results = calculator.compute_bleu(predictions, references)
        
        assert "bleu" in results
        assert "bleu_1" in results
        assert "bleu_4" in results
        assert 0.0 <= results["bleu"] <= 1.0
    
    def test_compute_bleu_empty(self):
        """Test BLEU dengan empty inputs."""
        calculator = MetricsCalculator()
        
        results = calculator.compute_bleu([], [])
        
        assert results["bleu"] == 0.0
    
    def test_compute_rouge(self):
        """Test ROUGE computation."""
        calculator = MetricsCalculator()
        
        predictions = ["the cat is on the mat", "there is a cat on the mat"]
        references = ["the cat is on the mat", "the cat is on the mat"]
        
        results = calculator.compute_rouge(predictions, references)
        
        assert "rouge_1" in results
        assert "rouge_2" in results
        assert "rouge_l" in results
        assert 0.0 <= results["rouge_1"] <= 1.0
    
    def test_compute_rouge_empty(self):
        """Test ROUGE dengan empty inputs."""
        calculator = MetricsCalculator()
        
        results = calculator.compute_rouge([], [])
        
        assert results["rouge_1"] == 0.0
    
    def test_compute_diversity(self):
        """Test diversity computation."""
        calculator = MetricsCalculator()
        
        predictions = [
            "the cat is on the mat",
            "the dog is in the house",
            "a bird flies in the sky"
        ]
        
        results = calculator.compute_diversity(predictions)
        
        assert "distinct_1" in results
        assert "distinct_2" in results
        assert 0.0 <= results["distinct_1"] <= 1.0
        assert 0.0 <= results["distinct_2"] <= 1.0
    
    def test_compute_diversity_empty(self):
        """Test diversity dengan empty inputs."""
        calculator = MetricsCalculator()
        
        results = calculator.compute_diversity([])
        
        assert results["distinct_1"] == 0.0
        assert results["distinct_2"] == 0.0
    
    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        calculator = MetricsCalculator()
        
        predictions = ["the cat is on the mat", "there is a cat on the mat"]
        references = ["the cat is on the mat", "the cat is on the mat"]
        
        results = calculator.compute_all_metrics(
            predictions,
            references,
            include_bertscore=False  # Skip BERTScore untuk speed
        )
        
        assert "bleu" in results
        assert "rouge_1" in results
        assert "distinct_1" in results
    
    def test_print_metrics_report(self, capsys):
        """Test printing metrics report."""
        calculator = MetricsCalculator()
        
        metrics = {
            "bleu": 0.5,
            "bleu_1": 0.6,
            "bleu_4": 0.4,
            "rouge_1": 0.7,
            "rouge_l": 0.6,
            "distinct_1": 0.8
        }
        
        calculator.print_metrics_report(metrics, title="Test Report")
        
        captured = capsys.readouterr()
        assert "Test Report" in captured.out
        assert "BLEU" in captured.out