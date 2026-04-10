"""Unit tests untuk DatasetLoader."""

import pytest
from pathlib import Path
from src.finetuned.data.dataset_loader import DatasetLoader


class TestDatasetLoader:
    """Test cases untuk DatasetLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DatasetLoader()
    
    def test_load_dataset_success(self):
        """Test loading valid dataset."""
        # Assuming dataset exists
        dataset_dir = "dataset_aqg/output_domain/"
        
        if Path(dataset_dir).exists():
            dataset = self.loader.load_dataset(dataset_dir, split="train")
            assert dataset is not None
            assert len(dataset) > 0
    
    def test_load_dataset_file_not_found(self):
        """Test loading non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_dataset("nonexistent_dir/", split="train")
    
    def test_validate_dataset_structure(self):
        """Test dataset validation."""
        dataset_dir = "dataset_aqg/output_domain/"
        
        if Path(dataset_dir).exists():
            dataset = self.loader.load_dataset(dataset_dir, split="train")
            validation_results = self.loader.validate_dataset(dataset)
            
            assert "total_entries" in validation_results
            assert "missing_fields" in validation_results
            assert "duplicate_count" in validation_results
            assert validation_results["total_entries"] > 0
            assert len(validation_results["missing_fields"]) == 0
    
    def test_analyze_token_distribution(self):
        """Test token distribution analysis."""
        from transformers import T5Tokenizer
        
        dataset_dir = "dataset_aqg/output_domain/"
        
        if Path(dataset_dir).exists():
            dataset = self.loader.load_dataset(dataset_dir, split="train")
            tokenizer = T5Tokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
            
            stats = self.loader.analyze_token_distribution(
                dataset.select(range(min(10, len(dataset)))),
                tokenizer,
                max_length=512
            )
            
            assert "mean_length" in stats
            assert "median_length" in stats
            assert "max_length_found" in stats
            assert "pct_exceeding_limit" in stats
            assert stats["mean_length"] > 0
