"""Unit tests untuk TokenizerTester."""

import pytest
from transformers import T5Tokenizer
from src.finetuned.data.tokenizer_tester import TokenizerTester


class TestTokenizerTester:
    """Test cases untuk TokenizerTester class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = T5Tokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
        self.tester = TokenizerTester(self.tokenizer)
    
    def test_markdown_handling(self):
        """Test markdown character handling."""
        results = self.tester.test_markdown_handling()
        
        assert isinstance(results, dict)
        assert "headings" in results
        assert "bold" in results
        assert "code" in results
        assert "newlines" in results
    
    def test_code_block_integrity(self):
        """Test code block preservation."""
        samples = [
            "def hello():\n    print('Hello')",
            "x = 10\ny = 20\nprint(x + y)"
        ]
        
        results = self.tester.test_code_block_integrity(samples)
        
        assert len(results) == len(samples)
        for result in results:
            assert "original" in result
            assert "decoded" in result
            assert "integrity_preserved" in result
            assert "token_count" in result
    
    def test_detect_oov_tokens(self):
        """Test OOV token detection."""
        from datasets import Dataset
        
        # Create dummy dataset
        data = {
            "input": ["Test input 1", "Test input 2"],
            "target": ["Test output 1", "Test output 2"]
        }
        dataset = Dataset.from_dict(data)
        
        oov_tokens = self.tester.detect_oov_tokens(dataset, max_samples=2)
        
        assert isinstance(oov_tokens, dict)
