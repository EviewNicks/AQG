"""Tokenizer tester untuk test compatibility dengan markdown dan code blocks."""

from typing import Dict, List, Any
from datasets import Dataset


class TokenizerTester:
    """Class untuk test tokenizer compatibility."""
    
    def __init__(self, tokenizer):
        """
        Initialize TokenizerTester.
        
        Args:
            tokenizer: PreTrainedTokenizer instance
        """
        self.tokenizer = tokenizer
    
    def test_markdown_handling(self) -> Dict[str, bool]:
        """
        Test tokenization untuk markdown special characters.
        
        Returns:
            Dict dengan test results untuk:
            - headings (#, ##)
            - bold (**text**)
            - code (`code`)
            - newlines (\n)
        """
        test_cases = {
            "headings": "# Heading 1\n## Heading 2",
            "bold": "**bold text** dan normal text",
            "code": "`inline code` dan normal text",
            "newlines": "line1\nline2\nline3",
            "mixed": "# Heading\n**Bold** dengan `code`"
        }
        
        results = {}
        print("\n=== Markdown Handling Test ===")
        
        for name, text in test_cases.items():
            tokens = self.tokenizer(text, return_tensors="pt")
            decoded = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            
            # Check if decoded text matches original (allowing minor whitespace differences)
            is_preserved = decoded.strip() == text.strip()
            results[name] = is_preserved
            
            status = "✓" if is_preserved else "✗"
            print(f"{status} {name:12}: {'PASS' if is_preserved else 'FAIL'}")
            if not is_preserved:
                print(f"  Original: {text[:50]}")
                print(f"  Decoded:  {decoded[:50]}")
        
        return results
    
    def test_code_block_integrity(self, samples: List[str]) -> List[Dict]:
        """
        Test apakah code blocks tetap intact setelah tokenization.
        
        Args:
            samples: List of text samples containing code blocks
            
        Returns:
            List of dicts dengan:
            - original: str
            - tokenized: List[int]
            - decoded: str
            - integrity_preserved: bool
        """
        results = []
        print("\n=== Code Block Integrity Test ===")
        
        for i, sample in enumerate(samples):
            tokens = self.tokenizer(sample, return_tensors="pt", truncation=False)
            token_ids = tokens["input_ids"][0].tolist()
            decoded = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            
            # Check integrity
            integrity_preserved = decoded.strip() == sample.strip()
            
            result = {
                "original": sample,
                "tokenized": token_ids,
                "decoded": decoded,
                "integrity_preserved": integrity_preserved,
                "token_count": len(token_ids)
            }
            results.append(result)
            
            status = "✓" if integrity_preserved else "✗"
            print(f"{status} Sample {i+1}: {len(token_ids)} tokens, {'PASS' if integrity_preserved else 'FAIL'}")
        
        passed = sum(1 for r in results if r["integrity_preserved"])
        print(f"\nPassed: {passed}/{len(samples)}")
        
        return results
    
    def detect_oov_tokens(self, dataset: Dataset, max_samples: int = 100) -> Dict[str, int]:
        """
        Detect out-of-vocabulary tokens.
        
        Args:
            dataset: HuggingFace Dataset object
            max_samples: Maximum number of samples to check
            
        Returns:
            Dict mapping OOV token -> frequency
        """
        oov_tokens = {}
        unk_token_id = self.tokenizer.unk_token_id
        
        print("\n=== OOV Token Detection ===")
        
        # Sample dataset
        samples = dataset.select(range(min(max_samples, len(dataset))))
        
        for item in samples:
            # Tokenize input and target
            for text in [item["input"], item["target"]]:
                tokens = self.tokenizer(text, return_tensors="pt")
                token_ids = tokens["input_ids"][0].tolist()
                
                # Check for UNK tokens
                if unk_token_id in token_ids:
                    # Find which words caused UNK
                    words = text.split()
                    for word in words:
                        word_tokens = self.tokenizer(word)["input_ids"]
                        if unk_token_id in word_tokens:
                            oov_tokens[word] = oov_tokens.get(word, 0) + 1
        
        # Print results
        if oov_tokens:
            print(f"Found {len(oov_tokens)} unique OOV tokens:")
            sorted_oov = sorted(oov_tokens.items(), key=lambda x: x[1], reverse=True)
            for token, freq in sorted_oov[:10]:  # Show top 10
                print(f"  {token:20}: {freq} occurrences")
            if len(sorted_oov) > 10:
                print(f"  ... and {len(sorted_oov) - 10} more")
        else:
            print("✓ No OOV tokens detected")
        
        return oov_tokens
    
    def generate_test_report(
        self, 
        dataset: Dataset,
        output_file: str = "tokenizer_test_report.txt"
    ) -> None:
        """
        Generate comprehensive test report.
        
        Args:
            dataset: HuggingFace Dataset object
            output_file: Path untuk save report
        """
        print("\n=== Generating Tokenizer Test Report ===")
        
        # Run all tests
        markdown_results = self.test_markdown_handling()
        
        # Sample code blocks from dataset
        code_samples = []
        for item in dataset.select(range(min(5, len(dataset)))):
            if "```" in item["input"]:
                code_samples.append(item["input"])
        
        if code_samples:
            code_results = self.test_code_block_integrity(code_samples)
        else:
            code_results = []
            print("⚠ No code blocks found in dataset samples")
        
        oov_tokens = self.detect_oov_tokens(dataset)
        
        # Write report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("TOKENIZER TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. Markdown Handling Test\n")
            f.write("-" * 60 + "\n")
            for test_name, passed in markdown_results.items():
                status = "PASS" if passed else "FAIL"
                f.write(f"  {test_name:15}: {status}\n")
            f.write("\n")
            
            f.write("2. Code Block Integrity Test\n")
            f.write("-" * 60 + "\n")
            if code_results:
                passed = sum(1 for r in code_results if r["integrity_preserved"])
                f.write(f"  Passed: {passed}/{len(code_results)}\n")
            else:
                f.write("  No code blocks tested\n")
            f.write("\n")
            
            f.write("3. OOV Token Detection\n")
            f.write("-" * 60 + "\n")
            if oov_tokens:
                f.write(f"  Found {len(oov_tokens)} unique OOV tokens\n")
                sorted_oov = sorted(oov_tokens.items(), key=lambda x: x[1], reverse=True)
                for token, freq in sorted_oov[:20]:
                    f.write(f"    {token:30}: {freq}\n")
            else:
                f.write("  No OOV tokens detected\n")
        
        print(f"✓ Report saved to {output_file}")
