"""Dataset loader untuk loading dan validasi JSONL datasets."""

from datasets import load_dataset, Dataset
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json


class DatasetLoader:
    """Class untuk load dan validate JSONL datasets."""
    
    def __init__(self):
        """Initialize DatasetLoader."""
        pass
    
    def load_dataset(
        self, 
        data_dir: str, 
        split: str = "train"
    ) -> Dataset:
        """
        Load dataset dari JSONL files.
        
        Args:
            data_dir: Path ke directory dataset (e.g., "dataset_aqg/output_domain/")
            split: "train", "validation", atau "test"
            
        Returns:
            HuggingFace Dataset object
            
        Raises:
            FileNotFoundError: Jika file tidak ditemukan
            ValueError: Jika format JSONL invalid
        """
        file_path = Path(data_dir) / f"{split}.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Please ensure the file exists in the specified directory."
            )
        
        try:
            dataset = load_dataset("json", data_files=str(file_path), split="train")
            print(f"✓ Loaded {len(dataset)} entries from {file_path}")
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {file_path}: {e}")
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate dataset structure dan integrity.
        
        Args:
            dataset: HuggingFace Dataset object
            
        Returns:
            Dict dengan validation results:
            - total_entries: int
            - missing_fields: List[str]
            - duplicate_count: int
            - avg_input_length: float
            - avg_target_length: float
            - has_metadata: bool
            - output_field: str (either 'target' or 'output')
        """
        # Check for input field (required)
        if "input" not in dataset.column_names:
            raise ValueError("Missing required field: 'input'")
        
        # Check for output field (support both 'target' and 'output')
        output_field = None
        if "target" in dataset.column_names:
            output_field = "target"
        elif "output" in dataset.column_names:
            output_field = "output"
        else:
            raise ValueError(
                "Missing output field. Dataset must have either 'target' or 'output' field.\n"
                f"Available fields: {dataset.column_names}"
            )
        
        print(f"✓ Using output field: '{output_field}'")
        
        # Calculate statistics
        total_entries = len(dataset)
        
        # Check for duplicates
        inputs = [item["input"] for item in dataset]
        duplicate_count = len(inputs) - len(set(inputs))
        
        # Calculate average lengths (use detected output field)
        avg_input_length = sum(len(item["input"]) for item in dataset) / total_entries
        avg_target_length = sum(len(item[output_field]) for item in dataset) / total_entries
        
        # Check metadata
        has_metadata = "metadata" in dataset.column_names
        
        results = {
            "total_entries": total_entries,
            "missing_fields": [],
            "duplicate_count": duplicate_count,
            "avg_input_length": round(avg_input_length, 2),
            "avg_target_length": round(avg_target_length, 2),
            "has_metadata": has_metadata,
            "output_field": output_field  # Store which field is used
        }
        
        # Print validation summary
        print("\n=== Dataset Validation Summary ===")
        print(f"Total Entries: {results['total_entries']}")
        print(f"Duplicate Count: {results['duplicate_count']}")
        print(f"Avg Input Length: {results['avg_input_length']} chars")
        print(f"Avg Target Length: {results['avg_target_length']} chars")
        print(f"Has Metadata: {results['has_metadata']}")
        
        if duplicate_count > 0:
            print(f"⚠ Warning: Found {duplicate_count} duplicate entries")
        else:
            print("✓ No duplicates found")
        
        return results
    
    def analyze_token_distribution(
        self, 
        dataset: Dataset, 
        tokenizer,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Analyze distribusi panjang token.
        
        Args:
            dataset: HuggingFace Dataset object
            tokenizer: PreTrainedTokenizer
            max_length: Maximum token length
            
        Returns:
            Dict dengan statistics:
            - mean_length: float
            - median_length: float
            - max_length_found: int
            - pct_exceeding_limit: float
            - histogram: List[Tuple[str, int]]
        """
        # Detect output field
        output_field = "target" if "target" in dataset.column_names else "output"
        
        token_lengths = []
        
        for item in dataset:
            # Tokenize input + output (support both 'target' and 'output')
            input_tokens = tokenizer(item["input"], truncation=False)["input_ids"]
            target_tokens = tokenizer(item[output_field], truncation=False)["input_ids"]
            total_length = len(input_tokens) + len(target_tokens)
            token_lengths.append(total_length)
        
        # Calculate statistics
        mean_length = sum(token_lengths) / len(token_lengths)
        sorted_lengths = sorted(token_lengths)
        median_length = sorted_lengths[len(sorted_lengths) // 2]
        max_length_found = max(token_lengths)
        
        # Count exceeding limit
        exceeding_count = sum(1 for length in token_lengths if length > max_length)
        pct_exceeding_limit = (exceeding_count / len(token_lengths)) * 100
        
        # Create histogram bins
        bins = [0, 128, 256, 384, 512, 640, 768, 1024, float('inf')]
        bin_labels = ["0-128", "128-256", "256-384", "384-512", "512-640", "640-768", "768-1024", "1024+"]
        histogram = []
        
        for i in range(len(bins) - 1):
            count = sum(1 for length in token_lengths if bins[i] <= length < bins[i+1])
            histogram.append((bin_labels[i], count))
        
        results = {
            "mean_length": round(mean_length, 2),
            "median_length": median_length,
            "max_length_found": max_length_found,
            "pct_exceeding_limit": round(pct_exceeding_limit, 2),
            "histogram": histogram
        }
        
        # Print analysis summary
        print("\n=== Token Distribution Analysis ===")
        print(f"Mean Length: {results['mean_length']} tokens")
        print(f"Median Length: {results['median_length']} tokens")
        print(f"Max Length: {results['max_length_found']} tokens")
        print(f"Exceeding {max_length} tokens: {results['pct_exceeding_limit']}%")
        print("\nHistogram:")
        for bin_label, count in histogram:
            bar = "█" * (count // 5) if count > 0 else ""
            print(f"  {bin_label:>10}: {count:>4} {bar}")
        
        if pct_exceeding_limit > 5:
            print(f"\n⚠ Warning: {pct_exceeding_limit}% of samples exceed max_length={max_length}")
        else:
            print(f"\n✓ Only {pct_exceeding_limit}% exceed max_length, acceptable")
        
        return results
