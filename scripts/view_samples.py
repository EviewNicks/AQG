"""Quick script to view transformed dataset samples."""
import json
from pathlib import Path

def view_samples(file_path: Path, n: int = 3):
    """View first n samples from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f.readlines()[:n]]
    
    print(f"\n{'='*80}")
    print(f"FILE: {file_path.name}")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(samples, 1):
        print(f"--- Sample {i} ---")
        print(f"INPUT (first 120 chars):\n  {sample['input'][:120]}...")
        print(f"\nTARGET:\n  {sample['target']}")
        print()

if __name__ == '__main__':
    output_dir = Path('dataset_aqg/dataset-task-v2')
    
    for split in ['train', 'validation', 'test']:
        view_samples(output_dir / f'{split}.jsonl', n=2)
