#!/usr/bin/env python3
"""
Merge Batch Files Script
Merges multiple batch JSONL files into a single main JSONL file.

Usage:
    python 03_merge_batches.py <main_file> <batch_file_1> <batch_file_2> ...
    
Example:
    python 03_merge_batches.py dataset.jsonl dataset_batch_1.jsonl dataset_batch_2.jsonl
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def merge_batches(main_file, batch_files):
    """
    Merge batch files into main file.
    
    Args:
        main_file: Path to main JSONL file
        batch_files: List of batch JSONL file paths
    """
    all_samples = []
    
    # Read main file
    print(f"📖 Reading main file: {main_file}")
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))
        print(f"   ✅ Loaded {len(all_samples)} samples from main file")
    except FileNotFoundError:
        print(f"   ⚠️  Main file not found, starting fresh")
    
    # Read batch files
    for batch_file in batch_files:
        print(f"📖 Reading batch file: {batch_file}")
        try:
            batch_count = 0
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
                        batch_count += 1
            print(f"   ✅ Loaded {batch_count} samples from batch file")
        except FileNotFoundError:
            print(f"   ❌ Batch file not found: {batch_file}")
            continue
    
    print(f"\n📊 Total samples before deduplication: {len(all_samples)}")
    
    # Remove duplicates - keep first occurrence
    seen_inputs = set()
    unique_samples = []
    duplicate_count = 0
    
    for sample in all_samples:
        input_text = sample.get('input', '')
        if input_text not in seen_inputs:
            seen_inputs.add(input_text)
            unique_samples.append(sample)
        else:
            duplicate_count += 1
    
    print(f"🔄 Duplicate inputs removed: {duplicate_count}")
    print(f"📊 Total samples after deduplication: {len(unique_samples)}")
    
    # Analyze distribution
    type_dist = defaultdict(int)
    difficulty_dist = defaultdict(int)
    
    for sample in unique_samples:
        metadata = sample.get('metadata', {})
        sample_type = metadata.get('type', 'unknown')
        difficulty = metadata.get('difficulty', 'unknown')
        
        type_dist[sample_type] += 1
        difficulty_dist[difficulty] += 1
    
    print(f"\n📈 Type Distribution:")
    for sample_type, count in sorted(type_dist.items()):
        pct = (count / len(unique_samples) * 100) if unique_samples else 0
        print(f"   {sample_type}: {count} ({pct:.1f}%)")
    
    print(f"\n📈 Difficulty Distribution:")
    for difficulty, count in sorted(difficulty_dist.items()):
        pct = (count / len(unique_samples) * 100) if unique_samples else 0
        print(f"   {difficulty}: {count} ({pct:.1f}%)")
    
    # Write merged file
    print(f"\n💾 Writing merged file: {main_file}")
    with open(main_file, 'w', encoding='utf-8') as f:
        for sample in unique_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Merge complete!")
    print(f"\n📋 Summary:")
    print(f"   Total samples: {len(unique_samples)}")
    print(f"   Duplicates removed: {duplicate_count}")
    print(f"   File: {main_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 03_merge_batches.py <main_file> <batch_file_1> <batch_file_2> ...")
        print("\nExample:")
        print("  python 03_merge_batches.py dataset.jsonl dataset_batch_1.jsonl dataset_batch_2.jsonl")
        sys.exit(1)
    
    main_file = sys.argv[1]
    batch_files = sys.argv[2:]
    
    merge_batches(main_file, batch_files)
