#!/usr/bin/env python3
"""
Add Type Metadata Script
Adds 'type' field to metadata based on presence of code blocks.

Rules:
- If output contains ```python or ```bash or ``` → type = 'code'
- Otherwise → type = 'knowledge'

Usage:
    python 04_add_type_metadata.py <dataset_file>
    
Example:
    python 04_add_type_metadata.py dataset.jsonl
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict


def detect_code_type(sample):
    """
    Detect if sample contains code based on output.
    
    Args:
        sample: Dictionary with 'input' and 'output' fields
        
    Returns:
        'code' if code block detected, 'knowledge' otherwise
    """
    output = sample.get('output', '')
    
    # Check for code blocks (```python, ```bash, ```, etc.)
    if re.search(r'```', output):
        return 'code'
    
    return 'knowledge'


def add_type_metadata(dataset_file):
    """
    Add type metadata to all samples in dataset.
    
    Args:
        dataset_file: Path to JSONL dataset file
    """
    print(f"📖 Reading dataset: {dataset_file}")
    
    samples = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Add type metadata
    print(f"\n🔍 Analyzing samples and adding type metadata...")
    
    type_dist = defaultdict(int)
    samples_updated = 0
    samples_already_have_type = 0
    
    for i, sample in enumerate(samples):
        # Ensure metadata exists
        if 'metadata' not in sample:
            sample['metadata'] = {}
        
        # Check if type already exists
        if 'type' in sample['metadata']:
            samples_already_have_type += 1
            sample_type = sample['metadata']['type']
        else:
            # Detect type from output
            sample_type = detect_code_type(sample)
            sample['metadata']['type'] = sample_type
            samples_updated += 1
        
        type_dist[sample_type] += 1
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(samples)} samples...")
    
    print(f"\n📊 Type Distribution:")
    for sample_type, count in sorted(type_dist.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"   {sample_type}: {count} ({pct:.1f}%)")
    
    print(f"\n📈 Metadata Update Summary:")
    print(f"   Samples updated: {samples_updated}")
    print(f"   Samples already had type: {samples_already_have_type}")
    
    # Validate type distribution
    knowledge_count = type_dist.get('knowledge', 0)
    code_count = type_dist.get('code', 0)
    total = len(samples)
    
    knowledge_pct = (knowledge_count / total * 100) if total else 0
    code_pct = (code_count / total * 100) if total else 0
    
    print(f"\n✅ Type Distribution Check:")
    print(f"   Knowledge: {knowledge_count} ({knowledge_pct:.1f}%) - {'✅' if knowledge_pct >= 60 else '⚠️'}")
    print(f"   Code: {code_count} ({code_pct:.1f}%) - {'✅' if code_pct <= 40 else '⚠️'}")
    
    if knowledge_pct >= 60 and code_pct <= 40:
        print(f"   Status: ✅ Type distribution is valid")
    else:
        print(f"   Status: ⚠️  Type distribution may need adjustment")
    
    # Write updated file
    print(f"\n💾 Writing updated dataset: {dataset_file}")
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Update complete!")
    print(f"\n📋 Summary:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Samples updated: {samples_updated}")
    print(f"   File: {dataset_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 04_add_type_metadata.py <dataset_file>")
        print("\nExample:")
        print("  python 04_add_type_metadata.py dataset.jsonl")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    
    if not Path(dataset_file).exists():
        print(f"❌ File not found: {dataset_file}")
        sys.exit(1)
    
    add_type_metadata(dataset_file)
