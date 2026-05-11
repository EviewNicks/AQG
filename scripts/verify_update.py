#!/usr/bin/env python3
"""
Quick verification script untuk memastikan update berhasil.

Usage:
    python scripts/verify_update.py
"""

import json
from pathlib import Path
from typing import Dict, List


def verify_file(file_path: Path, expected_prefix: str) -> Dict:
    """Verify single file."""
    stats = {
        'filename': file_path.name,
        'total_lines': 0,
        'correct_prefix': 0,
        'wrong_prefix': 0,
        'errors': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        stats['total_lines'] += 1
                        
                        if 'input' in data:
                            if data['input'].startswith(expected_prefix):
                                stats['correct_prefix'] += 1
                            else:
                                stats['wrong_prefix'] += 1
                                if stats['wrong_prefix'] <= 3:  # Show first 3 errors
                                    stats['errors'].append(
                                        f"Line {line_num}: {data['input'][:50]}..."
                                    )
                    except json.JSONDecodeError:
                        stats['errors'].append(f"Line {line_num}: JSON decode error")
    except FileNotFoundError:
        stats['errors'].append(f"File not found: {file_path}")
    
    return stats


def main():
    """Main verification."""
    print("=" * 70)
    print("DATASET UPDATE VERIFICATION")
    print("=" * 70)
    
    # Configuration
    updated_dir = Path("dataset_aqg/dataset-task-v3/00-dataset-no-code-updated")
    expected_prefix = "generate_mcq"
    
    files_to_check = [
        "accumulated.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl"
    ]
    
    print(f"\nDirectory: {updated_dir}")
    print(f"Expected Prefix: '{expected_prefix}'")
    print("-" * 70)
    
    all_stats = []
    total_correct = 0
    total_wrong = 0
    total_lines = 0
    
    for filename in files_to_check:
        file_path = updated_dir / filename
        print(f"\nVerifying: {filename}")
        
        stats = verify_file(file_path, expected_prefix)
        all_stats.append(stats)
        
        total_lines += stats['total_lines']
        total_correct += stats['correct_prefix']
        total_wrong += stats['wrong_prefix']
        
        # Print results
        if stats['total_lines'] > 0:
            success_rate = (stats['correct_prefix'] / stats['total_lines']) * 100
            print(f"  Lines: {stats['total_lines']}")
            print(f"  Correct: {stats['correct_prefix']} ({success_rate:.1f}%)")
            print(f"  Wrong: {stats['wrong_prefix']}")
            
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")
                for error in stats['errors'][:3]:
                    print(f"    - {error}")
            else:
                print(f"  ✓ All prefixes correct!")
        else:
            print(f"  ✗ No lines found or file error")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal Lines Checked: {total_lines}")
    print(f"Correct Prefix: {total_correct}")
    print(f"Wrong Prefix: {total_wrong}")
    
    if total_lines > 0:
        success_rate = (total_correct / total_lines) * 100
        print(f"Success Rate: {success_rate:.2f}%")
        
        if success_rate == 100.0:
            print("\n✅ VERIFICATION PASSED - All prefixes updated correctly!")
        else:
            print(f"\n⚠ VERIFICATION FAILED - {total_wrong} lines with wrong prefix")
    else:
        print("\n✗ No data to verify")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
