#!/usr/bin/env python3
"""
Script untuk validasi dataset yang sudah di-generate
Support multiple materi profiles (01, 02, dst)
"""
import json
import os
import glob
from pathlib import Path

def validate_sample(sample, line_num):
    """Validasi satu sample"""
    errors = []
    
    # Check required fields
    if 'input' not in sample:
        errors.append(f"Missing 'input' field")
    if 'output' not in sample:
        errors.append(f"Missing 'output' field")
    if 'metadata' not in sample:
        errors.append(f"Missing 'metadata' field")
    
    # Check input format
    if 'input' in sample:
        if not sample['input'].startswith('buat_soal_pilihan_ganda:'):
            errors.append(f"Input doesn't start with 'buat_soal_pilihan_ganda:'")
    
    # Check output format
    if 'output' in sample:
        output = sample['output']
        if 'question:' not in output:
            errors.append(f"Output missing 'question:' field")
        if 'answer:' not in output:
            errors.append(f"Output missing 'answer:' field")
        if 'distractors:' not in output:
            errors.append(f"Output missing 'distractors:' field")
    
    # Check metadata
    if 'metadata' in sample:
        meta = sample['metadata']
        if 'difficulty' not in meta:
            errors.append(f"Metadata missing 'difficulty'")
        elif meta['difficulty'] not in ['Mudah', 'Sedang', 'Sulit']:
            errors.append(f"Invalid difficulty: {meta['difficulty']}")
        
        if 'type' not in meta:
            errors.append(f"Metadata missing 'type'")
        elif meta['type'] not in ['knowledge', 'code']:
            errors.append(f"Invalid type: {meta['type']}")
    
    return errors

def validate_file(generated_file, verbose=False):
    """Validasi satu file dataset"""
    if not os.path.exists(generated_file):
        return None
    
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    knowledge_count = 0
    code_count = 0
    difficulty_dist = {'Mudah': 0, 'Sedang': 0, 'Sulit': 0}
    
    with open(generated_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_samples += 1
            try:
                sample = json.loads(line)
                errors = validate_sample(sample, line_num)
                
                if errors:
                    invalid_samples += 1
                    if verbose:
                        print(f"Line {line_num}: INVALID")
                        for error in errors:
                            print(f"  - {error}")
                else:
                    valid_samples += 1
                    # Count type and difficulty
                    if 'metadata' in sample:
                        meta = sample['metadata']
                        if meta['type'] == 'knowledge':
                            knowledge_count += 1
                        else:
                            code_count += 1
                        difficulty_dist[meta['difficulty']] += 1
            
            except json.JSONDecodeError as e:
                invalid_samples += 1
                if verbose:
                    print(f"Line {line_num}: JSON DECODE ERROR - {e}")
    
    return {
        'total': total_samples,
        'valid': valid_samples,
        'invalid': invalid_samples,
        'knowledge': knowledge_count,
        'code': code_count,
        'difficulty': difficulty_dist
    }

def main():
    """Main function"""
    base_path = "dataset_aqg/dataset-task-v4"
    
    # Find all materi folders
    materi_folders = sorted(glob.glob(os.path.join(base_path, "*")))
    materi_folders = [f for f in materi_folders if os.path.isdir(f)]
    
    if not materi_folders:
        print(f"No materi folders found in {base_path}")
        return
    
    print("=" * 80)
    print("DATASET VALIDATION - MULTIPLE PROFILES")
    print("=" * 80)
    
    overall_stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'knowledge': 0,
        'code': 0,
        'difficulty': {'Mudah': 0, 'Sedang': 0, 'Sulit': 0}
    }
    
    results = []
    
    for materi_folder in materi_folders:
        materi_name = os.path.basename(materi_folder)
        generated_file = os.path.join(materi_folder, f"{materi_name}_generated.jsonl")
        
        print(f"\nValidating: {materi_name}")
        print("-" * 80)
        
        stats = validate_file(generated_file, verbose=False)
        
        if stats is None:
            print(f"  ⚠️  Generated file not found: {generated_file}")
            continue
        
        results.append((materi_name, stats))
        
        # Update overall stats
        overall_stats['total'] += stats['total']
        overall_stats['valid'] += stats['valid']
        overall_stats['invalid'] += stats['invalid']
        overall_stats['knowledge'] += stats['knowledge']
        overall_stats['code'] += stats['code']
        for diff in ['Mudah', 'Sedang', 'Sulit']:
            overall_stats['difficulty'][diff] += stats['difficulty'][diff]
        
        # Print stats for this materi
        print(f"  Total samples: {stats['total']}")
        print(f"  Valid: {stats['valid']} | Invalid: {stats['invalid']}")
        
        if stats['total'] > 0:
            knowledge_pct = stats['knowledge'] / stats['total'] * 100
            code_pct = stats['code'] / stats['total'] * 100
            print(f"  Type: Knowledge {knowledge_pct:.1f}% | Code {code_pct:.1f}%")
            
            # Check requirements
            knowledge_ok = "✅" if knowledge_pct >= 60 else "❌"
            code_ok = "✅" if code_pct <= 40 else "❌"
            print(f"  Requirements: {knowledge_ok} Knowledge >= 60% | {code_ok} Code <= 40%")
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total profiles: {len(results)}")
    print(f"Total samples: {overall_stats['total']}")
    print(f"Valid: {overall_stats['valid']} | Invalid: {overall_stats['invalid']}")
    
    if overall_stats['total'] > 0:
        knowledge_pct = overall_stats['knowledge'] / overall_stats['total'] * 100
        code_pct = overall_stats['code'] / overall_stats['total'] * 100
        print(f"\nType Distribution:")
        print(f"  Knowledge: {overall_stats['knowledge']} ({knowledge_pct:.1f}%)")
        print(f"  Code: {overall_stats['code']} ({code_pct:.1f}%)")
        
        print(f"\nDifficulty Distribution:")
        for diff in ['Mudah', 'Sedang', 'Sulit']:
            count = overall_stats['difficulty'][diff]
            pct = count / overall_stats['total'] * 100
            print(f"  {diff}: {count} ({pct:.1f}%)")
        
        print(f"\nRequirements Check:")
        knowledge_ok = "✅" if knowledge_pct >= 60 else "❌"
        code_ok = "✅" if code_pct <= 40 else "❌"
        print(f"  {knowledge_ok} Knowledge >= 60%: {knowledge_pct:.1f}%")
        print(f"  {code_ok} Code <= 40%: {code_pct:.1f}%")

if __name__ == "__main__":
    main()
