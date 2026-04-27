#!/usr/bin/env python3
"""
Script untuk validasi kesesuaian dataset dengan Design Guide.
Checks:
1. Format JSONL
2. Field names (input/output vs input/target)
3. Task prefix "buat_soal_pilihan_ganda:"
4. Output format (question/answer/distractors)
5. Code blocks in questions (self-contained)
6. Input context completeness
7. Balance konseptual vs code blocks
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class DatasetValidator:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def validate_file(self, file_path: Path) -> Dict:
        """Validate single JSONL file"""
        results = {
            'total': 0,
            'errors': [],
            'warnings': [],
            'has_code_blocks': 0,
            'no_code_blocks': 0,
            'missing_prefix': 0,
            'wrong_field_names': 0,
            'wrong_output_format': 0,
            'short_input_context': 0,
            'code_not_in_question': 0,
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    try:
                        obj = json.loads(line)
                        results['total'] += 1
                        
                        # Check 1: Field names (should be input/output, not input/target)
                        if 'target' in obj:
                            results['wrong_field_names'] += 1
                            results['warnings'].append(
                                f"Line {line_num}: Uses 'target' instead of 'output'"
                            )
                        
                        if 'input' not in obj or 'output' not in obj:
                            if 'input' not in obj or 'target' not in obj:
                                results['errors'].append(
                                    f"Line {line_num}: Missing required fields"
                                )
                                continue
                        
                        input_text = obj.get('input', '')
                        output_text = obj.get('output', obj.get('target', ''))
                        
                        # Check 2: Task prefix
                        if not input_text.startswith('buat_soal_pilihan_ganda:'):
                            results['missing_prefix'] += 1
                            results['errors'].append(
                                f"Line {line_num}: Missing task prefix 'buat_soal_pilihan_ganda:'"
                            )
                        
                        # Check 3: Output format
                        if not self._check_output_format(output_text):
                            results['wrong_output_format'] += 1
                            results['errors'].append(
                                f"Line {line_num}: Wrong output format (should be question:/answer:/distractors:)"
                            )
                        
                        # Check 4: Input context completeness
                        # Remove prefix for analysis
                        context = input_text.replace('buat_soal_pilihan_ganda:', '').strip()
                        if len(context) < 50:  # Too short
                            results['short_input_context'] += 1
                            results['warnings'].append(
                                f"Line {line_num}: Input context too short ({len(context)} chars)"
                            )
                        
                        # Check 5: Code blocks
                        has_code_in_input = '```python' in input_text or '```' in input_text
                        has_code_in_output = '```python' in output_text or '```' in output_text
                        
                        if has_code_in_input or has_code_in_output:
                            results['has_code_blocks'] += 1
                            
                            # Check if question is self-contained
                            if has_code_in_input and not has_code_in_output:
                                # Question refers to code but doesn't include it
                                if any(phrase in output_text.lower() for phrase in 
                                      ['kode di atas', 'kode berikut', 'kode tersebut', 'program di atas']):
                                    results['code_not_in_question'] += 1
                                    results['warnings'].append(
                                        f"Line {line_num}: Question refers to code but doesn't include it (not self-contained)"
                                    )
                        else:
                            results['no_code_blocks'] += 1
                        
                    except json.JSONDecodeError as e:
                        results['errors'].append(
                            f"Line {line_num}: JSON decode error - {e}"
                        )
                    except Exception as e:
                        results['errors'].append(
                            f"Line {line_num}: Unexpected error - {e}"
                        )
        
        except Exception as e:
            results['errors'].append(f"File error: {e}")
        
        return results
    
    def _check_output_format(self, output: str) -> bool:
        """Check if output follows the correct format"""
        required_parts = ['question:', 'answer:', 'distractors:']
        return all(part in output.lower() for part in required_parts)
    
    def validate_dataset(self, dataset_dir: Path) -> Dict:
        """Validate entire dataset directory"""
        all_results = {
            'files_checked': 0,
            'total_samples': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'code_balance': {},
            'file_results': {}
        }
        
        # Find all JSONL files (exclude accumulated and split files)
        jsonl_files = []
        for section_dir in sorted(dataset_dir.iterdir()):
            if section_dir.is_dir() and not section_dir.name.startswith('00-'):
                jsonl_files.extend(section_dir.glob('*.jsonl'))
        
        print(f"\n{'='*80}")
        print(f"VALIDATING DATASET: {dataset_dir.name}")
        print(f"{'='*80}\n")
        
        for file_path in sorted(jsonl_files):
            results = self.validate_file(file_path)
            all_results['files_checked'] += 1
            all_results['total_samples'] += results['total']
            all_results['total_errors'] += len(results['errors'])
            all_results['total_warnings'] += len(results['warnings'])
            
            # Store file results
            rel_path = file_path.relative_to(dataset_dir)
            all_results['file_results'][str(rel_path)] = results
            
            # Print file summary
            if results['errors'] or results['warnings']:
                print(f"\n[FILE] {rel_path}")
                print(f"   Samples: {results['total']}")
                print(f"   Errors: {len(results['errors'])}")
                print(f"   Warnings: {len(results['warnings'])}")
                
                if results['errors']:
                    print(f"   [X] Errors:")
                    for error in results['errors'][:3]:  # Show first 3
                        print(f"      - {error}")
                    if len(results['errors']) > 3:
                        print(f"      ... and {len(results['errors']) - 3} more")
                
                if results['warnings']:
                    print(f"   [!] Warnings:")
                    for warning in results['warnings'][:3]:  # Show first 3
                        print(f"      - {warning}")
                    if len(results['warnings']) > 3:
                        print(f"      ... and {len(results['warnings']) - 3} more")
        
        # Calculate code balance
        total_code = sum(r['has_code_blocks'] for r in all_results['file_results'].values())
        total_no_code = sum(r['no_code_blocks'] for r in all_results['file_results'].values())
        total = total_code + total_no_code
        
        all_results['code_balance'] = {
            'with_code': total_code,
            'no_code': total_no_code,
            'code_percentage': (total_code / total * 100) if total > 0 else 0,
            'no_code_percentage': (total_no_code / total * 100) if total > 0 else 0
        }
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print validation summary"""
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Files checked: {results['files_checked']}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Total errors: {results['total_errors']}")
        print(f"Total warnings: {results['total_warnings']}")
        
        print(f"\n[CODE BALANCE]:")
        cb = results['code_balance']
        print(f"   With code blocks: {cb['with_code']} ({cb['code_percentage']:.1f}%)")
        print(f"   No code blocks: {cb['no_code']} ({cb['no_code_percentage']:.1f}%)")
        
        # Check if balance is good (target: 40-50% code, 50-60% no code)
        if 40 <= cb['code_percentage'] <= 50:
            print(f"   [OK] Balance is GOOD (target: 40-50% code)")
        elif cb['code_percentage'] > 50:
            print(f"   [!] Too many code blocks (target: 40-50%)")
        else:
            print(f"   [!] Too few code blocks (target: 40-50%)")
        
        print(f"\n{'='*80}\n")


def main():
    # Validate both datasets
    base_path = Path('dataset_aqg/dataset-task-v3')
    
    datasets = [
        base_path / '00-dataset',
        base_path / '00-dataset-no-code'
    ]
    
    for dataset_dir in datasets:
        if not dataset_dir.exists():
            print(f"❌ Dataset not found: {dataset_dir}")
            continue
        
        # Skip the accumulated dataset directories, validate source files
        validator = DatasetValidator(base_path)
        results = validator.validate_dataset(base_path)
        validator.print_summary(results)


if __name__ == '__main__':
    main()
