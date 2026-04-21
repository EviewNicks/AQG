"""
Transform dataset format to align with HuggingFace standard.

Removes:
- 'Konteks: ' prefix from input
- Prompt instruction from input
- 'Pertanyaan: ' prefix from target
- Answer and distractors from target (keep only question)

Preserves:
- Markdown formatting (##, **, ``, etc.)
- Metadata (saved separately)
- UTF-8 encoding

Based on analysis in docs/brainstorm-action-plan.md
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def clean_input(text: str) -> str:
    """
    Remove 'Konteks: ' prefix and prompt instruction from input.
    
    Args:
        text: Original input text
        
    Returns:
        Cleaned input text (context only)
    """
    # Remove 'Konteks: ' prefix
    if text.startswith('Konteks: '):
        text = text[len('Konteks: '):]
    
    # Remove prompt instruction (everything after '\n\nPrompt:')
    if '\n\nPrompt:' in text:
        text = text.split('\n\nPrompt:')[0]
    
    return text.strip()


def clean_target(text: str) -> str:
    """
    Extract only question from target, remove prefix and answer/distractors.
    
    Args:
        text: Original target text
        
    Returns:
        Cleaned target text (question only)
    """
    # Handle XML tags or model explanations before "Pertanyaan:"
    # Extract text after LAST occurrence of "Pertanyaan:" (case-insensitive)
    pertanyaan_pattern = re.compile(r'Pertanyaan:\s*', re.IGNORECASE)
    matches = list(pertanyaan_pattern.finditer(text))
    if matches:
        # Get text after the last "Pertanyaan:"
        last_match = matches[-1]
        text = text[last_match.end():]
    
    # Remove answer and distractors
    # Pattern 1: Find "Jawaban benar:" (case-insensitive) and remove everything after it
    jawaban_pattern = re.compile(r'\s*Jawaban benar:', re.IGNORECASE)
    match = jawaban_pattern.search(text)
    if match:
        text = text[:match.start()]
    
    # Pattern 2: Find "Distraktor:" and remove everything after it
    distraktor_pattern = re.compile(r'\s*Distraktor:', re.IGNORECASE)
    match = distraktor_pattern.search(text)
    if match:
        text = text[:match.start()]
    
    # Clean up and ensure ends with '?'
    text = text.strip()
    if not text.endswith('?'):
        text += '?'
    
    return text


def extract_metadata(item: Dict) -> Dict:
    """
    Extract metadata from item.
    
    Args:
        item: Original item with metadata
        
    Returns:
        Metadata dictionary
    """
    metadata = item.get('metadata', {})
    
    # If metadata not present, try to extract from target
    if not metadata:
        target = item.get('target', '')
        
        # Extract answer
        if 'Jawaban benar:' in target:
            match = re.search(r'Jawaban benar:\s*`?([^`\n]+)`?\.', target)
            if match:
                metadata['answer'] = match.group(1).strip()
        
        # Extract distractors
        if 'Distraktor:' in target:
            match = re.search(r'Distraktor:\s*(.+?)(?:\.\.\.|$)', target)
            if match:
                distractors_text = match.group(1)
                distractors = re.findall(r'\d+\)\s*`?([^`\n]+)`?', distractors_text)
                if distractors:
                    metadata['distractors'] = distractors
    
    return metadata


def validate_item(item: Dict, line_num: int) -> Tuple[bool, Optional[str]]:
    """
    Validate item structure and content.
    
    Args:
        item: Item to validate
        line_num: Line number for error reporting
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check keys
    if 'input' not in item or 'target' not in item:
        return False, f"Line {line_num}: Missing required keys (input/target)"
    
    # Check types
    if not isinstance(item['input'], str):
        return False, f"Line {line_num}: Input is not string"
    if not isinstance(item['target'], str):
        return False, f"Line {line_num}: Target is not string"
    
    # Check empty
    if not item['input'].strip():
        return False, f"Line {line_num}: Input is empty"
    if not item['target'].strip():
        return False, f"Line {line_num}: Target is empty"
    
    # Check length (warning, not error)
    if len(item['input']) > 10000:
        print(f"  ⚠️ Line {line_num}: Very long input ({len(item['input'])} chars)")
    if len(item['target']) > 1000:
        print(f"  ⚠️ Line {line_num}: Very long target ({len(item['target'])} chars)")
    
    return True, None


def transform_file(input_path: Path, output_path: Path, metadata_path: Path) -> Dict:
    """
    Transform one JSONL file and save metadata separately.
    
    Args:
        input_path: Path to original file
        output_path: Path to save transformed file
        metadata_path: Path to save metadata file
        
    Returns:
        Dict with statistics
    """
    stats = {
        'total': 0,
        'transformed': 0,
        'errors': 0,
        'had_konteks_prefix': 0,
        'had_prompt': 0,
        'had_pertanyaan_prefix': 0,
        'had_answer': 0
    }
    
    try:
        # Read input file with error handling
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    is_valid, error_msg = validate_item(item, line_num)
                    if is_valid:
                        data.append(item)
                    else:
                        print(f"  ⚠️ {error_msg}")
                        stats['errors'] += 1
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ Line {line_num}: Invalid JSON - {e}")
                    stats['errors'] += 1
        
        stats['total'] = len(data)
        
        # Transform and extract metadata
        transformed = []
        metadata_list = []
        
        for item in data:
            original_input = item['input']
            original_target = item['target']
            
            # Track what was removed
            if original_input.startswith('Konteks: '):
                stats['had_konteks_prefix'] += 1
            if '\n\nPrompt:' in original_input:
                stats['had_prompt'] += 1
            if original_target.startswith('Pertanyaan: '):
                stats['had_pertanyaan_prefix'] += 1
            if 'Jawaban benar:' in original_target:
                stats['had_answer'] += 1
            
            # Transform
            transformed_item = {
                'input': clean_input(original_input),
                'target': clean_target(original_target)
            }
            transformed.append(transformed_item)
            
            # Extract metadata
            metadata = extract_metadata(item)
            metadata_list.append(metadata)
            
            stats['transformed'] += 1
        
        # Save transformed data
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in transformed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for meta in metadata_list:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        return stats
    
    except FileNotFoundError:
        print(f"  ❌ File not found: {input_path}")
        return stats
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return stats


def verify_transformation(file_path: Path) -> Dict[str, List[int]]:
    """
    Verify that transformation was successful.
    
    Args:
        file_path: Path to transformed file
        
    Returns:
        Dict with problematic line numbers
    """
    issues = {
        'has_konteks': [],
        'has_prompt': [],
        'has_pertanyaan': [],
        'has_answer': [],
        'unclosed_code_blocks': [],  # Warning only (may exist in original data)
        'unbalanced_bold': []  # Warning only (may exist in original data)
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    
                    # Critical issues (should not exist after transformation)
                    if 'Konteks:' in item['input']:
                        issues['has_konteks'].append(i)
                    if 'Prompt:' in item['input']:
                        issues['has_prompt'].append(i)
                    if re.search(r'Pertanyaan:', item['target'], re.IGNORECASE):
                        issues['has_pertanyaan'].append(i)
                    if re.search(r'Jawaban benar:', item['target'], re.IGNORECASE):
                        issues['has_answer'].append(i)
                    
                    # Markdown issues (warning only - may exist in original)
                    if item['input'].count('```') % 2 != 0:
                        issues['unclosed_code_blocks'].append(i)
                    if item['input'].count('**') % 2 != 0:
                        issues['unbalanced_bold'].append(i)
                
                except json.JSONDecodeError:
                    pass
    
    except FileNotFoundError:
        print(f"  ❌ File not found: {file_path}")
    
    return issues


def main():
    """Main transformation pipeline."""
    source_dir = Path('dataset_aqg/dataset-task-spesifc')
    output_dir = Path('dataset_aqg/dataset-task-v2')
    
    print("=" * 70)
    print("DATASET FORMAT TRANSFORMATION (IMPROVED)")
    print("=" * 70)
    print()
    print("Transforming dataset to HuggingFace standard format:")
    print("  - Remove 'Konteks:' prefix from input")
    print("  - Remove prompt instruction from input")
    print("  - Remove 'Pertanyaan:' prefix from target")
    print("  - Keep only question in target (remove answer/distractors)")
    print("  - Save metadata separately")
    print()
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    print("Step 1: Creating output directory...")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory ready: {output_dir}")
    
    print()
    
    # Transform files
    print("Step 2: Transforming files...")
    all_stats = {}
    for split in ['train', 'validation', 'test']:
        input_file = source_dir / f'{split}.jsonl'
        output_file = output_dir / f'{split}.jsonl'
        metadata_file = output_dir / f'{split}_metadata.jsonl'
        
        print(f"\n  Processing {split}.jsonl...")
        stats = transform_file(input_file, output_file, metadata_file)
        all_stats[split] = stats
        
        if stats['transformed'] > 0:
            print(f"    ✓ Transformed {stats['transformed']} samples")
            print(f"      - Removed 'Konteks:' prefix: {stats['had_konteks_prefix']}")
            print(f"      - Removed prompt instruction: {stats['had_prompt']}")
            print(f"      - Removed 'Pertanyaan:' prefix: {stats['had_pertanyaan_prefix']}")
            print(f"      - Removed answer/distractors: {stats['had_answer']}")
            print(f"      - Errors: {stats['errors']}")
            print(f"    ✓ Metadata saved to: {split}_metadata.jsonl")
        else:
            print(f"    ❌ No samples transformed")
    
    print()
    
    # Verify transformation
    print("Step 3: Verifying transformation...")
    all_clean = True
    critical_issues_found = False
    
    for split in ['train', 'validation', 'test']:
        output_file = output_dir / f'{split}.jsonl'
        issues = verify_transformation(output_file)
        
        # Separate critical issues from warnings
        critical_issues = {
            'has_konteks': issues['has_konteks'],
            'has_prompt': issues['has_prompt'],
            'has_pertanyaan': issues['has_pertanyaan'],
            'has_answer': issues['has_answer']
        }
        
        warnings = {
            'unclosed_code_blocks': issues['unclosed_code_blocks'],
            'unbalanced_bold': issues['unbalanced_bold']
        }
        
        has_critical = any(len(v) > 0 for v in critical_issues.values())
        has_warnings = any(len(v) > 0 for v in warnings.values())
        
        if has_critical:
            critical_issues_found = True
            all_clean = False
            print(f"\n  ❌ {split}.jsonl has CRITICAL issues:")
            for issue_type, line_nums in critical_issues.items():
                if line_nums:
                    sample = line_nums[:5]
                    suffix = f"... ({len(line_nums) - 5} more)" if len(line_nums) > 5 else ""
                    print(f"      {issue_type}: lines {sample}{suffix}")
        
        if has_warnings:
            print(f"\n  ⚠️  {split}.jsonl has warnings (may exist in original data):")
            for issue_type, line_nums in warnings.items():
                if line_nums:
                    sample = line_nums[:5]
                    suffix = f"... ({len(line_nums) - 5} more)" if len(line_nums) > 5 else ""
                    print(f"      {issue_type}: lines {sample}{suffix}")
        
        if not has_critical and not has_warnings:
            print(f"\n  ✓ {split}.jsonl is clean")
    
    print()
    
    # Show sample
    print("Step 4: Sample comparison...")
    sample_file = output_dir / 'train.jsonl'
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
        
        print(f"\n  Sample transformed item:")
        print(f"  Input (first 150 chars): {sample['input'][:150]}...")
        print(f"  Target (first 150 chars): {sample['target'][:150]}...")
    except Exception as e:
        print(f"  ⚠️ Could not read sample: {e}")
    
    print()
    print("=" * 70)
    if all_clean and not critical_issues_found:
        print("✓ TRANSFORMATION SUCCESSFUL")
        print()
        print(f"Output saved to: {output_dir}")
        print()
        print("Files created:")
        print("  - train.jsonl, validation.jsonl, test.jsonl")
        print("  - train_metadata.jsonl, validation_metadata.jsonl, test_metadata.jsonl")
        print()
        print("Next steps:")
        print("  1. Update training script to use dataset-task-v2 folder")
        print("  2. Use *_metadata.jsonl for post-processing and analysis")
        print("  3. Re-run training")
        print("  4. Compare metrics (expect 50-70% improvement)")
    elif critical_issues_found:
        print("❌ TRANSFORMATION FAILED - CRITICAL ISSUES FOUND")
        print("  Please review the critical issues above and fix the script")
    else:
        print("⚠️  TRANSFORMATION COMPLETED WITH WARNINGS")
        print("  Warnings are non-critical (may exist in original data)")
        print("  You can proceed with training")
    print("=" * 70)


if __name__ == '__main__':
    main()