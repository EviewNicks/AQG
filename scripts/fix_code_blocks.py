#!/usr/bin/env python3
import json
import re

def wrap_code_blocks(text):
    """Wrap code blocks with ```python markers"""
    # Skip if already has code blocks
    if '```python' in text or '```bash' in text:
        return text
    
    # Pattern: "Perhatikan kode" followed by code lines
    # We'll look for patterns like:
    # "Perhatikan kode:\nx = 5\nprint(x)\nKode di atas..."
    
    # Find "Perhatikan kode" patterns
    pattern = r'(Perhatikan kode(?:\s+berikut)?:\n)((?:[^\n]+\n)+?)([A-Z][^\n]+)'
    
    def replacer(match):
        prefix = match.group(1)  # "Perhatikan kode:\n"
        code_block = match.group(2).rstrip('\n')  # The code lines
        explanation = match.group(3)  # Explanation starting with capital
        
        # Check if code_block looks like code (has =, (), etc.)
        if any(char in code_block for char in ['=', '(', ')', 'print', 'def', 'class', 'import', 'for', 'if', 'while']):
            return f'{prefix}```python\n{code_block}\n```\n{explanation}'
        else:
            return match.group(0)  # Return unchanged
    
    result = re.sub(pattern, replacer, text)
    return result

def process_jsonl_file(filepath):
    """Process a JSONL file and fix code blocks"""
    print(f'\nProcessing: {filepath}')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    changes = 0
    
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            original = obj['input']
            fixed = wrap_code_blocks(original)
            
            if fixed != original:
                changes += 1
                obj['input'] = fixed
                print(f'  Line {i}: FIXED')
            
            fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f'  Line {i}: ERROR - {e}')
            fixed_lines.append(line)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f'  Total: {len(lines)} lines, Fixed: {changes} lines')
    return changes

if __name__ == '__main__':
    files = [
        'dataset_aqg\dataset-task-v3/00-dataset/accumulated.jsonl',
        'dataset_aqg\dataset-task-v3/00-dataset/test.jsonl',
        'dataset_aqg\dataset-task-v3/00-dataset/train.jsonl',
        'dataset_aqg\dataset-task-v3/00-dataset/validation.jsonl',
    ]
    
    total_changes = 0
    for f in files:
        total_changes += process_jsonl_file(f)
    
    print(f'\n=== DONE ===')
    print(f'Total changes: {total_changes}')
