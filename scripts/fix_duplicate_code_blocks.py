#!/usr/bin/env python3
"""
Fix duplicate code blocks in questions.
Some questions have code blocks repeated twice.
"""

import json
import re
from pathlib import Path

def fix_duplicate_code_blocks(file_path: Path) -> int:
    """Remove duplicate code blocks from questions"""
    print(f"\nFixing: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    fixes = 0
    
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            output = obj.get('output', '')
            
            # Check for duplicate code blocks
            # Pattern: ```python\n...\n```\n repeated twice
            code_blocks = re.findall(r'```python\n.*?\n```', output, re.DOTALL)
            
            if len(code_blocks) >= 2:
                # Check if first two blocks are identical
                if code_blocks[0] == code_blocks[1]:
                    # Remove first occurrence, keep second
                    # This keeps the code block that's closer to the question text
                    output_fixed = output.replace(code_blocks[0] + '\n', '', 1)
                    obj['output'] = output_fixed
                    fixes += 1
                    print(f"  Line {i}: Removed duplicate code block")
            
            fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
            
        except json.JSONDecodeError:
            fixed_lines.append(line)
        except Exception as e:
            print(f"  Warning line {i}: {e}")
            fixed_lines.append(line)
    
    if fixes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        print(f"  Fixed {fixes} duplicate code blocks")
    else:
        print(f"  No duplicates found")
    
    return fixes

def main():
    base_path = Path('dataset_aqg/dataset-task-v3')
    
    # Check all files
    total_fixes = 0
    for section_dir in sorted(base_path.iterdir()):
        if section_dir.is_dir() and not section_dir.name.startswith('00-'):
            for jsonl_file in sorted(section_dir.glob('*.jsonl')):
                fixes = fix_duplicate_code_blocks(jsonl_file)
                total_fixes += fixes
    
    print(f"\n{'='*60}")
    print(f"Total fixes: {total_fixes}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
