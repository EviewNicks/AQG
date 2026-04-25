#!/usr/bin/env python3
"""
Script sederhana untuk memperbaiki format kode dalam dataset JSONL.
Menambahkan ```python wrapper di sekitar contoh kode.
"""

import json
import re

def fix_code_in_text(text):
    """
    Menambahkan ```python wrapper di sekitar kode Python.
    Mendeteksi pola: "Perhatikan kode" diikuti kode multi-line.
    """
    # Jangan proses jika sudah ada ```python
    if '```python' in text:
        return text
    
    # Pattern 1: "Perhatikan kode:\n<code>\n<explanation>"
    # Pattern 2: "Perhatikan kode berikut:\n<code>\n<explanation>"
    
    # Cari semua kemunculan "Perhatikan kode"
    if 'Perhatikan kode' not in text:
        return text
    
    # Split by "Perhatikan kode" dan proses setiap bagian
    parts = re.split(r'(Perhatikan kode(?:\s+berikut)?:)', text)
    
    result = []
    for i, part in enumerate(parts):
        if i == 0:
            # Bagian sebelum "Perhatikan kode" pertama
            result.append(part)
        elif 'Perhatikan kode' in part:
            # Ini adalah "Perhatikan kode:" atau "Perhatikan kode berikut:"
            result.append(part)
        else:
            # Ini adalah bagian setelah "Perhatikan kode:"
            # Cari kode dan penjelasan
            lines = part.split('\\n')
            
            # Kumpulkan baris kode (baris yang bukan penjelasan)
            code_lines = []
            explanation_lines = []
            in_code = True
            
            for line in lines:
                if not line.strip():
                    if in_code and code_lines:
                        # Newline setelah kode
                        code_lines.append(line)
                    else:
                        explanation_lines.append(line)
                elif in_code:
                    # Cek apakah ini masih kode atau sudah penjelasan
                    # Penjelasan biasanya dimulai dengan huruf kapital dan kata-kata Indonesia
                    if (line.strip() and 
                        not line.strip()[0].isupper() or 
                        any(keyword in line for keyword in ['=', '(', ')', '[', ']', '{', '}', ':', 'print', 'def', 'class', 'import', 'for', 'while', 'if', '#'])):
                        code_lines.append(line)
                    else:
                        in_code = False
                        explanation_lines.append(line)
                else:
                    explanation_lines.append(line)
            
            # Rekonstruksi dengan ```python wrapper
            if code_lines:
                result.append('\\n```python\\n')
                result.append('\\n'.join(code_lines).strip())
                result.append('\\n```')
            if explanation_lines:
                result.append('\\n')
                result.append('\\n'.join(explanation_lines))
    
    return ''.join(result)

def process_file(filepath):
    """
    Memproses file JSONL dan memperbaiki format kode.
    """
    print(f'\\nProcessing: {filepath}')
    
    # Backup original
    with open(filepath, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    
    fixed_lines = []
    fixed_count = 0
    
    for i, line in enumerate(original_lines, 1):
        try:
            data = json.loads(line.strip())
            original_input = data['input']
            
            # Fix code blocks
            fixed_input = fix_code_in_text(original_input)
            
            if fixed_input != original_input:
                fixed_count += 1
                data['input'] = fixed_input
                print(f'  Line {i}: Fixed')
            
            fixed_lines.append(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f'  Line {i}: Error - {e}')
            fixed_lines.append(line.strip())
    
    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\\n')
    
    print(f'  Total lines: {len(original_lines)}')
    print(f'  Fixed lines: {fixed_count}')
    print(f'  Success!')
    
    return len(original_lines), fixed_count

if __name__ == '__main__':
    files = [
        'dataset_aqg/dataset-task-v3/01-perkenalan-python/05-variable-and-assignment.jsonl',
        'dataset_aqg/dataset-task-v3/01-perkenalan-python/06-input-output-and-comment.jsonl',
    ]
    
    total_files = len(files)
    total_lines = 0
    total_fixed = 0
    
    for filepath in files:
        lines, fixed = process_file(filepath)
        total_lines += lines
        total_fixed += fixed
    
    print(f'\\n=== SUMMARY ===')
    print(f'Files processed: {total_files}')
    print(f'Total lines: {total_lines}')
    print(f'Total fixed: {total_fixed}')
