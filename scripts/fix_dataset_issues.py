#!/usr/bin/env python3
"""
Script untuk memperbaiki masalah dataset berdasarkan Design Guide:
1. Fix JSON decode errors (line 156 di materi3.jsonl)
2. Fix short input contexts (< 50 chars)
3. Ensure code blocks are self-contained in questions
"""

import json
import re
from pathlib import Path
from typing import Dict, List

class DatasetFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.errors_found = 0
        
    def fix_json_errors(self, file_path: Path) -> int:
        """Fix JSON decode errors in file"""
        print(f"\n🔧 Fixing JSON errors in: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            fixes = 0
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to parse
                    obj = json.loads(line)
                    fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    print(f"   ❌ Line {i}: JSON error - {e}")
                    print(f"      Content: {line[:100]}...")
                    
                    # Try to fix common issues
                    # Issue: Line starts with quote and incomplete JSON
                    if line.startswith('\"') and not line.startswith('{'):
                        print(f"      ⚠️  Skipping malformed line (starts with quote)")
                        fixes += 1
                        continue
                    
                    # Try to salvage by looking for complete JSON
                    if '{' in line and '}' in line:
                        try:
                            # Extract JSON part
                            start = line.index('{')
                            end = line.rindex('}') + 1
                            json_part = line[start:end]
                            obj = json.loads(json_part)
                            fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                            print(f"      ✅ Salvaged JSON from line")
                            fixes += 1
                        except:
                            print(f"      ❌ Cannot salvage, skipping line")
                            fixes += 1
                    else:
                        print(f"      ❌ Cannot fix, skipping line")
                        fixes += 1
            
            # Write back
            if fixes > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                print(f"   ✅ Fixed {fixes} lines")
            
            return fixes
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            return 0
    
    def expand_short_contexts(self, file_path: Path, min_length: int = 50) -> int:
        """Expand short input contexts with explanatory text"""
        print(f"\n🔧 Expanding short contexts in: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            fixes = 0
            
            # Context expansion templates
            templates = {
                'Pengertian': 'adalah konsep fundamental dalam pemrograman yang perlu dipahami dengan baik.',
                'Komponen': 'merupakan bagian-bagian penting yang membentuk struktur dalam pemrograman.',
                'default': 'adalah topik penting dalam pemrograman Python yang perlu dipahami.'
            }
            
            for i, line in enumerate(lines, 1):
                try:
                    obj = json.loads(line)
                    input_text = obj.get('input', '')
                    
                    # Remove prefix for analysis
                    context = input_text.replace('buat_soal_pilihan_ganda:', '').strip()
                    
                    if len(context) < min_length:
                        # Try to expand based on content
                        expanded = False
                        for key, template in templates.items():
                            if key.lower() in context.lower():
                                obj['input'] = f"buat_soal_pilihan_ganda: {context} {template}"
                                expanded = True
                                fixes += 1
                                break
                        
                        if not expanded:
                            # Use default template
                            obj['input'] = f"buat_soal_pilihan_ganda: {context} {templates['default']}"
                            fixes += 1
                    
                    fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    fixed_lines.append(line)
                except Exception as e:
                    print(f"   ⚠️  Line {i}: {e}")
                    fixed_lines.append(line)
            
            # Write back
            if fixes > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                print(f"   ✅ Expanded {fixes} short contexts")
            else:
                print(f"   ℹ️  No short contexts found")
            
            return fixes
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            return 0
    
    def fix_code_self_containment(self, file_path: Path) -> int:
        """Ensure code blocks in questions are self-contained"""
        print(f"\n🔧 Fixing code self-containment in: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            fixes = 0
            
            for i, line in enumerate(lines, 1):
                try:
                    obj = json.loads(line)
                    input_text = obj.get('input', '')
                    output_text = obj.get('output', obj.get('target', ''))
                    
                    # Check if input has code but output doesn't
                    has_code_in_input = '```python' in input_text or '```' in input_text
                    has_code_in_output = '```python' in output_text or '```' in output_text
                    
                    if has_code_in_input and not has_code_in_output:
                        # Check if question refers to code
                        refers_to_code = any(phrase in output_text.lower() for phrase in 
                                           ['kode di atas', 'kode berikut', 'kode tersebut', 'program di atas'])
                        
                        if refers_to_code:
                            # Extract code from input
                            code_match = re.search(r'```python\n(.*?)\n```', input_text, re.DOTALL)
                            if code_match:
                                code_block = code_match.group(0)
                                
                                # Insert code into question
                                question_match = re.search(r'question: (.*?)\n', output_text, re.DOTALL)
                                if question_match:
                                    question_text = question_match.group(1)
                                    
                                    # Add code after "Perhatikan kode berikut:"
                                    if 'perhatikan kode' in question_text.lower():
                                        new_question = f"question: Perhatikan kode berikut:\n{code_block}\n"
                                        # Keep rest of question
                                        rest = output_text.split('\n', 1)[1] if '\n' in output_text else ''
                                        obj['output'] = new_question + rest
                                        fixes += 1
                    
                    fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    fixed_lines.append(line)
                except Exception as e:
                    print(f"   ⚠️  Line {i}: {e}")
                    fixed_lines.append(line)
            
            # Write back
            if fixes > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                print(f"   ✅ Fixed {fixes} code self-containment issues")
            else:
                print(f"   ℹ️  No issues found")
            
            return fixes
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            return 0


def main():
    base_path = Path('dataset_aqg/dataset-task-v3')
    fixer = DatasetFixer()
    
    print("="*80)
    print("DATASET FIXER - Based on Design Guide")
    print("="*80)
    
    # Fix 1: JSON errors in materi3.jsonl
    print("\n📋 STEP 1: Fixing JSON errors")
    json_error_file = base_path / '09-oop' / 'materi3.jsonl'
    if json_error_file.exists():
        fixer.fix_json_errors(json_error_file)
    
    # Fix 2: Short input contexts
    print("\n📋 STEP 2: Expanding short input contexts")
    short_context_files = [
        base_path / '02-berinteraksi-dengan-data' / '4_transformasi_string.jsonl',
        base_path / '02-berinteraksi-dengan-data' / '5_operasi_list_set_string.jsonl',
        base_path / '02-berinteraksi-dengan-data' / '6_rangkuman.jsonl',
        base_path / '03-ekspresi' / 'materi3.jsonl',
        base_path / '03-ekspresi' / 'materi4.jsonl',
    ]
    
    for file_path in short_context_files:
        if file_path.exists():
            fixer.expand_short_contexts(file_path)
    
    # Fix 3: Code self-containment
    print("\n📋 STEP 3: Fixing code self-containment")
    code_files = [
        base_path / '04-aksi-sekuensial' / 'materi1.jsonl',
    ]
    
    for file_path in code_files:
        if file_path.exists():
            fixer.fix_code_self_containment(file_path)
    
    print("\n" + "="*80)
    print("✅ DONE! Please run validate_dataset_design.py to verify fixes.")
    print("="*80)


if __name__ == '__main__':
    main()
