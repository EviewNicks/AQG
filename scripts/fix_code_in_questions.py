import json
import re

def fix_code_question(line_data):
    """
    Fix questions that reference 'kode di atas' by including the code block in the question.
    """
    input_text = line_data['input']
    output_text = line_data['output']
    
    # Check if output contains "kode di atas" or similar references
    if 'kode di atas' not in output_text.lower():
        return line_data  # No fix needed
    
    # Extract code block from input
    code_match = re.search(r'```python\n(.*?)\n```', input_text, re.DOTALL)
    if not code_match:
        return line_data  # No code block found
    
    code_block = code_match.group(0)  # Get full code block with backticks
    
    # Extract the question part
    question_match = re.search(r'question: (.*?)\nanswer:', output_text, re.DOTALL)
    if not question_match:
        return line_data
    
    old_question = question_match.group(1)
    
    # Create new question with code block
    if 'kode di atas' in old_question.lower():
        # Replace "kode di atas" reference with actual code
        new_question = f"Perhatikan kode berikut:\n{code_block}\n{old_question}"
        
        # Replace in output
        new_output = output_text.replace(f'question: {old_question}', f'question: {new_question}')
        line_data['output'] = new_output
    
    return line_data

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_code_in_questions.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    # Generate output filename by adding _fixed before extension
    if input_file.endswith('.jsonl'):
        output_file = input_file.replace('.jsonl', '_fixed.jsonl')
    else:
        output_file = input_file + '_fixed'
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_count += 1
            line_data = json.loads(line.strip())
            
            original_output = line_data['output']
            fixed_data = fix_code_question(line_data)
            
            if fixed_data['output'] != original_output:
                fixed_count += 1
                print(f"Line {total_count}: Fixed")
            
            f_out.write(json.dumps(fixed_data, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Done!")
    print(f"Total lines: {total_count}")
    print(f"Fixed lines: {fixed_count}")
    print(f"Output: {output_file}")

if __name__ == '__main__':
    main()
