"""
Script untuk menganalisis dataset v3 (full dan no-code)
"""
import json
from collections import Counter
import re

def analyze_dataset(file_path):
    """Analisis dataset JSONL"""
    data = []
    
    # Baca file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Statistik dasar
    total = len(data)
    
    # Analisis metadata
    difficulties = Counter()
    has_code = 0
    no_code = 0
    
    # Analisis panjang teks
    input_lengths = []
    output_lengths = []
    question_lengths = []
    
    for item in data:
        # Metadata
        if 'metadata' in item and 'difficulty' in item['metadata']:
            difficulties[item['metadata']['difficulty']] += 1
        
        # Deteksi code block
        input_text = item['input']
        if '```' in input_text or 'print(' in input_text or 'def ' in input_text:
            has_code += 1
        else:
            no_code += 1
        
        # Panjang teks
        input_lengths.append(len(input_text.split()))
        output_lengths.append(len(item['output'].split()))
        
        # Extract question
        if 'question:' in item['output']:
            question = item['output'].split('question:')[1].split('\\n')[0].strip()
            question_lengths.append(len(question.split()))
    
    # Hitung rata-rata
    avg_input = sum(input_lengths) / len(input_lengths) if input_lengths else 0
    avg_output = sum(output_lengths) / len(output_lengths) if output_lengths else 0
    avg_question = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    
    return {
        'total': total,
        'difficulties': dict(difficulties),
        'has_code': has_code,
        'no_code': no_code,
        'avg_input_words': round(avg_input, 2),
        'avg_output_words': round(avg_output, 2),
        'avg_question_words': round(avg_question, 2),
        'min_input_words': min(input_lengths) if input_lengths else 0,
        'max_input_words': max(input_lengths) if input_lengths else 0,
    }

def get_sample(file_path, has_code=True):
    """Ambil 1 contoh dari dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            input_text = item['input']
            
            # Cek apakah ada code
            has_code_block = '```' in input_text or 'print(' in input_text or 'def ' in input_text
            
            if has_code and has_code_block:
                return item
            elif not has_code and not has_code_block:
                return item
    return None

if __name__ == '__main__':
    print("="*80)
    print("ANALISIS DATASET v3")
    print("="*80)
    
    # Analisis dataset full
    print("\n1. DATASET FULL (Knowledge + Code)")
    print("-" * 80)
    full_path = 'dataset_aqg/dataset-task-v3/00-dataset/accumulated.jsonl'
    full_stats = analyze_dataset(full_path)
    
    print(f"Total entri: {full_stats['total']}")
    print(f"\nDistribusi Kesulitan:")
    for diff, count in full_stats['difficulties'].items():
        pct = (count / full_stats['total']) * 100
        print(f"  - {diff}: {count} ({pct:.1f}%)")
    
    print(f"\nDeteksi Tipe Konten:")
    print(f"  - Dengan code: {full_stats['has_code']} ({(full_stats['has_code']/full_stats['total'])*100:.1f}%)")
    print(f"  - Tanpa code: {full_stats['no_code']} ({(full_stats['no_code']/full_stats['total'])*100:.1f}%)")
    
    print(f"\nStatistik Panjang Teks (dalam kata):")
    print(f"  - Rata-rata input: {full_stats['avg_input_words']} kata")
    print(f"  - Rata-rata output: {full_stats['avg_output_words']} kata")
    print(f"  - Rata-rata pertanyaan: {full_stats['avg_question_words']} kata")
    print(f"  - Range input: {full_stats['min_input_words']} - {full_stats['max_input_words']} kata")
    
    # Analisis dataset no-code
    print("\n\n2. DATASET NO-CODE")
    print("-" * 80)
    nocode_path = 'dataset_aqg/dataset-task-v3/00-dataset-no-code/accumulated.jsonl'
    nocode_stats = analyze_dataset(nocode_path)
    
    print(f"Total entri: {nocode_stats['total']}")
    print(f"\nDistribusi Kesulitan:")
    for diff, count in nocode_stats['difficulties'].items():
        pct = (count / nocode_stats['total']) * 100
        print(f"  - {diff}: {count} ({pct:.1f}%)")
    
    print(f"\nDeteksi Tipe Konten:")
    print(f"  - Dengan code: {nocode_stats['has_code']} ({(nocode_stats['has_code']/nocode_stats['total'])*100:.1f}%)")
    print(f"  - Tanpa code: {nocode_stats['no_code']} ({(nocode_stats['no_code']/nocode_stats['total'])*100:.1f}%)")
    
    print(f"\nStatistik Panjang Teks (dalam kata):")
    print(f"  - Rata-rata input: {nocode_stats['avg_input_words']} kata")
    print(f"  - Rata-rata output: {nocode_stats['avg_output_words']} kata")
    print(f"  - Rata-rata pertanyaan: {nocode_stats['avg_question_words']} kata")
    print(f"  - Range input: {nocode_stats['min_input_words']} - {nocode_stats['max_input_words']} kata")
    
    # Contoh dataset
    print("\n\n3. CONTOH DATASET")
    print("="*80)
    
    print("\n3.1. Contoh dari Dataset FULL (dengan code)")
    print("-" * 80)
    sample_code = get_sample(full_path, has_code=True)
    if sample_code:
        print(f"Input: {sample_code['input'][:200]}...")
        print(f"\nOutput: {sample_code['output'][:200]}...")
        print(f"\nMetadata: {sample_code['metadata']}")
    
    print("\n\n3.2. Contoh dari Dataset NO-CODE")
    print("-" * 80)
    sample_nocode = get_sample(nocode_path, has_code=False)
    if sample_nocode:
        print(f"Input: {sample_nocode['input'][:200]}...")
        print(f"\nOutput: {sample_nocode['output'][:200]}...")
        print(f"\nMetadata: {sample_nocode['metadata']}")
    
    print("\n" + "="*80)
    print("RINGKASAN")
    print("="*80)
    print(f"Dataset FULL: {full_stats['total']} entri")
    print(f"Dataset NO-CODE: {nocode_stats['total']} entri")
    print(f"Selisih (code-only): {full_stats['total'] - nocode_stats['total']} entri")
    print(f"\nPersentase code dalam dataset FULL: {(full_stats['has_code']/full_stats['total'])*100:.1f}%")
    print(f"Persentase no-code dalam dataset FULL: {(full_stats['no_code']/full_stats['total'])*100:.1f}%")
