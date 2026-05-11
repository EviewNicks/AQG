#!/usr/bin/env python3
"""
Script untuk merge dan split dataset TANPA code blocks dari dataset-task-v4

Fitur:
- Merge semua file JSONL dari semua section (kecuali 00-dataset*)
- FILTER: Hanya ambil data yang TIDAK mengandung code blocks (```python```)
- Validasi format data (input, output, metadata)
- Deduplikasi berdasarkan input string
- Split 80/10/10 (train/validation/test) secara random
- Generate dataset_info.json dengan statistik lengkap
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random


def has_code_block(text: str) -> bool:
    """
    Cek apakah text mengandung code block (```python atau ```)
    
    Returns:
        True jika ada code block, False jika tidak
    """
    # Pattern untuk code block markdown
    code_block_pattern = r'```(?:python)?'
    return bool(re.search(code_block_pattern, text))


def validate_datapoint(data: dict, line_num: int, file_path: str) -> Tuple[bool, List[str]]:
    """
    Validasi format satu data point
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Cek field wajib
    if "input" not in data:
        errors.append(f"Missing 'input' field")
    elif not isinstance(data["input"], str):
        errors.append(f"'input' must be string, got {type(data['input'])}")
    
    if "output" not in data:
        errors.append(f"Missing 'output' field")
    elif not isinstance(data["output"], str):
        errors.append(f"'output' must be string, got {type(data['output'])}")
    
    if "metadata" not in data:
        errors.append(f"Missing 'metadata' field")
    elif not isinstance(data["metadata"], dict):
        errors.append(f"'metadata' must be dict, got {type(data['metadata'])}")
    else:
        # Validasi metadata
        metadata = data["metadata"]
        if "difficulty" not in metadata:
            errors.append(f"Missing 'difficulty' in metadata")
        elif metadata["difficulty"] not in ["Mudah", "Sedang", "Sulit"]:
            errors.append(f"Invalid difficulty: {metadata['difficulty']}")
    
    if errors:
        return False, [f"Line {line_num} in {file_path}: {err}" for err in errors]
    
    return True, []


def load_jsonl_files(base_dir: str) -> Tuple[List[dict], List[dict], List[str], Dict[str, int]]:
    """
    Load semua file JSONL dari semua section (kecuali 00-dataset*)
    Filter: hanya ambil data TANPA code blocks
    
    Returns:
        (data_no_code, data_with_code, validation_errors, file_stats)
    """
    base_path = Path(base_dir)
    data_no_code = []
    data_with_code = []
    validation_errors = []
    file_stats = {}
    
    # Iterasi semua folder section
    for section_dir in sorted(base_path.iterdir()):
        if not section_dir.is_dir():
            continue
        
        # Skip folder 00-dataset dan 00-dataset-no-code
        if section_dir.name.startswith("00-dataset"):
            continue
        
        print(f"\n📂 Processing section: {section_dir.name}")
        section_no_code = 0
        section_with_code = 0
        
        # Iterasi semua file JSONL di section
        for jsonl_file in sorted(section_dir.glob("*.jsonl")):
            print(f"  📄 Reading: {jsonl_file.name}")
            file_no_code = 0
            file_with_code = 0
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # Validasi format
                            is_valid, errors = validate_datapoint(data, line_num, str(jsonl_file))
                            if not is_valid:
                                validation_errors.extend(errors)
                                continue
                            
                            # Tambahkan source info ke metadata
                            if "metadata" not in data:
                                data["metadata"] = {}
                            data["metadata"]["source_section"] = section_dir.name
                            data["metadata"]["source_file"] = jsonl_file.name
                            
                            # Cek apakah ada code block di input atau output
                            has_code_in_input = has_code_block(data["input"])
                            has_code_in_output = has_code_block(data["output"])
                            
                            if has_code_in_input or has_code_in_output:
                                # Data dengan code block
                                data_with_code.append(data)
                                file_with_code += 1
                                section_with_code += 1
                            else:
                                # Data tanpa code block (explain-only)
                                data_no_code.append(data)
                                file_no_code += 1
                                section_no_code += 1
                            
                        except json.JSONDecodeError as e:
                            validation_errors.append(
                                f"Line {line_num} in {jsonl_file}: JSON decode error - {e}"
                            )
            
            except Exception as e:
                validation_errors.append(f"Error reading {jsonl_file}: {e}")
            
            file_stats[str(jsonl_file.relative_to(base_path))] = {
                "no_code": file_no_code,
                "with_code": file_with_code,
                "total": file_no_code + file_with_code
            }
            print(f"    ✓ No code: {file_no_code} | With code: {file_with_code}")
        
        print(f"  ✓ Section - No code: {section_no_code} | With code: {section_with_code}")
    
    return data_no_code, data_with_code, validation_errors, file_stats


def deduplicate_data(data_list: List[dict]) -> Tuple[List[dict], int]:
    """
    Deduplikasi berdasarkan input string
    
    Returns:
        (unique_data, duplicate_count)
    """
    seen_inputs = set()
    unique_data = []
    duplicate_count = 0
    
    for data in data_list:
        input_str = data["input"]
        if input_str not in seen_inputs:
            seen_inputs.add(input_str)
            unique_data.append(data)
        else:
            duplicate_count += 1
    
    return unique_data, duplicate_count


def split_dataset(data_list: List[dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset secara random dengan ratio 80/10/10
    
    Returns:
        (train_data, val_data, test_data)
    """
    # Set random seed untuk reproducibility
    random.seed(seed)
    
    # Shuffle data
    shuffled_data = data_list.copy()
    random.shuffle(shuffled_data)
    
    # Hitung ukuran split
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data


def save_jsonl(data_list: List[dict], output_path: str):
    """Save data ke file JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def generate_statistics(train_data, val_data, test_data, all_data) -> dict:
    """Generate statistik dataset"""
    
    def get_difficulty_dist(data_list):
        return dict(Counter(d["metadata"]["difficulty"] for d in data_list))
    
    def get_type_dist(data_list):
        return dict(Counter(d["metadata"].get("type", "unknown") for d in data_list))
    
    def get_section_dist(data_list):
        return dict(Counter(d["metadata"].get("source_section", "unknown") for d in data_list))
    
    stats = {
        "total": len(all_data),
        "splits": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data)
        },
        "difficulty_distribution": {
            "overall": get_difficulty_dist(all_data),
            "train": get_difficulty_dist(train_data),
            "validation": get_difficulty_dist(val_data),
            "test": get_difficulty_dist(test_data)
        },
        "type_distribution": {
            "overall": get_type_dist(all_data),
            "train": get_type_dist(train_data),
            "validation": get_type_dist(val_data),
            "test": get_type_dist(test_data)
        },
        "section_distribution": {
            "overall": get_section_dist(all_data),
            "train": get_section_dist(train_data),
            "validation": get_section_dist(val_data),
            "test": get_section_dist(test_data)
        }
    }
    
    return stats


def main():
    # Path konfigurasi
    base_dir = "dataset_aqg/dataset-task-v4"
    output_dir = "dataset_aqg/dataset-task-v4/00-dataset-no-code"
    
    print("=" * 70)
    print("🚀 MERGE DATASET V4 (NO CODE) - AQG Dataset Pipeline")
    print("=" * 70)
    print("📌 Filter: Hanya data TANPA code blocks (explain-only)")
    print("=" * 70)
    
    # Buat output directory jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load semua file JSONL dan filter
    print("\n📥 Step 1: Loading and filtering JSONL files...")
    data_no_code, data_with_code, validation_errors, file_stats = load_jsonl_files(base_dir)
    
    total_loaded = len(data_no_code) + len(data_with_code)
    print(f"\n✓ Total data points loaded: {total_loaded}")
    print(f"  📝 Explain-only (no code): {len(data_no_code)}")
    print(f"  💻 With code blocks: {len(data_with_code)}")
    print(f"  📊 Filter rate: {len(data_with_code) / total_loaded * 100:.2f}% filtered out")
    
    # Tampilkan validation errors jika ada
    if validation_errors:
        print(f"\n⚠️  Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:10]:  # Tampilkan max 10 error
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors")
        
        # Simpan semua errors ke file
        error_file = os.path.join(output_dir, "validation_errors.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            for error in validation_errors:
                f.write(error + '\n')
        print(f"\n  ✓ All errors saved to: {error_file}")
    
    # 2. Deduplikasi
    print("\n🔍 Step 2: Deduplicating data (no code only)...")
    unique_data, duplicate_count = deduplicate_data(data_no_code)
    
    print(f"✓ Unique data points: {len(unique_data)}")
    print(f"✓ Duplicates removed: {duplicate_count}")
    if duplicate_count > 0:
        print(f"  📊 Deduplication rate: {duplicate_count / len(data_no_code) * 100:.2f}%")
    
    # 3. Save accumulated data
    print("\n💾 Step 3: Saving accumulated dataset (no code)...")
    accumulated_path = os.path.join(output_dir, "accumulated.jsonl")
    save_jsonl(unique_data, accumulated_path)
    print(f"✓ Saved to: {accumulated_path}")
    
    # 4. Split dataset
    print("\n✂️  Step 4: Splitting dataset (80/10/10)...")
    train_data, val_data, test_data = split_dataset(unique_data)
    
    print(f"✓ Train: {len(train_data)} ({len(train_data)/len(unique_data)*100:.1f}%)")
    print(f"✓ Validation: {len(val_data)} ({len(val_data)/len(unique_data)*100:.1f}%)")
    print(f"✓ Test: {len(test_data)} ({len(test_data)/len(unique_data)*100:.1f}%)")
    
    # 5. Save splits
    print("\n💾 Step 5: Saving split datasets...")
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(output_dir, "validation.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    print("✓ All splits saved")
    
    # 6. Generate statistics
    print("\n📊 Step 6: Generating statistics...")
    stats = generate_statistics(train_data, val_data, test_data, unique_data)
    
    # Tambahkan info tambahan
    stats["filter_info"] = {
        "total_loaded": total_loaded,
        "no_code_count": len(data_no_code),
        "with_code_count": len(data_with_code),
        "filter_rate_percent": round(len(data_with_code) / total_loaded * 100, 2)
    }
    stats["deduplication"] = {
        "original_count": len(data_no_code),
        "unique_count": len(unique_data),
        "duplicates_removed": duplicate_count
    }
    stats["validation_errors"] = len(validation_errors)
    stats["file_stats"] = file_stats
    
    # Save statistics
    stats_path = os.path.join(output_dir, "dataset_info.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Statistics saved to: {stats_path}")
    
    # 7. Print summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"Total data points (original): {total_loaded}")
    print(f"  - Explain-only (no code): {len(data_no_code)}")
    print(f"  - With code blocks: {len(data_with_code)} (filtered out)")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Unique data points (no code): {len(unique_data)}")
    print(f"Validation errors: {len(validation_errors)}")
    print(f"\nSplit distribution:")
    print(f"  - Train: {len(train_data)} ({len(train_data)/len(unique_data)*100:.1f}%)")
    print(f"  - Validation: {len(val_data)} ({len(val_data)/len(unique_data)*100:.1f}%)")
    print(f"  - Test: {len(test_data)} ({len(test_data)/len(unique_data)*100:.1f}%)")
    
    print(f"\nDifficulty distribution (overall):")
    for difficulty, count in stats["difficulty_distribution"]["overall"].items():
        print(f"  - {difficulty}: {count} ({count/len(unique_data)*100:.1f}%)")
    
    print(f"\nType distribution (overall):")
    for type_name, count in stats["type_distribution"]["overall"].items():
        print(f"  - {type_name}: {count} ({count/len(unique_data)*100:.1f}%)")
    
    print(f"\nSection distribution (overall):")
    for section, count in stats["section_distribution"]["overall"].items():
        print(f"  - {section}: {count} ({count/len(unique_data)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ DONE! Dataset (no code) ready for fine-tuning.")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {accumulated_path}")
    print(f"  - {os.path.join(output_dir, 'train.jsonl')}")
    print(f"  - {os.path.join(output_dir, 'validation.jsonl')}")
    print(f"  - {os.path.join(output_dir, 'test.jsonl')}")
    print(f"  - {stats_path}")
    if validation_errors:
        print(f"  - {error_file}")


if __name__ == "__main__":
    main()
