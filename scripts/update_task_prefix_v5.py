"""
Script to update task prefix from v4 to v5 dataset.
- Copy dataset from v4 to v5
- Update task prefix from 'buat_soal_pilihan_ganda:' to a clearer version
"""

import json
import os
from pathlib import Path

# Configuration
OLD_PREFIX = "buat_soal_pilihan_ganda:"
NEW_PREFIX = "buatlah soal pilihan ganda dengan format question, answer, dan 3 distractor dari konteks berikut:"

# Source and destination paths
SOURCE_DIR = Path("dataset_aqg/dataset-task-v4/00-dataset")
DEST_DIR = Path("dataset_aqg/dataset-task-v5/00-dataset")

# Files to process
FILES_TO_PROCESS = ["accumulated.jsonl", "train.jsonl", "test.jsonl", "validation.jsonl"]


def update_task_prefix(input_text: str) -> str:
    """Replace old task prefix with new one."""
    if input_text.startswith(OLD_PREFIX):
        return input_text.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return input_text


def process_jsonl_file(source_file: Path, dest_file: Path) -> dict:
    """Process a single JSONL file and update task prefix."""
    updated_count = 0
    total_count = 0
    
    # Ensure destination directory exists
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(source_file, 'r', encoding='utf-8') as src, \
         open(dest_file, 'w', encoding='utf-8') as dst:
        
        for line in src:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            try:
                data = json.loads(line)
                
                # Update input field
                if 'input' in data:
                    original_input = data['input']
                    updated_input = update_task_prefix(original_input)
                    
                    if original_input != updated_input:
                        data['input'] = updated_input
                        updated_count += 1
                
                # Write updated line
                dst.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {total_count} in {source_file.name}: {e}")
                continue
    
    return {
        'total': total_count,
        'updated': updated_count,
        'file': source_file.name
    }


def main():
    print("=" * 60)
    print("Updating Task Prefix: v4 -> v5")
    print("=" * 60)
    print(f"Old prefix: {OLD_PREFIX}")
    print(f"New prefix: {NEW_PREFIX}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print("=" * 60)
    
    total_processed = 0
    total_updated = 0
    
    for filename in FILES_TO_PROCESS:
        source_file = SOURCE_DIR / filename
        dest_file = DEST_DIR / filename
        
        if not source_file.exists():
            print(f"WARNING: {filename} not found in source directory")
            continue
        
        print(f"\nProcessing {filename}...")
        result = process_jsonl_file(source_file, dest_file)
        
        print(f"  - Total samples: {result['total']}")
        print(f"  - Updated: {result['updated']}")
        
        total_processed += result['total']
        total_updated += result['updated']
    
    # Copy dataset_info.json if exists
    source_info = SOURCE_DIR / "dataset_info.json"
    dest_info = DEST_DIR / "dataset_info.json"
    
    if source_info.exists():
        import shutil
        shutil.copy2(source_info, dest_info)
        print(f"\nCopied dataset_info.json")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(FILES_TO_PROCESS)}")
    print(f"Total samples processed: {total_processed}")
    print(f"Total samples updated: {total_updated}")
    print(f"Output directory: {DEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
