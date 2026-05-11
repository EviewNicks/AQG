#!/usr/bin/env python3
"""
Script untuk menggabungkan batch files ke dalam file generated utama
"""
import json
import os

# Paths
base_path = "dataset_aqg/dataset-task-v4/01-perkenalan-python"
generated_file = os.path.join(base_path, "01-perkenalan-python_generated.jsonl")
batch_files = [
    os.path.join(base_path, "batch_125_145.jsonl"),
    os.path.join(base_path, "batch_146_165.jsonl"),
    os.path.join(base_path, "batch_166_185.jsonl"),
    os.path.join(base_path, "batch_186_205.jsonl"),
]

# Read all batch files and append to generated file
total_samples = 0
with open(generated_file, 'a', encoding='utf-8') as out_f:
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            print(f"Processing {batch_file}...")
            with open(batch_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + '\n')
                        total_samples += 1
            print(f"  Added {total_samples} samples so far")

print(f"\nTotal samples added: {total_samples}")
print(f"Generated file: {generated_file}")

# Verify final count
final_count = 0
with open(generated_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            final_count += 1

print(f"Final total in generated file: {final_count}")
