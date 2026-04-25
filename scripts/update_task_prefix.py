#!/usr/bin/env python3
"""
Script untuk mengupdate task prefix dari 'buat_soal_pilihan_ganda' ke 'generate_mcq'
dalam dataset JSONL.

Usage:
    python scripts/update_task_prefix.py

Author: Dataset Update Script
Date: 2025
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


class DatasetPrefixUpdater:
    """Update task prefix dalam dataset JSONL files."""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        old_prefix: str = "buat_soal_pilihan_ganda",
        new_prefix: str = "generate_mcq"
    ):
        """
        Initialize updater.
        
        Args:
            source_dir: Source directory dengan file JSONL
            output_dir: Output directory untuk hasil update
            old_prefix: Prefix lama yang akan diganti
            new_prefix: Prefix baru
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_lines': 0,
            'lines_updated': 0,
            'lines_unchanged': 0,
            'errors': []
        }
    
    def update_line(self, line: str) -> Tuple[str, bool]:
        """
        Update single line dari JSONL file.
        
        Args:
            line: JSON line string
            
        Returns:
            Tuple of (updated_line, was_changed)
        """
        try:
            # Parse JSON
            data = json.loads(line.strip())
            
            # Check if 'input' field exists dan contains old prefix
            if 'input' in data and data['input'].startswith(self.old_prefix):
                # Replace prefix
                old_input = data['input']
                data['input'] = data['input'].replace(
                    self.old_prefix, 
                    self.new_prefix, 
                    1  # Only replace first occurrence
                )
                
                # Convert back to JSON string
                updated_line = json.dumps(data, ensure_ascii=False)
                return updated_line, True
            else:
                # No change needed
                return line.strip(), False
                
        except json.JSONDecodeError as e:
            self.stats['errors'].append(f"JSON decode error: {e}")
            return line.strip(), False
        except Exception as e:
            self.stats['errors'].append(f"Unexpected error: {e}")
            return line.strip(), False
    
    def process_file(self, input_file: Path, output_file: Path) -> Dict:
        """
        Process single JSONL file.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Dict dengan statistics untuk file ini
        """
        file_stats = {
            'filename': input_file.name,
            'total_lines': 0,
            'updated_lines': 0,
            'unchanged_lines': 0
        }
        
        print(f"\n  Processing: {input_file.name}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:
                
                for line_num, line in enumerate(f_in, 1):
                    if line.strip():  # Skip empty lines
                        updated_line, was_changed = self.update_line(line)
                        f_out.write(updated_line + '\n')
                        
                        file_stats['total_lines'] += 1
                        if was_changed:
                            file_stats['updated_lines'] += 1
                        else:
                            file_stats['unchanged_lines'] += 1
            
            print(f"    ✓ Lines: {file_stats['total_lines']} | "
                  f"Updated: {file_stats['updated_lines']} | "
                  f"Unchanged: {file_stats['unchanged_lines']}")
            
        except FileNotFoundError:
            error_msg = f"File not found: {input_file}"
            self.stats['errors'].append(error_msg)
            print(f"    ✗ {error_msg}")
        except Exception as e:
            error_msg = f"Error processing {input_file.name}: {e}"
            self.stats['errors'].append(error_msg)
            print(f"    ✗ {error_msg}")
        
        return file_stats
    
    def copy_non_jsonl_files(self):
        """Copy non-JSONL files (like dataset_info.json) to output directory."""
        print("\n  Copying non-JSONL files...")
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith('.jsonl'):
                output_path = self.output_dir / file_path.name
                shutil.copy2(file_path, output_path)
                print(f"    ✓ Copied: {file_path.name}")
    
    def run(self, target_files: List[str] = None):
        """
        Run update process.
        
        Args:
            target_files: List of specific files to process (None = all .jsonl files)
        """
        print("=" * 70)
        print("DATASET PREFIX UPDATER")
        print("=" * 70)
        print(f"\nSource Directory: {self.source_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Old Prefix: '{self.old_prefix}'")
        print(f"New Prefix: '{self.new_prefix}'")
        print("-" * 70)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output directory created: {self.output_dir}")
        
        # Determine files to process
        if target_files:
            files_to_process = [self.source_dir / f for f in target_files]
        else:
            files_to_process = list(self.source_dir.glob('*.jsonl'))
        
        if not files_to_process:
            print("\n✗ No JSONL files found to process!")
            return
        
        print(f"\nFiles to process: {len(files_to_process)}")
        
        # Process each file
        for input_file in files_to_process:
            if not input_file.exists():
                print(f"\n  ✗ File not found: {input_file.name}")
                continue
            
            output_file = self.output_dir / input_file.name
            file_stats = self.process_file(input_file, output_file)
            
            # Update global stats
            self.stats['files_processed'] += 1
            self.stats['total_lines'] += file_stats['total_lines']
            self.stats['lines_updated'] += file_stats['updated_lines']
            self.stats['lines_unchanged'] += file_stats['unchanged_lines']
        
        # Copy non-JSONL files
        self.copy_non_jsonl_files()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("UPDATE SUMMARY")
        print("=" * 70)
        
        print(f"\nFiles Processed: {self.stats['files_processed']}")
        print(f"Total Lines: {self.stats['total_lines']}")
        print(f"Lines Updated: {self.stats['lines_updated']}")
        print(f"Lines Unchanged: {self.stats['lines_unchanged']}")
        
        if self.stats['lines_updated'] > 0:
            update_pct = (self.stats['lines_updated'] / self.stats['total_lines']) * 100
            print(f"Update Rate: {update_pct:.2f}%")
        
        if self.stats['errors']:
            print(f"\n⚠ Errors Encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")
        else:
            print("\n✓ No errors encountered")
        
        print("\n" + "=" * 70)
        print(f"✓ Updated files saved to: {self.output_dir}")
        print("=" * 70)


def main():
    """Main execution function."""
    
    # Configuration
    SOURCE_DIR = "dataset_aqg/dataset-task-v3/00-dataset-no-code"
    OUTPUT_DIR = "dataset_aqg/dataset-task-v3/00-dataset-no-code-updated"
    
    # Target files (accumulated, train, validation, test)
    TARGET_FILES = [
        "accumulated.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl"
    ]
    
    # Create updater instance
    updater = DatasetPrefixUpdater(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        old_prefix="buat_soal_pilihan_ganda",
        new_prefix="generate_mcq"
    )
    
    # Run update
    updater.run(target_files=TARGET_FILES)
    
    print("\n✓ Update process completed!")
    print(f"\nNext steps:")
    print(f"  1. Review updated files in: {OUTPUT_DIR}")
    print(f"  2. Verify changes are correct")
    print(f"  3. If satisfied, you can replace original files or use updated version")


if __name__ == "__main__":
    main()
