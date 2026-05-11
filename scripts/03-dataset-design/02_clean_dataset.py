"""
Dataset Cleanup Script - Phase 2
Removes invalid samples and adds metadata.type to a single JSONL file

Usage:
    python scripts/03-dataset-design/02_clean_dataset.py <filepath>
    python scripts/03-dataset-design/02_clean_dataset.py <filepath> --dry-run

Example:
    python scripts/03-dataset-design/02_clean_dataset.py dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python.jsonl
"""

import json
import shutil
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_dataset_helpers import validate_sample, auto_detect_type

# ─── Config ───────────────────────────────────────────────────────────────────

BACKUP_SUFFIX = ".bak"
TARGET_PER_MATERI = 220


# ─── Clean ────────────────────────────────────────────────────────────────────

def clean_file(filepath: Path, dry_run: bool = False) -> dict:
    """
    Clean a single JSONL file:
    - Remove invalid samples
    - Add metadata.type (auto-detected)
    Returns cleanup stats.
    """
    stats = {
        "original_count": 0,
        "kept": 0,
        "removed": 0,
        "type_added": 0,
        "removed_reasons": [],
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return stats

    cleaned_samples = []
    seen_inputs = set()

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        stats["original_count"] += 1

        # JSON parse
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as e:
            stats["removed"] += 1
            stats["removed_reasons"].append((line_num, f"JSON decode error: {e}"))
            continue

        # Duplicate input check
        input_text = sample.get("input", "")
        if input_text in seen_inputs:
            stats["removed"] += 1
            stats["removed_reasons"].append((line_num, "Duplicate input"))
            continue
        seen_inputs.add(input_text)

        # Validate
        is_valid, hard_fails, _ = validate_sample(sample, line_num)
        if not is_valid:
            stats["removed"] += 1
            stats["removed_reasons"].append((line_num, "; ".join(hard_fails)))
            continue

        # Add/update metadata.type
        if "metadata" not in sample:
            sample["metadata"] = {}

        current_type = sample["metadata"].get("type")
        if current_type not in {"knowledge", "code"}:
            sample["metadata"]["type"] = auto_detect_type(input_text)
            stats["type_added"] += 1

        cleaned_samples.append(sample)
        stats["kept"] += 1

    # Write cleaned file
    if not dry_run:
        # Backup original
        backup_path = filepath.with_suffix(filepath.suffix + BACKUP_SUFFIX)
        shutil.copy2(filepath, backup_path)

        # Write cleaned
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in cleaned_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean a single JSONL dataset file")
    parser.add_argument("filepath", help="Path to the .jsonl file to clean")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    filepath = Path(args.filepath)

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if filepath.suffix != ".jsonl":
        print(f"Error: Expected a .jsonl file, got: {filepath.suffix}")
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "LIVE"

    print(f"\n{'═'*60}")
    print(f"  DATASET CLEANUP [{mode}]")
    print(f"  {filepath.name}")
    print(f"{'═'*60}")

    stats = clean_file(filepath, dry_run=args.dry_run)

    gap = TARGET_PER_MATERI - stats["kept"]

    print(f"\n  Original:    {stats['original_count']:>4}")
    print(f"  Kept:        {stats['kept']:>4}")
    print(f"  Removed:     {stats['removed']:>4}")
    print(f"  Type added:  {stats['type_added']:>4}")

    if stats["removed_reasons"]:
        print(f"\n  Removed samples:")
        for line_num, reason in stats["removed_reasons"]:
            print(f"    Line {line_num:>4}: {reason}")

    print(f"\n{'─'*60}")
    if args.dry_run:
        print(f"  ⚠️  DRY RUN — no files modified")
        print(f"  Run without --dry-run to apply changes")
    else:
        backup = filepath.with_suffix(filepath.suffix + ".bak")
        print(f"  ✅ Cleanup done. Backup saved to: {backup.name}")

    print(f"\n  NEXT STEP:")
    if gap > 0:
        print(f"  Generate {gap} new samples to reach target of {TARGET_PER_MATERI}")
        print(f"  File: {filepath}")
    else:
        print(f"  ✅ Target of {TARGET_PER_MATERI} already met!")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
