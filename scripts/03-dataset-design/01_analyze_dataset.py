"""
Dataset Analysis Script - Phase 1
Validates a single JSONL file against 03-Dataset-Design-Guide-v3.md rules
Output is saved to scripts/03-dataset-design/report-dataset.md

Usage:
    python scripts/03-dataset-design/01_analyze_dataset.py <filepath>

Example:
    python scripts/03-dataset-design/01_analyze_dataset.py dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python.jsonl
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from analyze_dataset_helpers import validate_sample, auto_detect_type

# ─── Config ───────────────────────────────────────────────────────────────────

TARGET_PER_MATERI = 220
REPORT_PATH = Path("scripts/03-dataset-design/report-dataset.md")


# ─── File Analysis ────────────────────────────────────────────────────────────

def analyze_file(filepath: Path) -> dict:
    report = {
        "filepath": str(filepath),
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "issues": [],
        "warnings": [],
        "type_dist": defaultdict(int),
        "difficulty_dist": defaultdict(int),
        "seen_inputs": set(),
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        report["issues"].append((0, [f"File read error: {e}"]))
        return report

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        report["total"] += 1

        try:
            sample = json.loads(line)
        except json.JSONDecodeError as e:
            report["invalid"] += 1
            report["issues"].append((line_num, [f"JSON decode error: {e}"]))
            continue

        input_text = sample.get("input", "")
        if input_text in report["seen_inputs"]:
            report["invalid"] += 1
            report["issues"].append((line_num, ["Duplicate input"]))
            continue
        report["seen_inputs"].add(input_text)

        is_valid, hard_fails, warns = validate_sample(sample, line_num)

        if not is_valid:
            report["invalid"] += 1
            report["issues"].append((line_num, hard_fails))
        else:
            report["valid"] += 1
            report["type_dist"][auto_detect_type(input_text)] += 1
            difficulty = sample.get("metadata", {}).get("difficulty", "Unknown")
            report["difficulty_dist"][difficulty] += 1

        if warns:
            report["warnings"].append((line_num, warns))

    return report


# ─── Markdown Report ──────────────────────────────────────────────────────────

def build_markdown(report: dict) -> str:
    filepath = Path(report["filepath"])
    total = report["total"]
    valid = report["valid"]
    invalid = report["invalid"]
    gap = TARGET_PER_MATERI - valid
    status = "✅ Target met" if gap <= 0 else f"❌ Need {gap} more samples"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# Dataset Analysis Report")
    lines.append(f"")
    lines.append(f"**File:** `{filepath}`  ")
    lines.append(f"**Date:** {now}  ")
    lines.append(f"**Target:** {TARGET_PER_MATERI} samples per materi")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Summary table
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {total} |")
    lines.append(f"| Valid | {valid} |")
    lines.append(f"| Invalid (to remove) | {invalid} |")
    lines.append(f"| Status | {status} |")
    lines.append(f"")

    # Type distribution
    knowledge = report["type_dist"].get("knowledge", 0)
    code = report["type_dist"].get("code", 0)
    total_typed = knowledge + code
    if total_typed > 0:
        k_pct = knowledge / total_typed * 100
        c_pct = code / total_typed * 100
        k_status = "✅" if k_pct >= 60 else "❌"
        c_status = "✅" if c_pct <= 40 else "❌"
        lines.append(f"## Type Distribution (auto-detected)")
        lines.append(f"")
        lines.append(f"| Type | Count | % | Status |")
        lines.append(f"|------|-------|---|--------|")
        lines.append(f"| knowledge | {knowledge} | {k_pct:.0f}% | {k_status} |")
        lines.append(f"| code | {code} | {c_pct:.0f}% | {c_status} |")
        if k_pct < 60:
            needed = int((0.6 * total_typed - knowledge) / 0.4) + 1
            lines.append(f"")
            lines.append(f"> ⚠️ Need ~{needed} more knowledge samples to reach 60% ratio")
        lines.append(f"")

    # Difficulty distribution
    if report["difficulty_dist"]:
        lines.append(f"## Difficulty Distribution")
        lines.append(f"")
        lines.append(f"| Difficulty | Count | % |")
        lines.append(f"|------------|-------|---|")
        for diff in ["Mudah", "Sedang", "Sulit"]:
            count = report["difficulty_dist"].get(diff, 0)
            pct = count / valid * 100 if valid > 0 else 0
            lines.append(f"| {diff} | {count} | {pct:.0f}% |")
        lines.append(f"")

    # Issues
    if report["issues"]:
        lines.append(f"## Issues ({len(report['issues'])} samples to remove)")
        lines.append(f"")
        lines.append(f"| Line | Reason |")
        lines.append(f"|------|--------|")
        for line_num, reasons in report["issues"]:
            for r in reasons:
                lines.append(f"| {line_num} | {r} |")
        lines.append(f"")

    # Warnings
    if report["warnings"]:
        lines.append(f"## Warnings ({len(report['warnings'])} samples)")
        lines.append(f"")
        lines.append(f"| Line | Warning |")
        lines.append(f"|------|---------|")
        for line_num, warns in report["warnings"]:
            for w in warns:
                lines.append(f"| {line_num} | {w} |")
        lines.append(f"")

    # Next steps
    lines.append(f"## Next Steps")
    lines.append(f"")
    if invalid > 0:
        lines.append(f"1. **Cleanup** — remove {invalid} invalid samples:")
        lines.append(f"   ```")
        lines.append(f"   python scripts/03-dataset-design/02_clean_dataset.py {report['filepath']}")
        lines.append(f"   ```")
    else:
        lines.append(f"1. **Cleanup** — no invalid samples, skip this step")
    if gap > 0:
        lines.append(f"2. **Generate** — create {gap} new samples to reach target of {TARGET_PER_MATERI}")
    else:
        lines.append(f"2. **Generate** — target already met ✅")
    lines.append(f"")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_analyze_dataset.py <filepath>")
        print("Example: python 01_analyze_dataset.py dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python.jsonl")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if filepath.suffix != ".jsonl":
        print(f"Error: Expected a .jsonl file, got: {filepath.suffix}")
        sys.exit(1)

    report = analyze_file(filepath)
    md = build_markdown(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")

    # Minimal terminal output
    valid = report["valid"]
    invalid = report["invalid"]
    gap = TARGET_PER_MATERI - valid
    print(f"✅ Report saved to {REPORT_PATH}")
    print(f"   Valid: {valid} | Invalid: {invalid} | Gap: {max(gap, 0)}")


if __name__ == "__main__":
    main()
