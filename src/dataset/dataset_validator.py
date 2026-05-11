"""
Dataset Validator for IndoNanoT5 MCQ Generation
Validates JSONL dataset against design guide specifications.

Design Guide: docs/dataset/02-Dataset-Design-Guide.md
Task: Multiple Choice Question Generation (MCQ-G)
Format: JSONL with input/output fields
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: str  # "CRITICAL", "WARNING", "MINOR"
    category: str  # "FORMAT", "CONTENT", "QUALITY"
    message: str
    line_number: Optional[int] = None
    sample: Optional[str] = None


@dataclass
class ValidationResult:
    """Results from validating a dataset file"""
    filepath: str
    total_samples: int
    valid_samples: int
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if dataset has no critical issues"""
        return not any(issue.severity == "CRITICAL" for issue in self.issues)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "CRITICAL")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "WARNING")
    
    @property
    def minor_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "MINOR")


class DatasetValidator:
    """
    Validator for IndoNanoT5 MCQ Generation dataset.
    
    Design Requirements:
    - Format: JSONL (one JSON per line)
    - Fields: "input" and "output" (required)
    - Input format: "generate_mcq: [PLAIN TEXT]"
    - Output format: "question: ...\nanswer: ...\ndistractors: ... | ... | ..."
    - No markdown formatting in input (except code blocks with ```)
    """
    
    # Design guide specifications
    REQUIRED_FIELDS = ["input", "output"]
    TASK_PREFIX = "buat_soal_pilihan_ganda:"  # Indonesian prefix
    OUTPUT_MARKERS = ["question:", "answer:", "distractors:"]
    MARKDOWN_PATTERNS = [
        r'^#{1,6}\s',  # Headers: #, ##, ###
        r'\*\*[^*]+\*\*',  # Bold: **text**
        r'__[^_]+__',  # Bold: __text__
        r'\*[^*]+\*',  # Italic: *text*
        r'_[^_]+_',  # Italic: _text_
    ]
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a single JSONL file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            result = ValidationResult(
                filepath=str(filepath),
                total_samples=0,
                valid_samples=0
            )
            result.issues.append(ValidationIssue(
                severity="CRITICAL",
                category="FORMAT",
                message=f"File not found: {filepath}"
            ))
            return result
        
        # Load and validate
        samples = []
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse JSON
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                        
                        # Validate sample
                        sample_issues = self._validate_sample(sample, line_num)
                        issues.extend(sample_issues)
                        
                    except json.JSONDecodeError as e:
                        issues.append(ValidationIssue(
                            severity="CRITICAL",
                            category="FORMAT",
                            message=f"Invalid JSON: {str(e)}",
                            line_number=line_num,
                            sample=line[:100]
                        ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity="CRITICAL",
                category="FORMAT",
                message=f"Error reading file: {str(e)}"
            ))
        
        # Detect duplicates
        duplicate_issues = self._detect_duplicates(samples)
        issues.extend(duplicate_issues)
        
        # Calculate statistics
        statistics = self._calculate_statistics(samples)
        
        # Count valid samples (no critical issues per sample)
        valid_count = len(samples) - sum(
            1 for issue in issues 
            if issue.severity == "CRITICAL" and issue.line_number is not None
        )
        
        result = ValidationResult(
            filepath=str(filepath),
            total_samples=len(samples),
            valid_samples=valid_count,
            issues=issues,
            statistics=statistics
        )
        
        self.results.append(result)
        return result
    
    def validate_folder(self, folder_path: str, recursive: bool = True) -> List[ValidationResult]:
        """Validate all JSONL files in a folder"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return []
        
        # Find all JSONL files
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        jsonl_files = list(folder.glob(pattern))
        
        if not jsonl_files:
            print(f"⚠️  No JSONL files found in: {folder}")
            return []
        
        print(f"📂 Found {len(jsonl_files)} JSONL files")
        print(f"🔍 Validating...\n")
        
        results = []
        for filepath in sorted(jsonl_files):
            print(f"  Validating: {filepath.name}...", end=" ")
            result = self.validate_file(str(filepath))
            results.append(result)
            
            # Quick status
            if result.critical_count > 0:
                print(f"❌ {result.critical_count} critical issues")
            elif result.warning_count > 0:
                print(f"⚠️  {result.warning_count} warnings")
            else:
                print("✅ OK")
        
        return results
    
    def _validate_sample(self, sample: Dict, line_num: int) -> List[ValidationIssue]:
        """Validate a single data sample"""
        issues = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                issues.append(ValidationIssue(
                    severity="CRITICAL",
                    category="FORMAT",
                    message=f"Missing required field: '{field}'",
                    line_number=line_num
                ))
        
        if "input" not in sample or "output" not in sample:
            return issues  # Can't validate further
        
        input_text = sample["input"]
        output_text = sample["output"]
        
        # Validate input format
        input_issues = self._validate_input(input_text, line_num)
        issues.extend(input_issues)
        
        # Validate output format
        output_issues = self._validate_output(output_text, line_num)
        issues.extend(output_issues)
        
        return issues
    
    def _validate_input(self, input_text: str, line_num: int) -> List[ValidationIssue]:
        """Validate input field"""
        issues = []
        
        # Check task prefix
        if not input_text.startswith(self.TASK_PREFIX):
            issues.append(ValidationIssue(
                severity="CRITICAL",
                category="FORMAT",
                message=f"Input missing task prefix '{self.TASK_PREFIX}'",
                line_number=line_num,
                sample=input_text[:100]
            ))
        
        # Check for markdown noise (except code blocks)
        markdown_issues = self._detect_markdown_noise(input_text, line_num)
        issues.extend(markdown_issues)
        
        # Check length (warning if too short or too long)
        word_count = len(input_text.split())
        if word_count < 10:
            issues.append(ValidationIssue(
                severity="WARNING",
                category="QUALITY",
                message=f"Input very short ({word_count} words)",
                line_number=line_num
            ))
        elif word_count > 200:
            issues.append(ValidationIssue(
                severity="WARNING",
                category="QUALITY",
                message=f"Input very long ({word_count} words, may exceed 512 tokens)",
                line_number=line_num
            ))
        
        return issues
    
    def _validate_output(self, output_text: str, line_num: int) -> List[ValidationIssue]:
        """Validate output field"""
        issues = []
        
        # Check required markers
        for marker in self.OUTPUT_MARKERS:
            if marker not in output_text.lower():
                issues.append(ValidationIssue(
                    severity="CRITICAL",
                    category="FORMAT",
                    message=f"Output missing required marker: '{marker}'",
                    line_number=line_num,
                    sample=output_text[:100]
                ))
        
        # Check distractor format (should have | separators)
        if "distractors:" in output_text.lower():
            distractor_line = output_text.lower().split("distractors:")[-1].strip()
            pipe_count = distractor_line.count("|")
            
            if pipe_count == 0:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="FORMAT",
                    message="Distractors not separated by '|'",
                    line_number=line_num
                ))
            elif pipe_count < 2:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="QUALITY",
                    message=f"Only {pipe_count + 1} distractors (expected 3-4)",
                    line_number=line_num
                ))
        
        return issues
    
    def _detect_markdown_noise(self, text: str, line_num: int) -> List[ValidationIssue]:
        """Detect markdown formatting in text (except code blocks)"""
        issues = []
        
        # Remove code blocks before checking
        text_without_code = re.sub(r'```[\s\S]*?```', '', text)
        
        for pattern in self.MARKDOWN_PATTERNS:
            matches = re.findall(pattern, text_without_code, re.MULTILINE)
            if matches:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="CONTENT",
                    message=f"Markdown formatting detected: {matches[0][:50]}",
                    line_number=line_num
                ))
                break  # Only report once per sample
        
        return issues
    
    def _detect_duplicates(self, samples: List[Dict]) -> List[ValidationIssue]:
        """Detect duplicate input strings"""
        issues = []
        
        input_counts = Counter(
            sample.get("input", "") for sample in samples
        )
        
        duplicates = {inp: count for inp, count in input_counts.items() if count > 1}
        
        if duplicates:
            for inp, count in duplicates.items():
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="QUALITY",
                    message=f"Duplicate input found {count} times",
                    sample=inp[:100]
                ))
        
        return issues
    
    def _calculate_statistics(self, samples: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        if not samples:
            return {}
        
        input_lengths = []
        output_lengths = []
        distractor_counts = []
        
        for sample in samples:
            if "input" in sample:
                input_lengths.append(len(sample["input"].split()))
            
            if "output" in sample:
                output_lengths.append(len(sample["output"].split()))
                
                # Count distractors
                output_lower = sample["output"].lower()
                if "distractors:" in output_lower:
                    distractor_line = output_lower.split("distractors:")[-1]
                    distractor_count = distractor_line.count("|") + 1
                    distractor_counts.append(distractor_count)
        
        stats = {
            "total_samples": len(samples),
            "input_length": {
                "min": min(input_lengths) if input_lengths else 0,
                "max": max(input_lengths) if input_lengths else 0,
                "avg": sum(input_lengths) / len(input_lengths) if input_lengths else 0
            },
            "output_length": {
                "min": min(output_lengths) if output_lengths else 0,
                "max": max(output_lengths) if output_lengths else 0,
                "avg": sum(output_lengths) / len(output_lengths) if output_lengths else 0
            },
            "distractor_counts": Counter(distractor_counts) if distractor_counts else {}
        }
        
        return stats
    
    def generate_report(self, results: List[ValidationResult]) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("DATASET VALIDATION REPORT")
        report.append("IndoNanoT5 MCQ Generation - Dataset v3")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_files = len(results)
        total_samples = sum(r.total_samples for r in results)
        total_valid = sum(r.valid_samples for r in results)
        total_critical = sum(r.critical_count for r in results)
        total_warnings = sum(r.warning_count for r in results)
        total_minor = sum(r.minor_count for r in results)
        
        report.append("📊 SUMMARY")
        report.append("-" * 80)
        report.append(f"Total files validated: {total_files}")
        report.append(f"Total samples: {total_samples}")
        report.append(f"Valid samples: {total_valid} ({total_valid/total_samples*100:.1f}%)" if total_samples > 0 else "Valid samples: 0")
        report.append("")
        report.append(f"🔴 Critical issues: {total_critical}")
        report.append(f"🟡 Warnings: {total_warnings}")
        report.append(f"🔵 Minor issues: {total_minor}")
        report.append("")
        
        # Per-file results
        report.append("📁 PER-FILE RESULTS")
        report.append("-" * 80)
        
        for result in results:
            status = "✅ PASS" if result.is_valid else "❌ FAIL"
            report.append(f"\n{status} {Path(result.filepath).name}")
            report.append(f"  Samples: {result.total_samples} | Valid: {result.valid_samples}")
            report.append(f"  Issues: 🔴 {result.critical_count} | 🟡 {result.warning_count} | 🔵 {result.minor_count}")
            
            # Show critical issues
            critical_issues = [i for i in result.issues if i.severity == "CRITICAL"]
            if critical_issues:
                report.append(f"  Critical issues:")
                for issue in critical_issues[:3]:  # Show first 3
                    report.append(f"    - Line {issue.line_number}: {issue.message}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_issues_to_dict(self, results: List[ValidationResult]) -> Dict:
        """Export issues to dictionary for further analysis"""
        export = {
            "summary": {
                "total_files": len(results),
                "total_samples": sum(r.total_samples for r in results),
                "valid_samples": sum(r.valid_samples for r in results),
                "critical_count": sum(r.critical_count for r in results),
                "warning_count": sum(r.warning_count for r in results),
                "minor_count": sum(r.minor_count for r in results)
            },
            "files": []
        }
        
        for result in results:
            file_data = {
                "filepath": result.filepath,
                "total_samples": result.total_samples,
                "valid_samples": result.valid_samples,
                "is_valid": result.is_valid,
                "statistics": result.statistics,
                "issues": [
                    {
                        "severity": issue.severity,
                        "category": issue.category,
                        "message": issue.message,
                        "line_number": issue.line_number
                    }
                    for issue in result.issues
                ]
            }
            export["files"].append(file_data)
        
        return export
