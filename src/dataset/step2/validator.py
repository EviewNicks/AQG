"""
Validator: memvalidasi format, panjang, dan kelengkapan setiap data point
sebelum masuk ke dataset final.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_TYPES = {"MCQ", "Code Completion"}
REQUIRED_METADATA_FIELDS = {"difficulty", "question_type", "concept", "misconception_tags"}
TARGET_REQUIRED_MARKERS = ["Pertanyaan:", "Jawaban benar:", "Distraktor:"]

MIN_INPUT_TOKENS = 50
MAX_INPUT_TOKENS = 600


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class ValidationResult:
    is_valid: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ValidDataPoint:
    input: str
    target: str
    metadata: dict


@dataclass
class RawDataPoint:
    """Data point mentah sebelum validasi — bisa berasal dari LLM atau manual."""
    input: str
    target: str
    metadata: dict
    source: str = "synthetic"  # "synthetic" | "manual" | "augmented"


def validate(datapoint: RawDataPoint) -> ValidationResult:
    """
    Memvalidasi satu data point.
    Mengembalikan ValidationResult dengan is_valid dan daftar failure_reasons.
    """
    reasons: List[str] = []

    # 1. Cek panjang input
    input_tokens = _estimate_tokens(datapoint.input)
    if input_tokens < MIN_INPUT_TOKENS:
        reasons.append(
            f"input terlalu pendek: {input_tokens} token (min {MIN_INPUT_TOKENS})"
        )
    if input_tokens > MAX_INPUT_TOKENS:
        reasons.append(
            f"input terlalu panjang: {input_tokens} token (max {MAX_INPUT_TOKENS})"
        )

    # 2. Cek target adalah string dan mengandung marker wajib
    if not isinstance(datapoint.target, str):
        reasons.append(f"target harus string, bukan {type(datapoint.target).__name__}")
    else:
        for marker in TARGET_REQUIRED_MARKERS:
            if marker not in datapoint.target:
                reasons.append(f"target tidak mengandung marker wajib: '{marker}'")

    # 3. Cek metadata adalah dict
    if not isinstance(datapoint.metadata, dict):
        reasons.append(f"metadata harus dict, bukan {type(datapoint.metadata).__name__}")
    else:
        # 4. Cek required fields
        for field_name in REQUIRED_METADATA_FIELDS:
            if field_name not in datapoint.metadata:
                reasons.append(f"metadata tidak memiliki field wajib: '{field_name}'")

        # 5. Cek nilai enum difficulty
        difficulty = datapoint.metadata.get("difficulty")
        if difficulty is not None and difficulty not in VALID_DIFFICULTIES:
            reasons.append(
                f"metadata.difficulty tidak valid: '{difficulty}' "
                f"(harus salah satu dari {sorted(VALID_DIFFICULTIES)})"
            )

        # 6. Cek nilai enum question_type
        question_type = datapoint.metadata.get("question_type")
        if question_type is not None and question_type not in VALID_QUESTION_TYPES:
            reasons.append(
                f"metadata.question_type tidak valid: '{question_type}' "
                f"(harus salah satu dari {sorted(VALID_QUESTION_TYPES)})"
            )

        # 7. Cek misconception_tags non-empty (Req 6.6)
        misconception_tags = datapoint.metadata.get("misconception_tags")
        if misconception_tags is not None:
            if not isinstance(misconception_tags, list) or len(misconception_tags) == 0:
                reasons.append(
                    "metadata.misconception_tags harus berupa list non-empty (minimal 1 tag)"
                )

    is_valid = len(reasons) == 0

    # 8. Set validated=True jika lolos semua validasi (Req 6.7)
    if is_valid and isinstance(datapoint.metadata, dict):
        datapoint.metadata["validated"] = True

    return ValidationResult(is_valid=is_valid, failure_reasons=reasons)


def validate_batch(
    datapoints: List[RawDataPoint],
) -> Tuple[List[ValidDataPoint], List[dict]]:
    """
    Memvalidasi batch data points.
    Mengembalikan (valid_list, failure_log).

    failure_log: list of dict dengan field 'reason' dan 'raw_datapoint'.
    """
    valid_list: List[ValidDataPoint] = []
    failure_log: List[dict] = []

    for dp in datapoints:
        result = validate(dp)
        if result.is_valid:
            valid_list.append(ValidDataPoint(
                input=dp.input,
                target=dp.target,
                metadata=dp.metadata,
            ))
        else:
            failure_log.append({
                "reasons": result.failure_reasons,
                "raw_datapoint": {
                    "input": dp.input[:200],  # truncate untuk log
                    "target": dp.target[:200],
                    "metadata": dp.metadata,
                    "source": dp.source,
                },
            })

    return valid_list, failure_log


def validation_report(
    total: int,
    valid_list: List[ValidDataPoint],
    failure_log: List[dict],
) -> dict:
    """Menghasilkan ringkasan laporan validasi."""
    failure_reason_counts: dict = {}
    for entry in failure_log:
        for reason in entry["reasons"]:
            # Ambil prefix reason untuk grouping
            key = reason.split(":")[0].strip()
            failure_reason_counts[key] = failure_reason_counts.get(key, 0) + 1

    return {
        "total_processed": total,
        "passed": len(valid_list),
        "failed": len(failure_log),
        "pass_rate": round(len(valid_list) / total * 100, 1) if total > 0 else 0.0,
        "failure_reason_summary": failure_reason_counts,
    }
