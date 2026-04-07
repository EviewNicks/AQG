"""
Tests untuk Validator — unit tests untuk semua aturan validasi.
"""
import pytest
from src.dataset.validator import (
    RawDataPoint,
    ValidDataPoint,
    ValidationResult,
    validate,
    validate_batch,
    validation_report,
    MIN_INPUT_TOKENS,
    MAX_INPUT_TOKENS,
)


def make_valid_dp(**overrides) -> RawDataPoint:
    """Helper: buat RawDataPoint yang valid, dengan override opsional."""
    defaults = dict(
        input="Konteks: " + ("Python adalah bahasa pemrograman. " * 10),
        target="Pertanyaan: Apa itu Python? Jawaban benar: Bahasa pemrograman. Distraktor: 1) Java 2) C++ 3) Ruby 4) Go",
        metadata={
            "difficulty": "easy",
            "question_type": "MCQ",
            "concept": "Sejarah Python",
            "misconception_tags": ["bahasa_lain"],  # non-empty sesuai Req 6.6
        },
        source="synthetic",
    )
    defaults.update(overrides)
    return RawDataPoint(**defaults)


# ── Unit tests: validate() ───────────────────────────────────────────────────

class TestValidateInput:
    def test_valid_datapoint_passes(self):
        dp = make_valid_dp()
        result = validate(dp)
        assert result.is_valid is True
        assert result.failure_reasons == []

    def test_input_too_short_fails(self):
        dp = make_valid_dp(input="Konteks: Python.")  # < 50 token
        result = validate(dp)
        assert result.is_valid is False
        assert any("terlalu pendek" in r for r in result.failure_reasons)

    def test_input_too_long_fails(self):
        long_input = "kata " * 600  # >> 600 token
        dp = make_valid_dp(input=long_input)
        result = validate(dp)
        assert result.is_valid is False
        assert any("terlalu panjang" in r for r in result.failure_reasons)

    def test_input_at_min_boundary_passes(self):
        # 50 token ≈ 50/1.3 ≈ 39 kata — pakai lebih banyak kata untuk aman
        input_text = "Python adalah bahasa pemrograman yang sangat populer. " * 8  # ~80 kata ≈ 104 token
        dp = make_valid_dp(input=input_text)
        result = validate(dp)
        length_errors = [r for r in result.failure_reasons if "terlalu pendek" in r]
        assert len(length_errors) == 0


class TestValidateTarget:
    def test_missing_pertanyaan_marker_fails(self):
        dp = make_valid_dp(
            target="Jawaban benar: X. Distraktor: 1) A 2) B 3) C 4) D"
        )
        result = validate(dp)
        assert result.is_valid is False
        assert any("Pertanyaan:" in r for r in result.failure_reasons)

    def test_missing_jawaban_marker_fails(self):
        dp = make_valid_dp(
            target="Pertanyaan: Apa itu Python? Distraktor: 1) A 2) B 3) C 4) D"
        )
        result = validate(dp)
        assert result.is_valid is False
        assert any("Jawaban benar:" in r for r in result.failure_reasons)

    def test_missing_distraktor_marker_fails(self):
        dp = make_valid_dp(
            target="Pertanyaan: Apa itu Python? Jawaban benar: Bahasa pemrograman."
        )
        result = validate(dp)
        assert result.is_valid is False
        assert any("Distraktor:" in r for r in result.failure_reasons)

    def test_target_not_string_fails(self):
        dp = make_valid_dp(target={"question": "test"})
        result = validate(dp)
        assert result.is_valid is False
        assert any("string" in r for r in result.failure_reasons)

    def test_all_markers_present_passes(self):
        dp = make_valid_dp()
        result = validate(dp)
        target_errors = [r for r in result.failure_reasons if "marker" in r]
        assert len(target_errors) == 0


class TestValidateMetadata:
    def test_missing_difficulty_fails(self):
        meta = {"question_type": "MCQ", "concept": "X", "misconception_tags": []}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("difficulty" in r for r in result.failure_reasons)

    def test_missing_question_type_fails(self):
        meta = {"difficulty": "easy", "concept": "X", "misconception_tags": []}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("question_type" in r for r in result.failure_reasons)

    def test_missing_concept_fails(self):
        meta = {"difficulty": "easy", "question_type": "MCQ", "misconception_tags": []}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("concept" in r for r in result.failure_reasons)

    def test_missing_misconception_tags_fails(self):
        meta = {"difficulty": "easy", "question_type": "MCQ", "concept": "X"}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("misconception_tags" in r for r in result.failure_reasons)

    def test_invalid_difficulty_fails(self):
        meta = {"difficulty": "super_hard", "question_type": "MCQ",
                "concept": "X", "misconception_tags": []}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("difficulty" in r for r in result.failure_reasons)

    def test_invalid_question_type_fails(self):
        meta = {"difficulty": "easy", "question_type": "Essay",
                "concept": "X", "misconception_tags": []}
        dp = make_valid_dp(metadata=meta)
        result = validate(dp)
        assert result.is_valid is False
        assert any("question_type" in r for r in result.failure_reasons)

    def test_all_valid_difficulties(self):
        for diff in ["easy", "medium", "hard"]:
            dp = make_valid_dp(metadata={
                "difficulty": diff, "question_type": "MCQ",
                "concept": "X", "misconception_tags": []
            })
            result = validate(dp)
            diff_errors = [r for r in result.failure_reasons if "difficulty" in r]
            assert len(diff_errors) == 0, f"difficulty '{diff}' seharusnya valid"

    def test_all_valid_question_types(self):
        for qt in ["MCQ", "Code Completion"]:
            dp = make_valid_dp(metadata={
                "difficulty": "easy", "question_type": qt,
                "concept": "X", "misconception_tags": []
            })
            result = validate(dp)
            qt_errors = [r for r in result.failure_reasons if "question_type" in r]
            assert len(qt_errors) == 0, f"question_type '{qt}' seharusnya valid"

    def test_metadata_not_dict_fails(self):
        dp = make_valid_dp(metadata="not a dict")
        result = validate(dp)
        assert result.is_valid is False
        assert any("dict" in r for r in result.failure_reasons)


# ── Unit tests: validate_batch() ─────────────────────────────────────────────

class TestValidateBatch:
    def test_all_valid_returns_all(self):
        dps = [make_valid_dp() for _ in range(5)]
        valid, failures = validate_batch(dps)
        assert len(valid) == 5
        assert len(failures) == 0

    def test_all_invalid_returns_none(self):
        dps = [make_valid_dp(input="short") for _ in range(3)]
        valid, failures = validate_batch(dps)
        assert len(valid) == 0
        assert len(failures) == 3

    def test_mixed_batch(self):
        dps = [make_valid_dp(), make_valid_dp(input="short"), make_valid_dp()]
        valid, failures = validate_batch(dps)
        assert len(valid) == 2
        assert len(failures) == 1

    def test_failure_log_has_reasons(self):
        dps = [make_valid_dp(input="short")]
        _, failures = validate_batch(dps)
        assert len(failures) == 1
        assert "reasons" in failures[0]
        assert len(failures[0]["reasons"]) > 0

    def test_valid_datapoints_are_valid_datapoint_type(self):
        dps = [make_valid_dp()]
        valid, _ = validate_batch(dps)
        assert isinstance(valid[0], ValidDataPoint)

    def test_empty_batch(self):
        valid, failures = validate_batch([])
        assert valid == []
        assert failures == []


# ── Unit tests: validation_report() ─────────────────────────────────────────

class TestValidationReport:
    def test_report_structure(self):
        dps = [make_valid_dp(), make_valid_dp(input="short")]
        valid, failures = validate_batch(dps)
        report = validation_report(len(dps), valid, failures)
        assert "total_processed" in report
        assert "passed" in report
        assert "failed" in report
        assert "pass_rate" in report

    def test_report_counts_correct(self):
        dps = [make_valid_dp(), make_valid_dp(), make_valid_dp(input="short")]
        valid, failures = validate_batch(dps)
        report = validation_report(len(dps), valid, failures)
        assert report["total_processed"] == 3
        assert report["passed"] == 2
        assert report["failed"] == 1

    def test_pass_rate_calculation(self):
        dps = [make_valid_dp() for _ in range(4)] + [make_valid_dp(input="short")]
        valid, failures = validate_batch(dps)
        report = validation_report(len(dps), valid, failures)
        assert report["pass_rate"] == 80.0

    def test_empty_report(self):
        report = validation_report(0, [], [])
        assert report["pass_rate"] == 0.0
