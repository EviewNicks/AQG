"""
Shared helper functions for dataset analysis and cleanup scripts.
"""

import re

VALID_DIFFICULTIES = {"Mudah", "Sedang", "Sulit"}
MAX_ANSWER_WORDS = 15
MAX_INPUT_WORDS = 400
VAGUE_QUESTION_PREFIXES = [
    "bagaimana dengan",
    "apa tentang",
    "ceritakan tentang",
]


def has_code_block(text: str) -> bool:
    return "```" in text


def auto_detect_type(input_text: str) -> str:
    return "code" if has_code_block(input_text) else "knowledge"


def parse_output(output_text: str) -> dict:
    result = {"question": None, "answer": None, "distractors": None}
    lines = output_text.strip().split("\n")
    current_key = None
    current_value = []

    for line in lines:
        if line.startswith("question:"):
            if current_key:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "question"
            current_value = [line[len("question:"):].strip()]
        elif line.startswith("answer:"):
            if current_key:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "answer"
            current_value = [line[len("answer:"):].strip()]
        elif line.startswith("distractors:"):
            if current_key:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "distractors"
            current_value = [line[len("distractors:"):].strip()]
        else:
            if current_key:
                current_value.append(line)

    if current_key:
        result[current_key] = "\n".join(current_value).strip()

    return result


def has_duplicate_text(text: str) -> bool:
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if lines[i].strip() and lines[i].strip() == lines[i + 1].strip():
            return True
    if re.search(r'(Perhatikan kode berikut[:\s]*\n+Perhatikan kode berikut)', text, re.IGNORECASE):
        return True
    return False


def count_distractors(distractors_text: str) -> int:
    if not distractors_text:
        return 0
    parts = [p.strip() for p in distractors_text.split("|")]
    return len([p for p in parts if p])


def is_vague_question(question_text: str) -> bool:
    if not question_text:
        return False
    q_lower = question_text.lower().strip()
    return any(q_lower.startswith(prefix) for prefix in VAGUE_QUESTION_PREFIXES)


def validate_sample(sample: dict, line_num: int) -> tuple:
    """
    Validate a single sample.
    Returns: (is_valid, hard_fail_reasons, warnings)
    """
    hard_fails = []
    warnings = []

    if "input" not in sample:
        hard_fails.append("Missing field: input")
    if "output" not in sample:
        hard_fails.append("Missing field: output")
    if "metadata" not in sample:
        hard_fails.append("Missing field: metadata")

    if hard_fails:
        return False, hard_fails, warnings

    input_text = sample.get("input", "")
    output_text = sample.get("output", "")
    metadata = sample.get("metadata", {})

    if not input_text.startswith("buat_soal_pilihan_ganda:"):
        hard_fails.append("Input missing task prefix 'buat_soal_pilihan_ganda:'")

    difficulty = metadata.get("difficulty")
    if not difficulty:
        hard_fails.append("Missing metadata.difficulty")
    elif difficulty not in VALID_DIFFICULTIES:
        hard_fails.append(f"Invalid metadata.difficulty: '{difficulty}'")

    if "question:" not in output_text:
        hard_fails.append("Output missing 'question:' field")
    if "answer:" not in output_text:
        hard_fails.append("Output missing 'answer:' field")
    if "distractors:" not in output_text:
        hard_fails.append("Output missing 'distractors:' field")

    parsed = parse_output(output_text)

    distractor_count = count_distractors(parsed.get("distractors", ""))
    if distractor_count != 3:
        hard_fails.append(f"Distractors must have exactly 3 options (found {distractor_count})")

    question_text = parsed.get("question", "")
    if has_duplicate_text(question_text):
        hard_fails.append("Duplicate text found in question field")

    if has_duplicate_text(output_text):
        hard_fails.append("Duplicate text found in output field")

    # Warnings
    input_words = len(input_text.split())
    if input_words > MAX_INPUT_WORDS:
        warnings.append(f"Input length {input_words} words (exceeds {MAX_INPUT_WORDS} word limit)")

    answer_text = parsed.get("answer", "")
    answer_words = len(answer_text.split()) if answer_text else 0
    if answer_words > MAX_ANSWER_WORDS:
        warnings.append(f"Answer is long ({answer_words} words)")

    if is_vague_question(question_text):
        warnings.append(f"Question may be vague: '{question_text[:60]}'")

    is_valid = len(hard_fails) == 0
    return is_valid, hard_fails, warnings
