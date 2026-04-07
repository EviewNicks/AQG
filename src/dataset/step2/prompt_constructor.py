"""
Prompt Constructor: membangun string input dari Chunk + TaskParams.
Format input konsisten untuk fine-tuning IndoT5.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List, Optional

from src.dataset.step2.chunker import Chunk

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_TYPES = {"MCQ", "Code Completion"}

PROMPT_TEMPLATE = (
    "Konteks: {context}\n\n"
    "Prompt: Buat satu soal {question_type} tentang {concept}, "
    "tingkat kesulitan: {difficulty}, bahasa Indonesia."
)


@dataclass
class TaskParams:
    concept: str        # konsep yang diuji, dari Master Concept List
    difficulty: str     # "easy" | "medium" | "hard"
    question_type: str  # "MCQ" | "Code Completion"

    def __post_init__(self) -> None:
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty harus salah satu dari {VALID_DIFFICULTIES}, "
                f"bukan '{self.difficulty}'"
            )
        if self.question_type not in VALID_QUESTION_TYPES:
            raise ValueError(
                f"question_type harus salah satu dari {VALID_QUESTION_TYPES}, "
                f"bukan '{self.question_type}'"
            )


@dataclass
class PromptInput:
    input: str          # string input lengkap untuk model
    chunk: Chunk        # chunk asal
    params: TaskParams  # parameter tugas


def build_prompt(chunk: Chunk, params: TaskParams) -> PromptInput:
    """
    Membangun satu PromptInput dari chunk dan parameter.
    Code block dalam context dipertahankan apa adanya.
    source_file path dinormalisasi ke forward slash.
    """
    # Normalisasi source_file ke forward slash (Req 4.5)
    normalized_source = str(PurePosixPath(chunk.source_file.replace("\\", "/")))

    input_str = PROMPT_TEMPLATE.format(
        context=chunk.text,
        question_type=params.question_type,
        concept=params.concept,
        difficulty=params.difficulty,
    )
    # Buat chunk baru dengan source_file yang sudah dinormalisasi
    normalized_chunk = Chunk(
        text=chunk.text,
        source_file=normalized_source,
        section_heading=chunk.section_heading,
        token_count=chunk.token_count,
        has_code=chunk.has_code,
    )
    return PromptInput(input=input_str, chunk=normalized_chunk, params=params)


def extract_concept_from_chunk(chunk: Chunk, candidate_concepts: List[str]) -> str:
    """
    Memilih konsep paling relevan dari candidate_concepts berdasarkan
    keyword matching dengan teks chunk. (Req 2.6)

    Strategi:
    1. Tokenisasi nama konsep menjadi kata-kata kunci
    2. Hitung berapa kata kunci yang muncul di teks chunk (case-insensitive)
    3. Pilih konsep dengan skor tertinggi
    4. Fallback: gunakan section_heading jika tidak ada konsep yang match

    Returns:
        Nama konsep yang paling relevan dengan isi chunk.
    """
    if not candidate_concepts:
        return chunk.section_heading or "Python"

    chunk_text_lower = chunk.text.lower()
    # Tambahkan heading ke teks yang dicari agar heading juga berkontribusi
    search_text = (chunk.section_heading + " " + chunk.text).lower()

    best_concept: Optional[str] = None
    best_score = 0

    for concept in candidate_concepts:
        # Tokenisasi nama konsep: pisah berdasarkan spasi, tanda hubung, dan tanda kurung
        keywords = re.split(r"[\s\-\(\)]+", concept.lower())
        keywords = [kw for kw in keywords if len(kw) > 2]  # abaikan kata < 3 karakter

        if not keywords:
            continue

        # Hitung berapa keyword yang muncul di teks
        score = sum(1 for kw in keywords if kw in search_text)

        # Bonus: jika nama konsep lengkap muncul sebagai substring
        if concept.lower() in search_text:
            score += len(keywords)

        if score > best_score:
            best_score = score
            best_concept = concept

    # Fallback ke section_heading jika tidak ada yang match (score == 0)
    if best_score == 0 or best_concept is None:
        return chunk.section_heading.lstrip("#").strip() or "Python"

    return best_concept


def build_prompts_for_chunk(
    chunk: Chunk,
    concepts: List[str],
    difficulties: List[str] | None = None,
    question_types: List[str] | None = None,
    auto_select_concept: bool = True,
) -> List[PromptInput]:
    """
    Menghasilkan beberapa PromptInput dari satu chunk dengan kombinasi parameter.

    Args:
        chunk: Chunk materi
        concepts: Daftar kandidat konsep untuk modul ini
        difficulties: List difficulty yang diinginkan (default: semua 3 level)
        question_types: List question type (default: ["MCQ"])
        auto_select_concept: Jika True, pilih konsep paling relevan dengan chunk (Req 2.6).
                             Jika False, gunakan semua concepts (perilaku lama).

    Default: semua difficulty × semua question_type untuk konsep terpilih.
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    if question_types is None:
        question_types = ["MCQ"]

    # Pilih konsep yang relevan dengan isi chunk (context grounding)
    if auto_select_concept and concepts:
        selected_concepts = [extract_concept_from_chunk(chunk, concepts)]
    else:
        selected_concepts = concepts

    prompts: List[PromptInput] = []
    for concept in selected_concepts:
        for difficulty in difficulties:
            for qtype in question_types:
                params = TaskParams(
                    concept=concept,
                    difficulty=difficulty,
                    question_type=qtype,
                )
                prompts.append(build_prompt(chunk, params))
    return prompts
