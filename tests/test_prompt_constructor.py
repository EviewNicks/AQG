"""
Tests untuk Prompt Constructor — verifikasi template, validasi, dan code preservation.
"""
import pytest
from src.dataset.chunker import Chunk, chunk_markdown
from src.dataset.prompt_constructor import (
    TaskParams,
    PromptInput,
    build_prompt,
    build_prompts_for_chunk,
    PROMPT_TEMPLATE,
    VALID_DIFFICULTIES,
    VALID_QUESTION_TYPES,
)

SAMPLE_FILE = "dataset_aqg/materi/01-Berkenalan-dengan-python/01-perkenalan-pythn.md"
SAMPLE_FILE_CODE = "dataset_aqg/materi/01-Berkenalan-dengan-python/05-variable-and-assignment.md"


def make_chunk(text: str, has_code: bool = False) -> Chunk:
    return Chunk(
        text=text,
        source_file="test.md",
        section_heading="# Test",
        token_count=len(text.split()),
        has_code=has_code,
    )


# ── Unit tests: TaskParams ───────────────────────────────────────────────────

class TestTaskParams:
    def test_valid_params(self):
        p = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        assert p.concept == "Sejarah Python"
        assert p.difficulty == "easy"
        assert p.question_type == "MCQ"

    def test_all_valid_difficulties(self):
        for d in ["easy", "medium", "hard"]:
            p = TaskParams(concept="X", difficulty=d, question_type="MCQ")
            assert p.difficulty == d

    def test_all_valid_question_types(self):
        for qt in ["MCQ", "Code Completion"]:
            p = TaskParams(concept="X", difficulty="easy", question_type=qt)
            assert p.question_type == qt

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            TaskParams(concept="X", difficulty="super_hard", question_type="MCQ")

    def test_invalid_question_type_raises(self):
        with pytest.raises(ValueError, match="question_type"):
            TaskParams(concept="X", difficulty="easy", question_type="Essay")

    def test_empty_difficulty_raises(self):
        with pytest.raises(ValueError):
            TaskParams(concept="X", difficulty="", question_type="MCQ")


# ── Unit tests: build_prompt ─────────────────────────────────────────────────

class TestBuildPrompt:
    def test_returns_prompt_input(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert isinstance(result, PromptInput)

    def test_input_contains_konteks_marker(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert "Konteks:" in result.input

    def test_input_contains_prompt_marker(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert "Prompt:" in result.input

    def test_input_contains_concept(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert "Sejarah Python" in result.input

    def test_input_contains_difficulty(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="Sejarah Python", difficulty="medium", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert "medium" in result.input

    def test_input_contains_question_type(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        params = TaskParams(concept="X", difficulty="easy", question_type="Code Completion")
        result = build_prompt(chunk, params)
        assert "Code Completion" in result.input

    def test_input_contains_chunk_text(self):
        chunk = make_chunk("Python adalah bahasa pemrograman multifungsi.")
        params = TaskParams(concept="X", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert "Python adalah bahasa pemrograman multifungsi." in result.input

    def test_chunk_reference_preserved(self):
        chunk = make_chunk("test text")
        params = TaskParams(concept="X", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        # chunk mungkin dinormalisasi (source_file path), cek equality bukan identity
        assert result.chunk.text == chunk.text
        assert result.chunk.section_heading == chunk.section_heading
        assert result.chunk.token_count == chunk.token_count
        assert result.chunk.has_code == chunk.has_code
        assert result.params is params

    def test_code_block_preserved_in_input(self):
        code = '```python\nprint("Hello World!")\n```'
        chunk = make_chunk(f"Contoh kode:\n{code}", has_code=True)
        params = TaskParams(concept="Fungsi print()", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunk, params)
        assert code in result.input

    def test_all_difficulty_x_type_combinations(self):
        """Semua kombinasi difficulty × question_type harus menghasilkan prompt valid."""
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        for diff in ["easy", "medium", "hard"]:
            for qtype in ["MCQ", "Code Completion"]:
                params = TaskParams(concept="Test", difficulty=diff, question_type=qtype)
                result = build_prompt(chunk, params)
                assert "Konteks:" in result.input
                assert "Prompt:" in result.input
                assert diff in result.input
                assert qtype in result.input


# ── Integration tests dengan materi nyata ───────────────────────────────────

class TestBuildPromptWithRealMaterial:
    def test_prompt_from_real_chunk(self):
        chunks = chunk_markdown(SAMPLE_FILE)
        assert len(chunks) > 0
        params = TaskParams(concept="Sejarah Python", difficulty="easy", question_type="MCQ")
        result = build_prompt(chunks[0], params)
        assert "Konteks:" in result.input
        assert "Sejarah Python" in result.input

    def test_code_block_preserved_from_real_file(self):
        """Chunk dari file dengan code block harus mempertahankan code di prompt."""
        chunks = chunk_markdown(SAMPLE_FILE_CODE)
        code_chunks = [c for c in chunks if c.has_code]
        assert len(code_chunks) > 0, "Harus ada chunk dengan code block"

        params = TaskParams(concept="Variable dan Assignment", difficulty="medium", question_type="MCQ")
        for chunk in code_chunks:
            result = build_prompt(chunk, params)
            assert "```" in result.input, "Code block harus ada di input prompt"


class TestBuildPromptsForChunk:
    def test_returns_multiple_prompts(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        prompts = build_prompts_for_chunk(
            chunk,
            concepts=["Sejarah Python", "Ciri Khas Python"],
            difficulties=["easy", "medium"],
            question_types=["MCQ"],
            auto_select_concept=False,  # gunakan semua concepts
        )
        assert len(prompts) == 4  # 2 concepts × 2 difficulties × 1 type

    def test_default_generates_mcq_only(self):
        chunk = make_chunk("Python adalah bahasa pemrograman.")
        prompts = build_prompts_for_chunk(chunk, concepts=["X"])
        for p in prompts:
            assert p.params.question_type == "MCQ"
