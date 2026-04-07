# Design Document: AQG Dataset Pipeline

## Overview

Pipeline modular untuk mengubah materi kursus Python Basics (format Markdown) menjadi dataset JSONL berkualitas tinggi, siap digunakan untuk fine-tuning model IndoT5 + LoRA. Pipeline terdiri dari 5 komponen utama yang berjalan secara berurutan: Chunker → Prompt Constructor → Synthetic Generator → Validator → Dataset Writer.

Scope: Hanya persiapan dataset. Fine-tuning model, evaluasi model, dan integrasi API berada di luar scope ini.

## Architecture

```
dataset_aqg/materi/
    └── **/*.md
         │
         ▼
    ┌─────────────┐
    │   Chunker   │  → memotong Markdown menjadi chunk 250–400 token
    └──────┬──────┘
           │ List[Chunk]
           ▼
    ┌──────────────────────┐
    │  Prompt Constructor  │  → membangun input string dari chunk + parameter
    └──────────┬───────────┘
               │ List[PromptInput]
               ▼
    ┌──────────────────────┐
    │  Synthetic Generator │  → memanggil LLM API untuk generate target
    └──────────┬───────────┘
               │ List[RawDataPoint]
               ▼
    ┌───────────────┐
    │   Validator   │  → memvalidasi format, panjang, dan kelengkapan field
    └──────┬────────┘
           │ List[ValidDataPoint]
           ▼
    ┌──────────────────┐
    │  Dataset Writer  │  → split train/val/test, simpan JSONL + info
    └──────────────────┘
           │
           ▼
    dataset_aqg/output/
        ├── train.jsonl
        ├── validation.jsonl
        ├── test.jsonl
        └── dataset_info.json
```

## Components and Interfaces

### 1. Chunker (`src/chunker.py`)

Membaca file Markdown dan memotongnya menjadi chunk berdasarkan heading dan batas paragraf.

```python
@dataclass
class Chunk:
    text: str               # teks chunk (konteks materi)
    source_file: str        # path file Markdown asal
    section_heading: str    # heading terdekat di atas chunk
    token_count: int        # estimasi jumlah token
    has_code: bool          # apakah chunk mengandung code block

def chunk_markdown(filepath: str, max_tokens: int = 400, min_tokens: int = 250) -> List[Chunk]:
    """Membaca satu file Markdown dan mengembalikan list of Chunk."""
    ...

def chunk_all_materials(materi_dir: str) -> List[Chunk]:
    """Iterasi semua file .md di materi_dir dan mengembalikan semua chunk."""
    ...
```

Strategi chunking:
- Split berdasarkan heading (`#`, `##`, `###`) — setiap heading memulai chunk baru
- Jika section terlalu panjang (> 400 token), split di batas kalimat terdekat
- Code block (```` ```python ... ``` ````) tidak pernah dipotong — selalu dipertahankan utuh dalam satu chunk
- Token count diestimasi dengan `len(text.split()) * 1.3` (pendekatan sederhana, tidak perlu tokenizer penuh)

### 2. Prompt Constructor (`src/prompt_constructor.py`)

Membangun string `input` dari chunk + parameter tugas.

```python
@dataclass
class TaskParams:
    concept: str            # konsep yang diuji, dari Master Concept List
    difficulty: str         # "easy" | "medium" | "hard"
    question_type: str      # "MCQ" | "Code Completion"

@dataclass
class PromptInput:
    input: str              # string input lengkap untuk model
    chunk: Chunk            # chunk asal
    params: TaskParams      # parameter tugas

PROMPT_TEMPLATE = (
    "Konteks: {context}\n\n"
    "Prompt: Buat satu soal {question_type} tentang {concept}, "
    "tingkat kesulitan: {difficulty}, bahasa Indonesia."
)

def build_prompt(chunk: Chunk, params: TaskParams) -> PromptInput:
    """Membangun satu PromptInput dari chunk dan parameter."""
    ...
```

### 3. Synthetic Generator (`src/synthetic_generator.py`)

Memanggil LLM API (GPT-4o atau Claude) untuk menghasilkan target dari PromptInput.

```python
@dataclass
class RawDataPoint:
    input: str
    target: str             # plain string hasil LLM
    metadata: dict
    source: str             # "synthetic" | "manual" | "augmented"

GENERATION_SYSTEM_PROMPT = """
Kamu adalah pembuat soal kuis Python untuk siswa Indonesia.
PENTING: Buat soal HANYA berdasarkan informasi yang ada di dalam teks Konteks yang diberikan. Jangan gunakan pengetahuan di luar konteks.

Berikan output HANYA dalam format berikut (tanpa teks lain):
Pertanyaan: <pertanyaan>? Jawaban benar: <jawaban>. Distraktor: 1) <d1> 2) <d2> 3) <d3> 4) <d4>. Misconception tags: <tag1>, <tag2>

Misconception tags adalah label singkat (bahasa Indonesia atau Inggris) yang menjelaskan miskonsepsi umum siswa yang ditargetkan oleh setiap distraktor. Contoh: tokoh_pemrograman_lain, salah_versi_python, bingung_tipe_data.
"""

def generate_datapoint(prompt_input: PromptInput, llm_client, max_retries: int = 2) -> Optional[RawDataPoint]:
    """Memanggil LLM dan mengembalikan RawDataPoint, atau None jika gagal."""
    ...
```

LLM yang didukung: OpenAI GPT-4o (via `openai` library) dan Anthropic Claude (via `anthropic` library). Dipilih via environment variable `LLM_PROVIDER`.

### 4. Validator (`src/validator.py`)

Memvalidasi setiap RawDataPoint sebelum masuk ke dataset final.

```python
@dataclass
class ValidationResult:
    is_valid: bool
    failure_reasons: List[str]

@dataclass
class ValidDataPoint:
    input: str
    target: str
    metadata: dict

def validate(datapoint: RawDataPoint) -> ValidationResult:
    """Memvalidasi satu data point. Mengembalikan ValidationResult."""
    ...

def validate_batch(datapoints: List[RawDataPoint]) -> Tuple[List[ValidDataPoint], List[dict]]:
    """Memvalidasi batch. Mengembalikan (valid_list, failure_log)."""
    ...
```

Aturan validasi:
- `input`: panjang 50–600 token
- `target`: harus mengandung substring "Pertanyaan:", "Jawaban benar:", "Distraktor:"
- `metadata`: harus memiliki field `difficulty`, `question_type`, `concept`, `misconception_tags`
- `metadata.difficulty`: hanya `"easy"`, `"medium"`, `"hard"`
- `metadata.question_type`: hanya `"MCQ"`, `"Code Completion"`
- `metadata.misconception_tags`: harus berupa list non-empty (minimal 1 tag)
- `metadata.validated`: di-set ke `true` setelah data point lolos semua validasi
- `metadata.source_file`: path dinormalisasi menggunakan forward slash (`/`)

### 5. Dataset Writer (`src/dataset_writer.py`)

Menyimpan data valid ke JSONL dengan split train/val/test.

```python
def write_dataset(
    datapoints: List[ValidDataPoint],
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "difficulty"
) -> None:
    """Split dan simpan dataset ke JSONL files."""
    ...
```

### 6. Augmentor (`src/augmentor.py`) — opsional

Menghasilkan varian dari seed data yang sudah ada.

```python
def augment_by_difficulty(datapoint: ValidDataPoint, llm_client) -> List[ValidDataPoint]:
    """Menghasilkan varian easy/medium/hard dari satu data point."""
    ...

def deduplicate(datapoints: List[ValidDataPoint]) -> List[ValidDataPoint]:
    """Menghapus entri dengan input string yang identik."""
    ...
```

## Data Models

### Skema JSONL (satu baris per data point)

```json
{
  "input": "Konteks: Python adalah bahasa pemrograman multifungsi yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR).\n\nPrompt: Buat satu soal MCQ tentang Sejarah Python, tingkat kesulitan: easy, bahasa Indonesia.",
  "target": "Pertanyaan: Siapakah pencipta bahasa pemrograman Python? Jawaban benar: Guido van Rossum. Distraktor: 1) Dennis Ritchie 2) James Gosling 3) Bjarne Stroustrup 4) Linus Torvalds",
  "metadata": {
    "difficulty": "easy",
    "question_type": "MCQ",
    "concept": "Sejarah Python",
    "misconception_tags": ["tokoh_pemrograman_lain"],
    "source_file": "01-Berkenalan-dengan-python/01-perkenalan-pythn.md",
    "section": "Pengenalan Python",
    "source": "synthetic",
    "validated": true
  }
}
```

Catatan kritis: `target` adalah **plain string**, bukan JSON object. Ini wajib untuk fine-tuning seq2seq.

### Master Concept List (`src/concept_list.py`)

Daftar konsep yang digunakan sebagai nilai `metadata.concept`, dikelompokkan per modul:

```python
CONCEPTS = {
    "01-berkenalan-dengan-python": [
        "Sejarah Python", "Ciri Khas Python", "Versi Python 2.x",
        "Versi Python 3.x", "Python Software Foundation", "Zen of Python",
        "Fungsi print()", "Menjalankan File Python", "Variable dan Assignment",
        "Input Output Python"
    ],
    "02-berinteraksi-dengan-data": [
        "Abstraksi Data", "Tipe Data Python", "Integer", "Float", "String",
        "Boolean", "List", "Dictionary", "Set", "Type Conversion"
    ],
    # ... dst untuk modul lainnya
}
```

### dataset_info.json

```json
{
  "total": 1500,
  "splits": {"train": 1050, "validation": 225, "test": 225},
  "concept_distribution": {"Sejarah Python": 45, "Variable dan Assignment": 60, "...": "..."},
  "difficulty_distribution": {"easy": 500, "medium": 600, "hard": 400},
  "source_distribution": {"synthetic": 1200, "manual": 200, "augmented": 100},
  "generated_at": "2026-03-31"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Chunk token count invariant

*For any* Markdown file processed by the Chunker, every resulting chunk SHALL have a token count between 1 and 400 (inclusive). No chunk shall exceed the maximum token limit, including edge cases where a single paragraph is longer than 400 tokens.

**Validates: Requirements 1.1, 1.4**

### Property 2: Code block integrity

*For any* chunk produced by the Chunker, if the chunk contains a Python code block opening marker (` ```python `), it SHALL also contain the corresponding closing marker (` ``` `). No chunk shall contain an unclosed code block.

**Validates: Requirements 1.2**

### Property 3: Chunk metadata completeness

*For any* chunk produced by the Chunker, the chunk object SHALL have non-empty values for `source_file`, `section_heading`, and `token_count > 0`.

**Validates: Requirements 1.5**

### Property 4: Prompt template conformance

*For any* chunk and task parameters, the `input` string produced by the Prompt Constructor SHALL contain the substring "Konteks:" and the substring "Prompt:", and SHALL contain the `concept`, `difficulty`, and `question_type` values from the task parameters.

**Validates: Requirements 2.1, 2.2**

### Property 5: Code preservation in prompt

*For any* chunk that contains a Python code block, the `input` string produced by the Prompt Constructor SHALL contain the same code block text unchanged.

**Validates: Requirements 2.5**

### Property 6: Target format invariant

*For any* valid data point in the dataset, the `target` field SHALL be a string (not a dict or list), and SHALL contain all three required section markers: "Pertanyaan:", "Jawaban benar:", "Distraktor:".

**Validates: Requirements 3.1, 3.2**

### Property 7: Metadata schema validity

*For any* valid data point in the dataset, the `metadata` field SHALL be a dict containing all required keys (`difficulty`, `question_type`, `concept`, `misconception_tags`), with `difficulty` ∈ {`"easy"`, `"medium"`, `"hard"`} and `question_type` ∈ {`"MCQ"`, `"Code Completion"`}.

**Validates: Requirements 4.1, 4.3, 4.4**

### Property 8: JSONL round-trip consistency

*For any* list of valid data points written to a JSONL file, loading that file back SHALL produce an equivalent list where each entry has exactly the keys `input`, `target`, `metadata`, with `target` as a string and `metadata` as a dict.

**Validates: Requirements 5.1, 5.4**

### Property 9: Split stratification

*For any* dataset split by the Dataset Writer, each split (train, val, test) SHALL contain at least one entry for each difficulty level that exists in the original dataset.

**Validates: Requirements 5.3**

### Property 10: Validator rejects invalid inputs

*For any* data point where `input` token count is < 50 or > 600, the Validator SHALL return `is_valid = False` with a non-empty `failure_reasons` list.

**Validates: Requirements 6.1**

### Property 11: Augmentation deduplication

*For any* list of data points produced by the Augmentor, no two entries SHALL have identical `input` strings after deduplication is applied.

**Validates: Requirements 8.5**

### Property 12: Misconception tags non-empty

*For any* valid data point in the dataset, `metadata.misconception_tags` SHALL be a non-empty list containing at least one string tag.

**Validates: Requirements 4.6, 6.6, 7.6**

### Property 13: Validated flag set on passing data points

*For any* data point that passes all validation checks, `metadata.validated` SHALL be `true`.

**Validates: Requirements 4.7, 6.7**

### Property 14: Source file path normalization

*For any* valid data point, `metadata.source_file` SHALL use forward slashes (`/`) as path separators, regardless of the operating system used to generate the data.

**Validates: Requirements 4.5**

## Error Handling

- LLM API failure: retry up to 2 times, then skip and log. Pipeline tidak berhenti karena satu entry gagal.
- Validation failure: log ke `validation_failures.jsonl` dengan field `reason` dan `raw_datapoint`.
- File not found: log warning dan lanjut ke file berikutnya.
- Rate limit (LLM API): exponential backoff dengan delay 1s, 2s, 4s.
- Malformed LLM response: jika target tidak mengandung "Pertanyaan:", coba parse ulang atau skip.

## Testing Strategy

### Unit Tests

Menggunakan `pytest`. Fokus pada:
- Chunker: test dengan Markdown sederhana, Markdown dengan code block, Markdown dengan heading bertingkat
- Prompt Constructor: test template output untuk setiap kombinasi question_type × difficulty
- Validator: test setiap aturan validasi secara individual (boundary values, missing fields, invalid enum)
- Dataset Writer: test split ratio, test JSONL format output

### Property-Based Tests

Menggunakan `hypothesis` (Python PBT library). Minimum 100 iterasi per property.

Setiap property test harus diberi tag komentar:
```python
# Feature: aqg-dataset-pipeline, Property N: <property_text>
```

Strategi generator:
- `st.text()` untuk konten Markdown dengan variasi heading dan paragraf
- `st.sampled_from(["easy", "medium", "hard"])` untuk difficulty
- `st.sampled_from(["MCQ", "Code Completion"])` untuk question_type
- Generator khusus untuk Markdown dengan code block: inject ` ```python\nprint("x")\n``` ` ke dalam teks acak

Konfigurasi:
```python
settings = Settings(max_examples=100, deadline=None)
```
