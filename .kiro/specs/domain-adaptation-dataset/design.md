# Design Document: Domain Adaptation Dataset Pipeline

## Overview

Pipeline modular untuk mengubah seluruh materi kursus Python Basics (11 modul, 55+ lesson, format Markdown) menjadi dataset JSONL untuk tahap Domain Adaptation fine-tuning IndoT5. Pipeline menghasilkan tiga format data yang saling melengkapi: Span Corruption (gaya T5 pre-training), Summarization, dan QA Generik.

Pipeline ini adalah **Tahap 1** dari pendekatan hibrida:
- Tahap 1 (spec ini): Domain Adaptation → `IndoT5-Python`
- Tahap 2 (spec `aqg-dataset-pipeline`): Task-Specific AQG → `IndoT5-Python-AQG`

Scope: Hanya persiapan dataset domain adaptation. Fine-tuning model berada di luar scope ini.

## Architecture

```
dataset_aqg/materi/
    └── **/*.md  (11 modul, 55+ lesson)
         │
         ▼
    ┌─────────────┐
    │   Chunker   │  → reuse dari aqg-dataset-pipeline, chunk 128–512 token
    └──────┬──────┘
           │ List[Chunk]
           ▼
    ┌──────────────────────────────────────────┐
    │              Formatter                   │
    │  ┌──────────────────┐                   │
    │  │  SpanCorruptor   │ → zero LLM cost   │
    │  ├──────────────────┤                   │
    │  │   Summarizer     │ → via LLM API     │
    │  ├──────────────────┤                   │
    │  │   QAGenerator    │ → zero LLM cost   │
    │  └──────────────────┘                   │
    └──────────────────┬───────────────────────┘
                       │ List[RawDomainDataPoint]
                       ▼
               ┌───────────────┐
               │   Validator   │  → cek format, panjang, metadata
               └──────┬────────┘
                      │ List[ValidDomainDataPoint]
                      ▼
              ┌──────────────────┐
              │  Domain Writer   │  → split 80/10/10, simpan JSONL
              └──────────────────┘
                      │
                      ▼
              dataset_aqg/output_domain/
                  ├── train.jsonl
                  ├── validation.jsonl
                  ├── test.jsonl
                  └── dataset_info.json
```

## Components and Interfaces

### 1. Chunker (reuse dari `src/dataset/chunker.py`)

Chunker yang sudah ada di `aqg-dataset-pipeline` di-reuse langsung. Perbedaan hanya pada parameter token range: domain adaptation menggunakan 128–512 token (lebih lebar dari AQG yang 250–400).

```python
# Import langsung dari modul yang sudah ada
from src.dataset.chunker import Chunk, chunk_markdown, chunk_all_materials

# Panggil dengan parameter domain adaptation
chunks = chunk_all_materials(materi_dir, max_tokens=512, min_tokens=128)
# Komponen baru ditempatkan di src/dataset/step1/
```

Tidak ada perubahan pada `chunker.py` — parameter `max_tokens` dan `min_tokens` sudah ada sebagai argumen fungsi.

### 2. Formatter (`src/dataset/step1/formatter.py`)

Tiga sub-komponen dalam satu modul, masing-masing mengubah `Chunk` menjadi `RawDomainDataPoint`.

```python
@dataclass
class RawDomainDataPoint:
    input: str          # string input untuk model
    target: str         # string target untuk model
    metadata: dict      # format, source_file, module_name, section_heading, token_count, has_code
```

#### 2a. SpanCorruptor

Mengimplementasikan span corruption gaya T5 original. Tidak membutuhkan LLM.

```python
def corrupt_spans(chunk: Chunk, corruption_rate: float = 0.15) -> RawDomainDataPoint:
    """
    Mask 15% token dengan sentinel tokens.
    Input:  "Python adalah <extra_id_0> yang <extra_id_1> pada tahun 1991"
    Target: "<extra_id_0> bahasa pemrograman <extra_id_1> dirilis <extra_id_2>"
    """
    ...
```

Algoritma span corruption:
1. Tokenisasi sederhana: split by whitespace (konsisten dengan estimasi token di Chunker)
2. Hitung jumlah token yang akan di-mask: `n_mask = max(1, int(len(tokens) * corruption_rate))`
3. Pilih span secara acak: panjang span 2–5 token, tidak overlap, tidak di dalam code block
4. Ganti setiap span dengan sentinel token `<extra_id_N>` (N dimulai dari 0)
5. `input`: teks dengan span diganti sentinel
6. `target`: sentinel diikuti span asli, diakhiri `<extra_id_N+1>` sebagai end marker

Contoh konkret:
```
Teks asli: "Python adalah bahasa pemrograman yang dirilis pada tahun 1991 oleh Guido van Rossum"

Input:  "Python adalah <extra_id_0> yang dirilis pada <extra_id_1> oleh Guido van Rossum"
Target: "<extra_id_0> bahasa pemrograman <extra_id_1> tahun 1991 <extra_id_2>"
```

Catatan: Code block tidak di-mask untuk menjaga integritas kode Python.

#### 2b. Summarizer

Memanggil LLM via OpenRouter untuk menghasilkan ringkasan.

```python
SUMMARIZATION_SYSTEM_PROMPT = """
Kamu adalah asisten pendidikan Python berbahasa Indonesia.
Tugas kamu adalah merangkum teks materi Python berikut menjadi 2-4 kalimat yang padat dan informatif.
Pertahankan semua istilah teknis Python (nama fungsi, tipe data, keyword) dalam ringkasan.
Gunakan bahasa Indonesia yang jelas dan mudah dipahami siswa.
Berikan HANYA ringkasan, tanpa penjelasan tambahan.
"""

def summarize_chunk(chunk: Chunk, llm_client, max_retries: int = 2) -> Optional[RawDomainDataPoint]:
    """
    Input:  "Rangkum teks berikut:\n\n{chunk.text}"
    Target: "<ringkasan 2-4 kalimat>"
    """
    ...
```

#### 2c. QAGenerator

Ekstrak pasangan QA faktual dari teks menggunakan rule-based heuristics. Tidak membutuhkan LLM.

```python
def extract_qa_pairs(chunk: Chunk) -> List[RawDomainDataPoint]:
    """
    Ekstrak QA dari bold text, inline code, dan heading.
    Input:  "Apa itu list dalam Python?"
    Target: "List merupakan kumpulan data terurut (ordered sequence) yang dapat menyimpan berbagai tipe data."
    """
    ...
```

Strategi ekstraksi:
1. **Bold terms** (`**term**`): generate `"Apa itu {term} dalam Python?"` → kalimat yang mengandung term sebagai target
2. **Inline code** (`` `term` ``): generate `"Jelaskan {term} dalam Python."` → kalimat konteks sebagai target
3. **Heading text**: generate `"Apa yang dimaksud dengan {heading}?"` → kalimat pertama section sebagai target
4. Deduplikasi: jika term muncul di bold dan inline code, ambil satu saja
5. Minimal 1 QA per chunk yang memiliki bold term atau inline code

### 3. Validator (`src/dataset/step1/validator.py`)

Validasi ringan — lebih longgar dari AQG validator karena format domain adaptation lebih beragam.

```python
@dataclass
class DomainValidationResult:
    is_valid: bool
    failure_reasons: List[str]

@dataclass
class ValidDomainDataPoint:
    input: str
    target: str
    metadata: dict

VALID_FORMATS = {"span_corruption", "summarization", "qa_generic"}

def validate_domain(datapoint: RawDomainDataPoint) -> DomainValidationResult:
    """Validasi satu domain data point."""
    ...

def validate_domain_batch(
    datapoints: List[RawDomainDataPoint]
) -> Tuple[List[ValidDomainDataPoint], List[dict]]:
    """Validasi batch. Kembalikan (valid_list, failure_log)."""
    ...
```

Aturan validasi:
- `input`: panjang 10–1024 token
- `target`: non-empty string (minimal 5 karakter)
- `metadata.format`: hanya `"span_corruption"`, `"summarization"`, `"qa_generic"`
- `metadata.source_file`: non-empty string
- `metadata.module_name`: non-empty string

### 4. Domain Dataset Writer (`src/dataset/step1/dataset_writer.py`)

```python
def write_domain_dataset(
    datapoints: List[ValidDomainDataPoint],
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    stratify_by: str = "format"
) -> None:
    """Split dan simpan dataset domain adaptation ke JSONL files."""
    ...
```

Split stratifikasi berdasarkan `format` — memastikan ketiga format terwakili di setiap split.

### 5. Pipeline Runner (`dataset_aqg/run_domain_pipeline.py`)

```
python run_domain_pipeline.py \
    --materi-dir dataset_aqg/materi \
    --output-dir dataset_aqg/output_domain \
    --formats span_corruption,qa_generic,summarization \
    --max-per-chunk 3
```

Checkpointing per modul: setelah setiap modul selesai diproses, simpan progress ke `output_dir/checkpoint.json`. Saat restart, skip modul yang sudah ada di checkpoint.

## Data Models

### Skema JSONL (satu baris per data point)

**Format Span Corruption:**
```json
{
  "input": "Python adalah <extra_id_0> yang <extra_id_1> pada tahun 1991 oleh Guido van Rossum.",
  "target": "<extra_id_0> bahasa pemrograman <extra_id_1> dirilis <extra_id_2>",
  "metadata": {
    "format": "span_corruption",
    "source_file": "01-Berkenalan-dengan-python/01-perkenalan-python.md",
    "module_name": "01-berkenalan-dengan-python",
    "section_heading": "Sejarah Python",
    "token_count": 45,
    "has_code": false
  }
}
```

**Format Summarization:**
```json
{
  "input": "Rangkum teks berikut:\n\nTipe data `boolean` hanya bernilai `True` atau `False`...",
  "target": "Tipe data boolean dalam Python hanya memiliki dua nilai: True dan False. Boolean digunakan untuk merepresentasikan nilai kebenaran dan sering digunakan dalam kondisi percabangan.",
  "metadata": {
    "format": "summarization",
    "source_file": "02-berinteraksi-dengan-data/03-type-data.md",
    "module_name": "02-berinteraksi-dengan-data",
    "section_heading": "Boolean",
    "token_count": 120,
    "has_code": true
  }
}
```

**Format QA Generik:**
```json
{
  "input": "Apa itu list dalam Python?",
  "target": "List merupakan kumpulan data terurut (ordered sequence) dan salah satu tipe data yang paling sering digunakan.",
  "metadata": {
    "format": "qa_generic",
    "source_file": "02-berinteraksi-dengan-data/03-type-data.md",
    "module_name": "02-berinteraksi-dengan-data",
    "section_heading": "List",
    "token_count": 89,
    "has_code": false
  }
}
```

### dataset_info.json

```json
{
  "total": 3500,
  "splits": {"train": 2800, "validation": 350, "test": 350},
  "format_distribution": {
    "span_corruption": 1500,
    "summarization": 800,
    "qa_generic": 1200
  },
  "module_distribution": {
    "01-berkenalan-dengan-python": 320,
    "02-berinteraksi-dengan-data": 410
  },
  "generated_at": "2026-04-07"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Chunk token count invariant (domain range)

*For any* Markdown file processed by the Chunker with domain adaptation parameters, every resulting chunk SHALL have a token count between 1 and 512 (inclusive).

**Validates: Requirements 1.1, 1.4**

### Property 2: Code block integrity in chunks

*For any* chunk produced by the Chunker, if the chunk contains a Python code block opening marker (` ```python `), it SHALL also contain the corresponding closing marker (` ``` `).

**Validates: Requirements 1.2**

### Property 3: Span corruption sentinel balance

*For any* span-corrupted data point, the number of sentinel tokens in `input` SHALL equal the number of sentinel tokens in `target` minus one (the end marker). Formally: `count_sentinels(input) == count_sentinels(target) - 1`.

**Validates: Requirements 2.1, 2.2**

### Property 4: Span corruption rate invariant

*For any* chunk processed by the SpanCorruptor, the number of masked tokens SHALL be between 1 and `ceil(token_count * 0.25)` — never zero and never exceeding 25% of the original token count.

**Validates: Requirements 2.1**

### Property 5: Code block not masked in span corruption

*For any* chunk that contains a Python code block, the span-corrupted `input` SHALL contain the code block text unchanged (no sentinel tokens inside code blocks).

**Validates: Requirements 2.4**

### Property 6: Summarization input prefix

*For any* summarization data point, the `input` string SHALL start with the prefix `"Rangkum teks berikut:"`.

**Validates: Requirements 3.1**

### Property 7: QA generic term presence

*For any* QA generic data point, the term used in the `input` question SHALL appear in the `target` answer string.

**Validates: Requirements 4.3**

### Property 8: JSONL round-trip consistency

*For any* list of valid domain data points written to a JSONL file, loading that file back SHALL produce an equivalent list where each entry has exactly the keys `input`, `target`, `metadata`, with both `input` and `target` as strings and `metadata` as a dict.

**Validates: Requirements 5.1, 5.4**

### Property 9: Split stratification by format

*For any* dataset split by the Domain Writer, each split (train, val, test) SHALL contain at least one entry for each format that exists in the original dataset.

**Validates: Requirements 5.3**

### Property 10: Validator rejects out-of-range inputs

*For any* data point where `input` token count is < 10 or > 1024, the Validator SHALL return `is_valid = False` with a non-empty `failure_reasons` list.

**Validates: Requirements 6.1**

### Property 11: Metadata format enum validity

*For any* valid domain data point, `metadata.format` SHALL be one of `"span_corruption"`, `"summarization"`, or `"qa_generic"`.

**Validates: Requirements 6.4**

## Error Handling

- LLM API failure (Summarizer): retry 2x dengan exponential backoff (1s, 2s), lalu skip dan log.
- Chunk terlalu pendek untuk summarization (< 100 token): skip format summarization untuk chunk tersebut, lanjut ke format lain.
- File Markdown tidak bisa dibaca: log warning, lanjut ke file berikutnya.
- Tidak ada bold term / inline code di chunk: QAGenerator skip chunk tersebut (tidak error).
- Sentinel token overflow (> 100 span): batasi maksimal 10 span per chunk untuk span corruption.

## Testing Strategy

### Unit Tests

Menggunakan `pytest`. Fokus pada:
- SpanCorruptor: test dengan teks sederhana, teks dengan code block, teks pendek (edge case)
- QAGenerator: test ekstraksi dari bold text, inline code, heading; test chunk tanpa term apapun
- Validator: test boundary values (9 token, 10 token, 1024 token, 1025 token), missing fields, invalid enum
- Domain Writer: test split ratio, test JSONL format output, test stratifikasi

### Property-Based Tests

Menggunakan `hypothesis`. Minimum 100 iterasi per property.

```python
# Feature: domain-adaptation-dataset, Property N: <property_text>
settings = Settings(max_examples=100, deadline=None)
```

Generator strategies:
- `st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10)` untuk teks biasa
- Generator khusus untuk teks dengan code block: inject ` ```python\nprint("x")\n``` ` ke dalam teks acak
- `st.sampled_from(["span_corruption", "summarization", "qa_generic"])` untuk format
- Generator untuk chunk dengan bold terms: inject `**term**` ke dalam teks acak
