# Requirements Document

## Introduction

Pipeline persiapan dataset untuk tahap Domain Adaptation pada proyek AQG Python. Pipeline ini mengubah seluruh materi kursus Python Basics (11 modul, format Markdown) menjadi dataset JSONL yang digunakan untuk melatih IndoT5 agar memiliki pemahaman mendalam tentang domain pendidikan Python berbahasa Indonesia — sebelum dilakukan task-specific fine-tuning AQG.

Scope: Hanya mencakup persiapan dataset domain adaptation. Fine-tuning model, evaluasi model, dan integrasi API berada di luar scope ini. Pipeline AQG task-specific (spec `aqg-dataset-pipeline`) tetap berjalan terpisah.

## Glossary

- **Domain Adaptation**: Tahap fine-tuning pertama di mana IndoT5 dilatih pada korpus materi Python untuk membangun pemahaman domain sebelum task-specific fine-tuning.
- **Korpus**: Kumpulan seluruh teks materi Python dari 11 modul yang digunakan sebagai sumber data domain adaptation.
- **Chunk**: Potongan teks materi berukuran 128–512 token, unit dasar dataset domain adaptation.
- **Format Span Corruption**: Format pre-training T5 di mana sebagian token dalam teks di-mask dan model belajar merekonstruksinya. Cocok untuk domain adaptation IndoT5.
- **Format Summarization**: Pasangan teks panjang (input) dan ringkasan singkat (target). Mengajarkan model memahami inti konten materi.
- **Format QA Generik**: Pasangan pertanyaan sederhana (input) dan jawaban faktual (target) yang diekstrak langsung dari teks materi tanpa LLM.
- **Modul**: Satu folder materi yang berisi beberapa file Markdown lesson (contoh: `01-Berkenalan-dengan-python/`).
- **Lesson**: Satu file Markdown yang membahas satu topik spesifik dalam sebuah modul.
- **IndoT5**: Model T5 berbahasa Indonesia yang menjadi base model untuk fine-tuning.
- **LoRA**: Low-Rank Adaptation — teknik fine-tuning efisien yang digunakan pada kedua tahap fine-tuning.
- **Chunker**: Komponen yang memotong file Markdown menjadi chunk-chunk (di-reuse dari `aqg-dataset-pipeline`).
- **Formatter**: Komponen yang mengubah chunk menjadi pasangan input-target sesuai format yang dipilih.
- **Domain Dataset Writer**: Komponen yang menyimpan dataset domain adaptation ke JSONL.

## Requirements

### Requirement 1: Chunking Materi untuk Domain Adaptation

**User Story:** As a researcher, I want to split all 11 modules of Python course material into text chunks, so that I can use them as the corpus for domain adaptation fine-tuning.

#### Acceptance Criteria

1. WHEN a Markdown file is provided, THE Chunker SHALL split it into chunks of 128–512 tokens based on heading and paragraph boundaries.
2. WHEN a chunk contains a Python code block, THE Chunker SHALL preserve the code block intact within the chunk without splitting it mid-block.
3. WHEN a heading is encountered, THE Chunker SHALL start a new chunk at that heading boundary.
4. IF a section exceeds 512 tokens, THEN THE Chunker SHALL split it at the nearest sentence boundary without exceeding the token limit.
5. THE Chunker SHALL attach metadata to each chunk: `source_file`, `module_name`, `section_heading`, `token_count`, `has_code`.
6. THE Chunker SHALL process all 11 module directories recursively and return a flat list of all chunks.

### Requirement 2: Format Span Corruption (T5 Pre-training Style)

**User Story:** As a researcher, I want to generate span corruption training pairs from text chunks, so that IndoT5 can learn to reconstruct masked spans of Python course content.

#### Acceptance Criteria

1. WHEN a chunk is provided, THE Span_Corruptor SHALL mask 15% of tokens by replacing contiguous spans with a sentinel token (e.g., `<extra_id_0>`, `<extra_id_1>`).
2. THE Span_Corruptor SHALL produce an `input` string containing the masked text and a `target` string containing only the masked spans with their sentinel tokens.
3. THE Span_Corruptor SHALL mask spans of 2–5 consecutive tokens, not individual tokens.
4. WHEN a chunk contains a Python code block, THE Span_Corruptor SHALL apply masking to code tokens as well, treating code as part of the corpus.
5. THE Span_Corruptor SHALL tag each output entry with `"format": "span_corruption"` in metadata.

### Requirement 3: Format Summarization

**User Story:** As a researcher, I want to generate summarization training pairs from text chunks, so that IndoT5 learns to understand and condense Python course material.

#### Acceptance Criteria

1. WHEN a chunk of at least 100 tokens is provided, THE Summarizer SHALL produce an `input` string prefixed with `"Rangkum teks berikut: {chunk_text}"` and a `target` string containing a concise summary.
2. THE Summarizer SHALL generate summaries using an LLM API (via OpenRouter), instructed to summarize in Bahasa Indonesia in 2–4 sentences.
3. THE Summarizer SHALL preserve key technical terms (Python keywords, function names, type names) in the generated summary.
4. IF the LLM response fails or is empty, THEN THE Summarizer SHALL skip the entry and log the failure.
5. THE Summarizer SHALL tag each output entry with `"format": "summarization"` in metadata.
6. THE Summarizer SHALL retry failed LLM calls up to 2 times with exponential backoff before skipping.

### Requirement 4: Format QA Generik

**User Story:** As a researcher, I want to generate simple factual QA pairs from text chunks without using an LLM, so that I can produce domain adaptation data at scale with zero API cost.

#### Acceptance Criteria

1. WHEN a chunk is provided, THE QA_Generator SHALL extract factual QA pairs using rule-based heuristics from the chunk text.
2. THE QA_Generator SHALL generate questions of the form `"Apa itu {term}?"` or `"Jelaskan {term} dalam Python."` where `{term}` is a key concept identified in the chunk.
3. THE QA_Generator SHALL use the sentence containing the term definition as the `target` answer.
4. THE QA_Generator SHALL identify key terms by: bold text (`**term**`), inline code (`` `term` ``), and heading text.
5. THE QA_Generator SHALL produce at least one QA pair per chunk that contains a bold term or inline code term.
6. THE QA_Generator SHALL tag each output entry with `"format": "qa_generic"` in metadata.

### Requirement 5: Dataset Output Format

**User Story:** As a researcher, I want the domain adaptation dataset saved as JSONL, so that it is directly loadable by Hugging Face `datasets` library for fine-tuning IndoT5.

#### Acceptance Criteria

1. THE Pipeline SHALL save the final dataset as JSONL files with one JSON object per line.
2. THE Pipeline SHALL produce three split files: `train.jsonl` (80%), `validation.jsonl` (10%), `test.jsonl` (10%).
3. WHEN splitting, THE Pipeline SHALL stratify by `format` to ensure all three formats (span_corruption, summarization, qa_generic) are represented in each split.
4. EACH line in the JSONL file SHALL contain exactly three keys: `input`, `target`, `metadata`.
5. THE Pipeline SHALL save a `dataset_info.json` file containing: total count, split counts, format distribution, module distribution, and `generated_at` timestamp.
6. THE Metadata for each entry SHALL contain: `format`, `source_file`, `module_name`, `section_heading`, `token_count`, `has_code`.

### Requirement 6: Validasi Dataset

**User Story:** As a researcher, I want each data point validated before saving, so that the domain adaptation dataset maintains minimum quality standards.

#### Acceptance Criteria

1. WHEN a data point is validated, THE Validator SHALL check that `input` length is between 10 and 1024 tokens.
2. WHEN a data point is validated, THE Validator SHALL check that `target` is a non-empty string.
3. WHEN a data point is validated, THE Validator SHALL check that `metadata` contains all required fields: `format`, `source_file`, `module_name`.
4. WHEN `format` is set, THE Validator SHALL accept only values: `"span_corruption"`, `"summarization"`, `"qa_generic"`.
5. IF a data point fails validation, THEN THE Validator SHALL log the failure reason and skip the invalid entry.
6. THE Validator SHALL produce a validation report showing: total processed, passed, failed, and failure reasons grouped by format.

### Requirement 7: Pipeline Runner

**User Story:** As a researcher, I want a single script to run the entire domain adaptation dataset pipeline, so that I can generate the full corpus with one command.

#### Acceptance Criteria

1. THE Runner SHALL accept CLI arguments: `--materi-dir`, `--output-dir`, `--formats` (comma-separated: `span_corruption,summarization,qa_generic`), `--max-per-chunk`.
2. THE Runner SHALL process all modules in `--materi-dir` and generate data in all specified formats.
3. THE Runner SHALL display a progress bar using `tqdm` showing current module and format being processed.
4. THE Runner SHALL save `validation_failures.jsonl` in the output directory.
5. WHEN the pipeline is interrupted and restarted, THE Runner SHALL skip modules that have already been fully processed (checkpointing per module).
6. THE Runner SHALL print a summary at the end: total entries generated, passed validation, failed validation, per-format counts.
