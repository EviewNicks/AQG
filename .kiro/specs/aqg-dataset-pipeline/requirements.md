# Requirements Document

## Introduction

Pipeline persiapan dataset untuk proyek AQG (Automatic Question Generation) Python berbahasa Indonesia. Pipeline ini mengubah materi kursus Python Basics dalam format Markdown menjadi pasangan data text-to-text (seq2seq) berkualitas tinggi, siap digunakan untuk fine-tuning model IndoT5 + LoRA.

Scope: Hanya mencakup tahap persiapan dataset — dari raw Markdown sampai file JSONL tervalidasi. Integrasi API, fine-tuning model, dan deployment berada di luar scope ini.

## Glossary

- **Pipeline**: Sistem pemrosesan data modular dari Markdown ke JSONL.
- **Chunk**: Potongan teks materi berukuran 250–400 token berdasarkan sub-heading atau learning objective.
- **Pasangan Data**: Satu unit dataset terdiri dari `input` (string), `target` (string), dan `metadata` (JSON object).
- **Input**: Prompt lengkap ke model — gabungan konteks materi + instruksi tugas.
- **Target**: Output yang diharapkan model — plain string berisi pertanyaan + jawaban benar + distraktor.
- **Metadata**: Informasi tambahan (difficulty, concept, misconception_tags, question_type) — tidak masuk ke model saat training.
- **Distraktor**: Pilihan jawaban salah yang plausible dan pedagogis, berdasarkan miskonsepsi umum siswa.
- **MCQ**: Multiple Choice Question — soal pilihan ganda.
- **Code Completion**: Soal melengkapi kode Python yang tidak lengkap.
- **JSONL**: JSON Lines — format file dengan satu JSON object per baris.
- **Seed Data**: Pasangan data awal yang dibuat manual atau dari sumber publik, sebelum augmentasi.
- **Augmentasi**: Teknik memperbanyak data dari seed yang sudah ada (paraphrasing, variasi difficulty, dll).
- **Master Concept List**: Daftar referensi konsep Python yang digunakan sebagai nilai `metadata.concept`.
- **Chunker**: Komponen yang memotong file Markdown menjadi chunk-chunk.
- **Validator**: Komponen yang memvalidasi format dan kualitas pasangan data.
- **Concept Extractor**: Sub-komponen yang mengidentifikasi konsep utama dari sebuah chunk secara otomatis.
- **Context Grounding**: Prinsip bahwa soal yang dihasilkan harus sepenuhnya bersumber dari teks konteks yang diberikan, bukan dari pengetahuan internal LLM.
- **Misconception Tag**: Label yang mengidentifikasi miskonsepsi umum siswa yang ditargetkan oleh sebuah distraktor.

## Requirements

### Requirement 1: Chunking Materi Markdown

**User Story:** As a researcher, I want to automatically split Markdown course files into context chunks, so that I can use them as input context for dataset generation.

#### Acceptance Criteria

1. WHEN a Markdown file is provided, THE Chunker SHALL split it into chunks of 250–400 tokens based on sub-headings and paragraph boundaries.
2. WHEN a chunk contains a Python code block, THE Chunker SHALL preserve the code block intact within the chunk without splitting it mid-block.
3. WHEN a heading is encountered, THE Chunker SHALL start a new chunk at that heading boundary.
4. IF a paragraph exceeds 400 tokens, THEN THE Chunker SHALL split it at the nearest sentence boundary without exceeding the token limit.
5. THE Chunker SHALL attach metadata to each chunk: source file path, section heading, and estimated token count.

### Requirement 2: Prompt Construction

**User Story:** As a researcher, I want to construct structured prompts from chunks, so that the prompts are consistent and suitable for fine-tuning IndoT5.

#### Acceptance Criteria

1. WHEN a chunk and task parameters (concept, difficulty, question_type) are provided, THE Prompt_Constructor SHALL produce a single `input` string combining context and instruction.
2. THE Prompt_Constructor SHALL use a fixed template format: `"Konteks: {context}\n\nPrompt: Buat satu soal {question_type} tentang {concept}, tingkat kesulitan: {difficulty}, bahasa Indonesia."`.
3. THE Prompt_Constructor SHALL support two question types: `MCQ` and `Code Completion`.
4. THE Prompt_Constructor SHALL support three difficulty levels: `easy`, `medium`, `hard`.
5. WHEN the context contains Python code, THE Prompt_Constructor SHALL preserve the code formatting in the input string.
6. WHEN assigning a concept to a chunk, THE Prompt_Constructor SHALL only use concepts that are present or directly inferable from the chunk text, not concepts from the Master Concept List that are absent from the chunk.
7. THE Pipeline SHALL distribute `difficulty` values across generated data points such that each difficulty level (`easy`, `medium`, `hard`) is represented in the final dataset.

### Requirement 3: Target Format

**User Story:** As a researcher, I want targets stored as plain text strings, so that they are directly usable as training labels for the seq2seq model.

#### Acceptance Criteria

1. THE Dataset SHALL store `target` as a plain string, not as a nested JSON object.
2. THE Target_Format SHALL follow this structure: `"Pertanyaan: {question}? Jawaban benar: {answer}. Distraktor: 1) {d1} 2) {d2} 3) {d3} 4) {d4}"`.
3. WHEN a Code Completion question is stored, THE Target_Format SHALL use: `"Pertanyaan: {question} Jawaban benar: {answer}. Distraktor: 1) {d1} 2) {d2} 3) {d3}"`.
4. THE Dataset SHALL store `metadata` as a separate JSON object column, not embedded inside `target`.

### Requirement 4: Metadata Schema

**User Story:** As a researcher, I want consistent metadata for every data point, so that I can filter, analyze, and validate the dataset systematically.

#### Acceptance Criteria

1. THE Metadata SHALL contain these required fields: `difficulty`, `question_type`, `concept`, `misconception_tags`.
2. THE Metadata SHALL contain these optional fields: `source_file`, `section`, `plausibility_score`, `validated`.
3. WHEN `difficulty` is set, THE Validator SHALL accept only values: `easy`, `medium`, `hard`.
4. WHEN `question_type` is set, THE Validator SHALL accept only values: `MCQ`, `Code Completion`.
5. THE Metadata SHALL include a `source_file` field referencing the original Markdown file path using forward slashes (`/`) regardless of operating system.
6. WHEN a data point is generated by the Synthetic_Generator, THE Metadata SHALL include a non-empty `misconception_tags` list containing at least one tag identifying the misconception targeted by the distractors.
7. WHEN a data point passes validation, THE Metadata SHALL set `validated` to `true`.

### Requirement 5: Dataset Output Format

**User Story:** As a researcher, I want the dataset saved as JSONL, so that it is directly loadable by Hugging Face `datasets` library for fine-tuning.

#### Acceptance Criteria

1. THE Pipeline SHALL save the final dataset as JSONL files with one JSON object per line.
2. THE Pipeline SHALL produce three split files: `train.jsonl` (70%), `validation.jsonl` (15%), `test.jsonl` (15%).
3. WHEN splitting, THE Pipeline SHALL stratify by `difficulty` to ensure balanced distribution across splits.
4. EACH line in the JSONL file SHALL contain exactly three keys: `input`, `target`, `metadata`.
5. THE Pipeline SHALL also save a `dataset_info.json` file containing: total count, split counts, concept distribution, and difficulty distribution.

### Requirement 6: Validation

**User Story:** As a researcher, I want each data point validated before saving, so that the dataset maintains high quality standards.

#### Acceptance Criteria

1. WHEN a data point is validated, THE Validator SHALL check that `input` length is between 50 and 600 tokens.
2. WHEN a data point is validated, THE Validator SHALL check that `target` contains all required sections: "Pertanyaan:", "Jawaban benar:", "Distraktor:".
3. WHEN a data point is validated, THE Validator SHALL check that `metadata` contains all required fields.
4. IF a data point fails validation, THEN THE Validator SHALL log the failure reason and skip the invalid entry.
5. THE Validator SHALL produce a validation report showing: total processed, passed, failed, and failure reasons.
6. WHEN a data point is validated, THE Validator SHALL check that `metadata.misconception_tags` is a non-empty list.
7. WHEN a data point passes all validation checks, THE Validator SHALL set `metadata.validated` to `true` before the data point is written to the dataset.

### Requirement 7: Synthetic Data Generation

**User Story:** As a researcher, I want to generate synthetic data using an LLM (GPT-4o/Claude), so that I can scale the dataset beyond what manual annotation allows.

#### Acceptance Criteria

1. WHEN a chunk and task parameters are provided, THE Synthetic_Generator SHALL call an LLM API to generate a complete data point (question + answer + distractors).
2. THE Synthetic_Generator SHALL use a structured prompt that instructs the LLM to output in the exact target format.
3. WHEN the LLM response is received, THE Synthetic_Generator SHALL parse and validate it before adding to the dataset.
4. IF the LLM response fails validation, THEN THE Synthetic_Generator SHALL retry up to 2 times before skipping.
5. THE Synthetic_Generator SHALL tag generated entries with `"source": "synthetic"` in metadata.
6. THE Synthetic_Generator SHALL instruct the LLM via system prompt to generate `misconception_tags` that identify the specific misconceptions targeted by each distractor.
7. THE Synthetic_Generator SHALL instruct the LLM to only generate questions based on information explicitly present in the provided context, not from external knowledge.

### Requirement 8: Data Augmentation

**User Story:** As a researcher, I want to augment existing seed data, so that I can reach the target of 1,500–3,000 data pairs from a smaller seed set.

#### Acceptance Criteria

1. WHEN augmentation is run on a seed entry, THE Augmentor SHALL produce at least one variant by paraphrasing the context in `input`.
2. THE Augmentor SHALL support difficulty variation: generating easy/medium/hard variants from the same context.
3. WHEN paraphrasing, THE Augmentor SHALL preserve all Python code blocks unchanged.
4. THE Augmentor SHALL tag augmented entries with `"source": "augmented"` and `"augmented_from": "{original_id}"` in metadata.
5. THE Augmentor SHALL not produce duplicate entries — entries with identical `input` strings SHALL be deduplicated.
