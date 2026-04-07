# Implementation Plan: AQG Dataset Pipeline

## Overview

Implementasi pipeline persiapan dataset dari Markdown ke JSONL, mengikuti arsitektur modular di design.md. Setiap task membangun di atas task sebelumnya, dimulai dari komponen paling fundamental (Chunker) hingga pipeline end-to-end.

## Tasks

- [x] 1. Setup struktur project dan dependencies
  - Buat folder `src/dataset/` dan `tests/` di root project
  - Buat `src/__init__.py`, `src/dataset/__init__.py`, dan `tests/__init__.py`
  - Tambahkan `hypothesis`, `pytest`, `langchain-openai`, `python-dotenv` ke `requirements.txt`
  - LLM provider: OpenRouter via `langchain_openai.ChatOpenAI` (sudah ada di `.env`)
  - _Requirements: semua_

- [x] 2. Implementasi Chunker
  - [x] 2.1 Implementasi `Chunk` dataclass dan fungsi `chunk_markdown()`
    - Buat `src/dataset/chunker.py`
    - Implementasi split berdasarkan heading (`#`, `##`, `###`)
    - Implementasi split di batas kalimat jika section > 400 token
    - Preserve code block utuh (jangan split di tengah ` ```python ... ``` `)
    - Attach metadata: `source_file`, `section_heading`, `token_count`, `has_code`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property tests untuk Chunker
    - **Property 1: Chunk token count invariant** — for any Markdown, all chunks <= 400 tokens
    - **Property 2: Code block integrity** — for any chunk with opening ```, closing ``` must exist
    - **Property 3: Chunk metadata completeness** — all chunks have non-empty source_file, section_heading, token_count > 0
    - **Validates: Requirements 1.1, 1.2, 1.4, 1.5**

  - [x] 2.3 Implementasi `chunk_all_materials()`
    - Iterasi semua file `.md` secara rekursif di direktori materi
    - Kembalikan flat list of Chunk dari semua file
    - _Requirements: 1.1_

- [x] 3. Implementasi Prompt Constructor
  - [x] 3.1 Implementasi `TaskParams`, `PromptInput`, dan `build_prompt()`
    - Buat `src/dataset/prompt_constructor.py`
    - Implementasi `PROMPT_TEMPLATE` sesuai design
    - Validasi `difficulty` dan `question_type` di constructor
    - Preserve code block formatting dalam output input string
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property tests untuk Prompt Constructor
    - **Property 4: Prompt template conformance** — for any chunk + params, input contains "Konteks:", "Prompt:", concept, difficulty, question_type
    - **Property 5: Code preservation in prompt** — for any chunk with code block, input string contains same code unchanged
    - **Validates: Requirements 2.1, 2.2, 2.5**

  - [x] 3.3 Buat `src/dataset/concept_list.py`
    - Definisikan `CONCEPTS` dict dengan konsep per modul sesuai design
    - _Requirements: 4.1_

- [x] 4. Checkpoint — Pastikan semua tests pass
  - Jalankan `pytest dataset_aqg/tests/ -v`
  - Pastikan Chunker dan Prompt Constructor berfungsi dengan materi nyata di `dataset_aqg/materi/`

- [x] 5. Implementasi Validator
  - [x] 5.1 Implementasi `ValidationResult`, `ValidDataPoint`, dan `validate()`
    - Buat `src/dataset/validator.py`
    - Cek panjang input: 50–600 token
    - Cek target mengandung "Pertanyaan:", "Jawaban benar:", "Distraktor:"
    - Cek metadata memiliki semua required fields
    - Cek nilai enum untuk `difficulty` dan `question_type`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 5.2 Write property tests untuk Validator
    - **Property 7: Metadata schema validity** — for any valid data point, metadata has all required keys with valid enum values
    - **Property 10: Validator rejects invalid inputs** — for any input with token count < 50 or > 600, is_valid = False
    - **Validates: Requirements 4.1, 4.3, 4.4, 6.1**

  - [x] 5.3 Implementasi `validate_batch()` dan validation report
    - Kembalikan `(valid_list, failure_log)`
    - Log failure ke list of dict dengan field `reason` dan `raw_datapoint`
    - _Requirements: 6.4, 6.5_

- [x] 6. Implementasi Dataset Writer
  - [x] 6.1 Implementasi `write_dataset()`
    - Buat `src/dataset/dataset_writer.py`
    - Split train/val/test dengan ratio 70/15/15
    - Stratifikasi berdasarkan `difficulty`
    - Simpan setiap split sebagai JSONL (satu JSON object per baris)
    - Simpan `dataset_info.json` dengan statistik
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 6.2 Write property tests untuk Dataset Writer
    - **Property 6: Target format invariant** — for any valid data point, target is string containing all three section markers
    - **Property 8: JSONL round-trip consistency** — write then load produces equivalent data with correct types
    - **Property 9: Split stratification** — each split contains at least one entry per difficulty level
    - **Validates: Requirements 3.1, 3.2, 5.1, 5.3, 5.4**

- [x] 7. Implementasi Synthetic Generator
  - [x] 7.1 Implementasi `RawDataPoint` dan `generate_datapoint()`
    - Buat `src/dataset/synthetic_generator.py`
    - Support OpenRouter via `langchain_openai.ChatOpenAI` (sesuai `.env`)
    - Implementasi retry logic (max 2 retries) dengan exponential backoff
    - Tag output dengan `"source": "synthetic"` di metadata
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write unit tests untuk Synthetic Generator
    - Test retry logic dengan mock LLM yang gagal N kali
    - Test bahwa output di-tag dengan `source: synthetic`
    - Test parsing response LLM yang valid dan tidak valid
    - _Requirements: 7.3, 7.4, 7.5_

- [x] 8. Checkpoint — Pastikan semua tests pass
  - Jalankan `pytest dataset_aqg/tests/ -v`
  - Test pipeline end-to-end dengan 2–3 file Markdown nyata (tanpa LLM, gunakan mock)

- [x] 9. Perbaikan kualitas dataset (berdasarkan evaluasi)
  - [x] 9.1 Perbaiki context grounding di Prompt Constructor
    - Update logika pemilihan `concept` agar hanya menggunakan konsep yang relevan dengan isi chunk
    - Tambahkan fungsi `extract_concept_from_chunk(chunk, candidate_concepts)` yang memilih konsep paling relevan berdasarkan keyword matching antara teks chunk dan nama konsep
    - Jika tidak ada konsep yang relevan ditemukan, gunakan heading section sebagai fallback concept
    - Normalisasi `source_file` path menggunakan `pathlib.PurePosixPath` agar selalu menggunakan forward slash
    - _Requirements: 2.6, 4.5_

  - [x] 9.2 Revisi `GENERATION_SYSTEM_PROMPT` untuk misconception tags dan context grounding
    - Update `GENERATION_SYSTEM_PROMPT` di `src/dataset/synthetic_generator.py` sesuai design.md yang baru
    - Tambahkan instruksi eksplisit: soal hanya boleh dibuat dari informasi dalam konteks
    - Tambahkan instruksi untuk menghasilkan `misconception_tags` non-empty
    - Update parser response LLM untuk mengekstrak `misconception_tags` dari output
    - _Requirements: 7.6, 7.7_

  - [x] 9.3 Update Validator untuk aturan validasi baru
    - Tambahkan cek: `metadata.misconception_tags` harus non-empty list
    - Tambahkan: set `metadata.validated = True` setelah data point lolos semua validasi
    - _Requirements: 6.6, 6.7_

  - [ ]* 9.4 Write property tests untuk perbaikan kualitas
    - **Property 12: Misconception tags non-empty** — for any valid data point, misconception_tags is a non-empty list
    - **Property 13: Validated flag set** — for any data point passing validation, metadata.validated is True
    - **Property 14: Source file path normalization** — for any valid data point, source_file uses forward slashes
    - **Validates: Requirements 4.5, 4.6, 4.7, 6.6, 6.7**

- [ ] 10. Implementasi Augmentor (opsional)
  - [ ]* 10.1 Implementasi `augment_by_difficulty()` dan `deduplicate()`
    - Buat `src/dataset/augmentor.py`
    - Generate varian easy/medium/hard dari satu data point
    - Tag dengan `"source": "augmented"` dan `"augmented_from": "{original_id}"`
    - _Requirements: 8.1, 8.2, 8.4_

  - [ ]* 10.2 Write property test untuk Augmentor
    - **Property 11: Augmentation deduplication** — after dedup, no two entries have identical input strings
    - **Validates: Requirements 8.5**

- [x] 11. Buat pipeline runner script
  - Buat `dataset_aqg/run_pipeline.py` sebagai entry point
  - Terima argumen: `--materi-dir`, `--output-dir`, `--max-per-chunk`, `--llm-provider`, `--section`
  - Jalankan seluruh pipeline: chunk → prompt → generate → validate → write
  - Implementasi checkpointing: simpan progress ke `output_dir/checkpoint.jsonl` setelah setiap section selesai
  - WHEN pipeline dilanjutkan, THE Runner SHALL skip section yang sudah ada di checkpoint
  - Tampilkan progress bar dengan `tqdm`
  - Simpan `validation_failures.jsonl` di output dir
  - _Requirements: semua_

- [x] 12. Final checkpoint — Jalankan pipeline dengan materi nyata
  - Jalankan pipeline pada minimal 2 modul materi (`01-Berkenalan-dengan-python`)
  - Verifikasi output JSONL bisa di-load dengan `datasets.load_dataset("json", ...)`
  - Pastikan semua tests pass: `pytest dataset_aqg/tests/ -v`

## Notes

- Tasks bertanda `*` adalah opsional (tests) — bisa dilewati untuk MVP lebih cepat
- Setiap property test harus diberi komentar: `# Feature: aqg-dataset-pipeline, Property N: <text>`
- LLM API key disimpan di `.env`, tidak pernah di-commit ke repo
- Token count menggunakan estimasi `len(text.split()) * 1.3`, bukan tokenizer penuh (cukup untuk chunking)
- Format target adalah plain string — jangan ubah ke JSON object
