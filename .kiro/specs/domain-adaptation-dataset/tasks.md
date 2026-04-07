# Implementation Plan: Domain Adaptation Dataset Pipeline

## Overview

Implementasi pipeline persiapan dataset domain adaptation dari Markdown ke JSONL. Komponen Chunker di-reuse dari `aqg-dataset-pipeline` tanpa modifikasi. Komponen baru ditempatkan di `src/domain/`. Setiap task membangun di atas task sebelumnya.

## Tasks

- [ ] 1. Setup struktur modul domain
  - Buat folder `src/dataset/step1/` dan file `src/dataset/step1/__init__.py`
  - Verifikasi import `from src.dataset.chunker import Chunk, chunk_markdown, chunk_all_materials` berjalan
  - Tambahkan dependency yang dibutuhkan ke `requirements.txt` jika belum ada
  - _Requirements: semua_

- [ ] 2. Implementasi SpanCorruptor
  - [ ] 2.1 Implementasi `corrupt_spans()` di `src/dataset/step1/formatter.py`
    - Buat `src/dataset/step1/formatter.py` dengan dataclass `RawDomainDataPoint`
    - Implementasi tokenisasi sederhana (split by whitespace, konsisten dengan Chunker)
    - Implementasi pemilihan span acak: panjang 2–5 token, tidak overlap, tidak di dalam code block
    - Ganti span dengan sentinel `<extra_id_N>`, bangun `input` dan `target` string
    - Tag metadata dengan `"format": "span_corruption"`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.2 Write property tests untuk SpanCorruptor
    - **Property 3: Span corruption sentinel balance** — count_sentinels(input) == count_sentinels(target) - 1
    - **Property 4: Span corruption rate invariant** — masked tokens antara 1 dan ceil(token_count * 0.25)
    - **Property 5: Code block not masked** — code block text tidak berubah di input setelah corruption
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [ ] 3. Implementasi QAGenerator
  - [ ] 3.1 Implementasi `extract_qa_pairs()` di `src/dataset/step1/formatter.py`
    - Ekstrak bold terms (`**term**`) → generate `"Apa itu {term} dalam Python?"`
    - Ekstrak inline code terms (`` `term` ``) → generate `"Jelaskan {term} dalam Python."`
    - Ekstrak heading text → generate `"Apa yang dimaksud dengan {heading}?"`
    - Gunakan kalimat yang mengandung term sebagai target answer
    - Deduplikasi jika term muncul di bold dan inline code sekaligus
    - Tag metadata dengan `"format": "qa_generic"`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 3.2 Write property tests untuk QAGenerator
    - **Property 7: QA term presence** — term di input question muncul di target answer
    - **Validates: Requirements 4.3**

- [ ] 4. Checkpoint — Pastikan SpanCorruptor dan QAGenerator berfungsi
  - Jalankan `pytest tests/ -v -k "domain"`
  - Test manual dengan satu file Markdown nyata dari `dataset_aqg/materi/`

- [ ] 5. Implementasi Summarizer
  - [ ] 5.1 Implementasi `summarize_chunk()` di `src/dataset/step1/formatter.py`
    - Tambahkan `SUMMARIZATION_SYSTEM_PROMPT` sesuai design
    - Implementasi panggilan LLM via `langchain_openai.ChatOpenAI` (OpenRouter)
    - Format input: `"Rangkum teks berikut:\n\n{chunk.text}"`
    - Implementasi retry logic: max 2 retries dengan exponential backoff (1s, 2s)
    - Skip chunk dengan token_count < 100
    - Tag metadata dengan `"format": "summarization"`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 5.2 Write unit tests untuk Summarizer
    - Test bahwa input selalu diawali `"Rangkum teks berikut:"`
    - Test retry logic dengan mock LLM yang gagal 1x lalu berhasil
    - Test skip chunk < 100 token
    - _Requirements: 3.1, 3.4, 3.6_

- [ ] 6. Implementasi Domain Validator
  - [ ] 6.1 Implementasi `validate_domain()` dan `validate_domain_batch()` di `src/dataset/step1/validator.py`
    - Buat `src/dataset/step1/validator.py` dengan dataclass `DomainValidationResult` dan `ValidDomainDataPoint`
    - Cek panjang input: 10–1024 token
    - Cek target non-empty (minimal 5 karakter)
    - Cek metadata memiliki field: `format`, `source_file`, `module_name`
    - Cek enum `format`: hanya `"span_corruption"`, `"summarization"`, `"qa_generic"`
    - `validate_domain_batch()` kembalikan `(valid_list, failure_log)`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 6.2 Write property tests untuk Domain Validator
    - **Property 10: Validator rejects out-of-range inputs** — input < 10 atau > 1024 token → is_valid = False
    - **Property 11: Metadata format enum validity** — valid data point selalu punya format yang valid
    - **Validates: Requirements 6.1, 6.4**

- [ ] 7. Implementasi Domain Dataset Writer
  - [ ] 7.1 Implementasi `write_domain_dataset()` di `src/dataset/step1/dataset_writer.py`
    - Buat `src/dataset/step1/dataset_writer.py`
    - Split train/val/test dengan ratio 80/10/10
    - Stratifikasi berdasarkan `format` (pastikan ketiga format ada di setiap split)
    - Simpan setiap split sebagai JSONL (satu JSON object per baris, tiga key: `input`, `target`, `metadata`)
    - Simpan `dataset_info.json` dengan: total, splits, format_distribution, module_distribution, generated_at
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 7.2 Write property tests untuk Domain Dataset Writer
    - **Property 8: JSONL round-trip consistency** — write then load menghasilkan data equivalent dengan tipe yang benar
    - **Property 9: Split stratification by format** — setiap split mengandung minimal satu entry per format
    - **Validates: Requirements 5.1, 5.3, 5.4**

- [ ] 8. Checkpoint — Pastikan semua tests pass
  - Jalankan `pytest tests/ -v -k "domain"`
  - Test pipeline parsial: chunk → corrupt_spans + extract_qa_pairs → validate → write (tanpa LLM)

- [ ] 9. Buat Pipeline Runner
  - [ ] 9.1 Implementasi `dataset_aqg/run_domain_pipeline.py`
    - Terima argumen CLI: `--materi-dir`, `--output-dir`, `--formats`, `--max-per-chunk`
    - Jalankan pipeline: chunk → format (sesuai --formats) → validate → write
    - Implementasi checkpointing per modul: simpan ke `output_dir/checkpoint.json`
    - Skip modul yang sudah ada di checkpoint saat restart
    - Tampilkan progress bar dengan `tqdm` (per modul dan per format)
    - Simpan `validation_failures.jsonl` di output dir
    - Print summary di akhir: total generated, passed, failed, per-format counts
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 10. Final checkpoint — Jalankan pipeline dengan materi nyata
  - Jalankan pipeline pada minimal 2 modul dengan format `span_corruption,qa_generic` (tanpa LLM)
  - Verifikasi output JSONL bisa di-load dengan `datasets.load_dataset("json", ...)`
  - Pastikan semua tests pass: `pytest tests/ -v -k "domain"`

## Notes

- Tasks bertanda `*` adalah opsional (tests) — bisa dilewati untuk MVP lebih cepat
- Setiap property test harus diberi komentar: `# Feature: domain-adaptation-dataset, Property N: <text>`
- Komponen baru ditempatkan di `src/dataset/step1/` — konsisten dengan struktur AQG pipeline di `src/dataset/`
- Format `summarization` membutuhkan LLM; gunakan `--formats span_corruption,qa_generic` untuk zero-cost run
- Token count menggunakan estimasi `len(text.split()) * 1.3`, konsisten dengan AQG pipeline
