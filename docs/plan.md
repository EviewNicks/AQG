# Rencana Pengembangan Sistem AQG Python

## Gambaran Besar

Proyek ini membangun sistem **Automatic Question Generation (AQG)** berbasis IndoNanoT5 + LoRA untuk menghasilkan soal kuis pemrograman Python secara otomatis dalam bahasa Indonesia. Sistem dirancang untuk platform pembelajaran coding adaptif yang menargetkan siswa tingkat menengah.

Pendekatan yang dipilih adalah **hybrid 2-tahap fine-tuning**:

```
IndoNanoT5-base
      │
      ▼ Tahap 1: Domain Adaptation
IndoT5-Python
      │
      ▼ Tahap 2: Task-Specific Fine-tuning (AQG)
IndoT5-Python-AQG
      │
      ▼
API Endpoint → Frontend
```

---

## Tahap 1: Domain Adaptation — IndoT5-Python

### Tujuan
Mengajarkan IndoNanoT5-base pemahaman mendalam tentang materi Python Basics berbahasa Indonesia sebelum diajarkan tugas spesifik AQG. Ini mengurangi hallucination dan meningkatkan relevansi kontekstual output model.

### Sumber Data
Materi kursus Python Basics di `dataset_aqg/materi/` — 11 modul, masing-masing berisi 3–6 lesson + 1 file rangkuman:

| Modul | Topik |
|---|---|
| 01 | Berkenalan dengan Python |
| 02 | Berinteraksi dengan Data |
| 03 | Ekspresi |
| 04 | Aksi Sekuensial |
| 05 | Control Flow |
| 06 | Array |
| 07 | Matriks |
| 08 | Subprogram |
| 09 | OOP |
| 10 | Style Guide |
| 11 | Unit Testing |

### Format Dataset (3 Format Kombinasi)

**Format A — Summarization** (zero-cost, data sudah tersedia)

Pasangan: semua lesson dalam satu modul → rangkuman modul.

```json
{
  "input": "ringkas: # Pengenalan Python\n\nPython adalah bahasa pemrograman multifungsi...",
  "target": "Python dirilis 1991 oleh Guido van Rossum, bersifat readable..."
}
```

Estimasi: ~50 pasang (per-modul dan per-section).

**Format B — Cloze / Fill-in-the-blank** (rule-based + GPT-4o)

Mask kata kunci (angka, nama, istilah teknis) dari chunk materi.

```json
{
  "input": "lengkapi: Python dirilis tahun ___ oleh ___. Bahasa ini tidak memerlukan ___ di akhir baris.",
  "target": "1991 | Guido van Rossum | titik koma"
}
```

Estimasi: ~200 pasang.

**Format C — Simple QA** (GPT-4o, 3–5 QA per chunk)

```json
{
  "input": "jawab: Apa perbedaan Python 2.x dan Python 3.x?",
  "target": "Python 3.x tidak bersifat backward-compatible dengan Python 2.x..."
}
```

Estimasi: ~400 pasang.

### Target Volume
~650 pasang total — cukup untuk domain adaptation IndoNanoT5-base dengan LoRA pada domain yang sangat spesifik ini.

### Split
- Train: 70% (~455)
- Validation: 15% (~98)
- Test: 15% (~98)

---

## Tahap 2: Task-Specific Fine-tuning — IndoT5-Python-AQG

### Tujuan
Mengajarkan IndoT5-Python (hasil Tahap 1) untuk menghasilkan soal MCQ dan Code Completion lengkap dengan jawaban benar dan 3–4 distraktor pedagogis.

### Status
Pipeline dataset untuk Tahap 2 **sudah selesai dibangun** di `.kiro/specs/aqg-dataset-pipeline`. Dataset `accumulated.jsonl` sudah tersedia di `dataset_aqg/output_modul/` dan `dataset_aqg/output_2modul/`.

### Format Dataset
```json
{
  "input": "Konteks: <chunk materi>\n\nPrompt: Buat satu soal MCQ tentang <konsep>, tingkat kesulitan: <easy|medium|hard>, bahasa Indonesia.",
  "target": "Pertanyaan: <pertanyaan>? Jawaban benar: <jawaban>. Distraktor: 1) <d1> 2) <d2> 3) <d3> 4) <d4>",
  "metadata": {
    "difficulty": "easy|medium|hard",
    "question_type": "MCQ|Code Completion",
    "concept": "<konsep>",
    "misconception_tags": ["<tag1>", "<tag2>"]
  }
}
```

### Target Volume
1.500–3.000 pasang (setelah augmentasi).

---

## Roadmap Pengerjaan

### Fase yang Sudah Selesai ✅
- [x] Pipeline dataset Tahap 2 (AQG) — `aqg-dataset-pipeline` spec
- [x] Dataset awal Tahap 2 tersedia di `dataset_aqg/output_modul/`

### Fase Berikutnya
- [ ] **Spec baru: `domain-adaptation-dataset`** — pipeline untuk membangun dataset Tahap 1
  - Format A: Summarization dari file yang sudah ada (otomatis)
  - Format B: Cloze generation (rule-based)
  - Format C: Simple QA generation (via GPT-4o)
  - Output: `dataset_aqg/output_domain/` dalam format JSONL

- [ ] **Fine-tuning Tahap 1** — Domain Adaptation IndoNanoT5-base → IndoT5-Python
  - LoRA: rank=8, alpha=16, dropout=0.1
  - Data: ~650 pasang dari pipeline domain adaptation

- [ ] **Perbaikan dataset Tahap 2** — Task 9 di `aqg-dataset-pipeline`
  - Context grounding
  - Misconception tags
  - Variasi difficulty

- [ ] **Fine-tuning Tahap 2** — IndoT5-Python → IndoT5-Python-AQG
  - Base model: hasil Tahap 1
  - Data: 1.500–3.000 pasang AQG

- [ ] **Evaluasi** — BLEU-4, ROUGE-L, BERTScore + human evaluation

---

## Keputusan Teknis

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Base model | IndoNanoT5-base | Benchmark IndoNLG tertinggi (71.89), monolingual Indonesia, lebih sedikit hallucination |
| Fine-tuning strategy | LoRA (rank=8) | Efisien secara komputasi, mengurangi catastrophic forgetting |
| Domain adaptation format | Summarization + Cloze + QA | Memanfaatkan data yang sudah ada, sesuai T5 pretraining objective |
| LLM untuk generate | GPT-4o via OpenRouter | Kualitas tinggi, sudah terintegrasi di pipeline existing |
| Output format | JSONL (Hugging Face compatible) | Langsung loadable dengan `datasets` library |
| Scope Tahap 1 | AQG saja (bukan explain code) | Fokus untuk penelitian ini; explain code jadi roadmap berikutnya |
