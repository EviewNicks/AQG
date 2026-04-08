# Deskripsi dan Akuisisi Dataset

## 1. Pendahuluan

Penelitian ini mengembangkan sistem *Automatic Question Generation* (AQG) berbahasa Indonesia untuk materi pemrograman Python menggunakan model IndoT5 dengan teknik *fine-tuning* LoRA. Dalam rangka membangun sistem yang tidak hanya mampu menghasilkan soal kuis secara otomatis, tetapi juga memiliki pemahaman mendalam terhadap domain pendidikan Python, penelitian ini mengadopsi pendekatan *fine-tuning* dua tahap (*two-stage fine-tuning*) yang bersifat hibrida.

Pendekatan hibrida ini didasarkan pada pertimbangan bahwa *task-specific fine-tuning* secara langsung tanpa fondasi pemahaman domain yang kuat berpotensi menghasilkan soal yang tidak *grounded* pada konteks materi — model cenderung mengandalkan pengetahuan umumnya daripada informasi yang tersedia dalam teks yang diberikan. Oleh karena itu, proses persiapan dataset dibagi menjadi dua tahap yang saling melengkapi:

- **Tahap 1 — Domain Adaptation**: Membangun pemahaman model terhadap domain pendidikan Python berbahasa Indonesia melalui dataset korpus materi.
- **Tahap 2 — Task-Specific AQG**: Melatih model untuk menghasilkan soal kuis terstruktur (pertanyaan, jawaban benar, distraktor) dari konteks materi yang diberikan.

Kedua dataset disiapkan secara otomatis menggunakan pipeline modular yang dikembangkan khusus untuk penelitian ini, dengan sumber data utama berupa materi kursus Python Basics dalam format Markdown.

---

## 2. Sumber Data

Sumber data utama penelitian ini adalah materi kursus **"Memulai Pemrograman dengan Python"** yang terdiri dari 11 modul dan 55+ *lesson* dalam format Markdown. Materi ini mencakup topik-topik fundamental pemrograman Python berbahasa Indonesia, mulai dari pengenalan bahasa hingga konsep lanjutan seperti OOP dan *unit testing*.

**Struktur materi:**

| Modul | Topik |
|-------|-------|
| 01 | Berkenalan dengan Python |
| 02 | Berinteraksi dengan Data |
| 03 | Ekspresi |
| 04 | Aksi Sekuensial |
| 05 | Control Flow |
| 06 | Array |
| 07 | Matriks |
| 08 | Subprogram |
| 09 | Object-Oriented Programming |
| 10 | Style Guide |
| 11 | Unit Testing |

Setiap file Markdown berisi teks penjelasan konsep, contoh kode Python, dan ilustrasi penggunaan. Format ini dipilih karena memungkinkan pemrosesan otomatis yang terstruktur — heading Markdown (`#`, `##`, `###`) digunakan sebagai penanda batas antar topik, sementara *code block* (` ```python ... ``` `) diidentifikasi dan dipertahankan keutuhannya selama proses *chunking*.

---

## 3. Arsitektur Pipeline Persiapan Dataset

Sebelum menjelaskan masing-masing dataset, penting untuk memahami alur umum persiapan data yang diterapkan pada kedua tahap. Kedua pipeline berbagi komponen **Chunker** yang sama, namun berbeda pada tahap transformasi dan format output-nya.

```
Materi Markdown (11 Modul)
         │
         ▼
    ┌─────────────┐
    │   Chunker   │  → memotong teks menjadi chunk berdasarkan heading
    └──────┬──────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
[Tahap 1]     [Tahap 2]
Domain        AQG Pipeline
Adaptation    (Prompt Constructor
Pipeline      → LLM Generator
(Formatter    → Validator
→ Validator   → Dataset Writer)
→ Writer)
     │            │
     ▼            ▼
output_domain/ output_modul/
(JSONL)        (JSONL per modul)
```

**Chunker** bekerja dengan cara memotong setiap file Markdown menjadi potongan teks (*chunk*) berdasarkan batas *heading* dan paragraf. Setiap *chunk* dilengkapi metadata: `source_file`, `section_heading`, `token_count`, dan `has_code`. Estimasi jumlah token menggunakan pendekatan sederhana: `len(text.split()) * 1.3`.

---

## 4. Dataset Tahap 1: Domain Adaptation

### 4.1 Tujuan dan Motivasi

Dataset domain adaptation bertujuan untuk membekali model IndoT5 dengan pemahaman mendalam terhadap terminologi, gaya bahasa, dan konsep-konsep dalam domain pendidikan Python berbahasa Indonesia. Tanpa tahap ini, model yang langsung di-*fine-tune* pada dataset AQG berisiko menghasilkan soal yang tidak relevan secara kontekstual atau mengandung informasi yang tidak bersumber dari materi yang diberikan (*hallucination*).

### 4.2 Format Data

Dataset domain adaptation menggunakan dua format data yang saling melengkapi, keduanya mengikuti skema *text-to-text* yang kompatibel dengan arsitektur T5:

**a. Span Corruption (Format T5 Pre-training)**

Format ini mengadaptasi teknik *masked language modeling* gaya T5 original. Sebagian token dalam teks di-*mask* menggunakan *sentinel token* (`<extra_id_0>`, `<extra_id_1>`, dst.), dan model belajar merekonstruksi span yang hilang tersebut. Tingkat *masking* sebesar 15% dari total token, dengan panjang setiap span 2–5 token.

Contoh:

```json
{
  "input": "Python adalah <extra_id_0> yang dirilis pada <extra_id_1> oleh Guido van Rossum.",
  "target": "<extra_id_0> bahasa pemrograman <extra_id_1> tahun 1991 <extra_id_2>",
  "metadata": {
    "format": "span_corruption",
    "source_file": "01-Berkenalan-dengan-python/01-perkenalan-pythn.md",
    "module_name": "01-berkenalan-dengan-python",
    "section_heading": "# Pengenalan Python",
    "token_count": 358,
    "has_code": true
  }
}
```

**b. QA Generik (Rule-Based, Zero LLM Cost)**

Format ini mengekstrak pasangan tanya-jawab faktual dari teks materi menggunakan heuristik berbasis aturan, tanpa memerlukan pemanggilan LLM. Istilah kunci diidentifikasi dari: teks *bold* (`**term**`), *inline code* (`` `term` ``), dan teks *heading*. Dari setiap istilah, dihasilkan pertanyaan generik dan kalimat yang mengandung istilah tersebut digunakan sebagai jawaban.

Contoh:

```json
{
  "input": "Apa itu list dalam Python?",
  "target": "List merupakan kumpulan data terurut (ordered sequence) yang dapat menyimpan berbagai tipe data.",
  "metadata": {
    "format": "qa_generic",
    "source_file": "02-berinteraksi-dengan-data/03-type-data.md",
    "module_name": "02-berinteraksi-dengan-data",
    "section_heading": "# List",
    "token_count": 89,
    "has_code": false
  }
}
```

### 4.3 Statistik Dataset

Seluruh 11 modul berhasil diproses oleh pipeline domain adaptation. Dataset yang dihasilkan memiliki karakteristik sebagai berikut:

| Atribut | Nilai |
|---------|-------|
| Total data point | 340 |
| Split train | 271 (80%) |
| Split validation | 33 (10%) |
| Split test | 36 (10%) |
| Format span_corruption | 118 entries (34,7%) |
| Format qa_generic | 222 entries (65,3%) |
| Format summarization | 0 (tidak dijalankan) |

Distribusi per modul:

| Modul | Jumlah |
|-------|--------|
| 01-berkenalan-dengan-python | 47 |
| 02-berinteraksi-dengan-data | 43 |
| 03-ekspresi | 39 |
| 04-aksi-sekuensial | 23 |
| 05-control-flow | 21 |
| 06-array | 27 |
| 07-matriks | 27 |
| 08-subprogram | 32 |
| 09-oop | 26 |
| 10-style-guide | 30 |
| 11-unit-testing | 25 |

> Catatan: Format *summarization* tidak dijalankan pada tahap ini karena memerlukan pemanggilan LLM API. Pipeline dijalankan dalam mode *zero-cost* menggunakan hanya `span_corruption` dan `qa_generic`.

### 4.4 Skema JSONL

Setiap baris dalam file JSONL dataset domain adaptation memiliki tiga kunci utama:

- `input` — string teks input untuk model
- `target` — string teks target yang diharapkan dihasilkan model
- `metadata` — objek JSON berisi: `format`, `source_file`, `module_name`, `section_heading`, `token_count`, `has_code`

---

## 5. Dataset Tahap 2: Task-Specific AQG

### 5.1 Tujuan dan Motivasi

Dataset AQG dirancang untuk melatih model menghasilkan soal kuis terstruktur dalam format *Multiple Choice Question* (MCQ) berbahasa Indonesia. Setiap data point merupakan pasangan *input* (konteks materi + instruksi tugas) dan *target* (soal lengkap beserta jawaban dan distraktor).

Prinsip utama yang diterapkan dalam pembuatan dataset ini adalah **context grounding** — soal yang dihasilkan harus sepenuhnya bersumber dari informasi yang tersedia dalam teks konteks yang diberikan, bukan dari pengetahuan umum model LLM. Prinsip ini ditegakkan melalui instruksi eksplisit dalam *system prompt* yang dikirimkan ke LLM.

### 5.2 Format Data

Setiap data point mengikuti skema *text-to-text* dengan struktur sebagai berikut:

**Input** — gabungan konteks materi dan instruksi tugas menggunakan template tetap:

```
Konteks: {teks_materi}

Prompt: Buat satu soal {question_type} tentang {concept}, tingkat kesulitan: {difficulty}, bahasa Indonesia.
```

**Target** — plain string berisi soal lengkap (bukan JSON object), dengan format:

```
Pertanyaan: {pertanyaan}? Jawaban benar: {jawaban}. Distraktor: 1) {d1} 2) {d2} 3) {d3} 4) {d4}
```

Contoh data point lengkap:

```json
{
  "input": "Konteks: # Pengenalan Python\n\nPython adalah bahasa pemrograman multifungsi yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR)...\n\nPrompt: Buat satu soal MCQ tentang Ciri Khas Python, tingkat kesulitan: easy, bahasa Indonesia.",
  "target": "Pertanyaan: Apa ciri khas Python yang paling dikenal terkait penggunaan tanda baca pada akhir kode program? Jawaban benar: Python tidak mewajibkan penggunaan titik koma atau semi colon (`;`) pada setiap akhir kode programnya. Distraktor: 1) Python mewajibkan penggunaan titik koma (`;`) pada setiap akhir kode program 2) Python menggunakan kurung kurawal `{}` untuk mengakhiri setiap baris kode 3) Python mewajibkan penggunaan titik dua `:` di akhir setiap baris kode 4) Python tidak menggunakan tanda baca apapun dalam penulisan kode program",
  "metadata": {
    "difficulty": "easy",
    "question_type": "MCQ",
    "concept": "Ciri Khas Python",
    "misconception_tags": ["salah_paham_sintaks", "bingung_dengan_c", "salah_tanda_baca", "mengira_tanpa_aturan"],
    "source_file": "01-Berkenalan-dengan-python/01-perkenalan-pythn.md",
    "section": "# Pengenalan Python",
    "source": "synthetic",
    "validated": true
  }
}
```

### 5.3 Proses Generasi Sintetis

Data AQG dihasilkan secara sintetis menggunakan LLM (GPT-4o via OpenRouter API). Proses generasi mengikuti alur berikut:

1. **Chunking** — materi Markdown dipotong menjadi chunk 250–400 token
2. **Concept Extraction** — konsep yang relevan dipilih berdasarkan *keyword matching* antara teks chunk dan *Master Concept List*
3. **Prompt Construction** — input string dibangun menggunakan template tetap dengan parameter `concept`, `difficulty`, dan `question_type`
4. **LLM Generation** — LLM dipanggil dengan *system prompt* yang menginstruksikan: (a) soal hanya boleh dibuat dari informasi dalam konteks, (b) distraktor harus mencerminkan miskonsepsi umum siswa, (c) output harus menyertakan `misconception_tags`
5. **Validation** — setiap data point divalidasi: panjang input (50–600 token), kelengkapan format target, kelengkapan metadata, dan non-empty `misconception_tags`

Setiap data point yang lolos validasi secara otomatis mendapatkan flag `"validated": true` pada metadata-nya.

### 5.4 Metadata dan Misconception Tags

Setiap data point AQG dilengkapi metadata yang kaya untuk keperluan analisis dan filtering:

| Field | Deskripsi |
|-------|-----------|
| `difficulty` | Tingkat kesulitan: `easy`, `medium`, `hard` |
| `question_type` | Tipe soal: `MCQ`, `Code Completion` |
| `concept` | Konsep Python yang diuji (dari Master Concept List) |
| `misconception_tags` | Label miskonsepsi yang ditargetkan oleh distraktor |
| `source_file` | Path file Markdown asal |
| `source` | Asal data: `synthetic` |
| `validated` | Status validasi: `true` jika lolos semua aturan |

`misconception_tags` merupakan label singkat yang mengidentifikasi miskonsepsi umum siswa yang ditargetkan oleh setiap distraktor. Contoh: `salah_paham_sintaks`, `bingung_dengan_c`, `kebalikan_fakta`. Label ini berguna untuk analisis pedagogis dan evaluasi kualitas distraktor.

### 5.5 Statistik Dataset

Dataset AQG dihasilkan per modul secara terpisah, kemudian dapat digabungkan untuk fine-tuning. Berikut distribusi per modul:

| Modul                       | Jumlah Data Point |     |     |     |     |     |     |     |     |     |     |     |     |     |
| -----------------------------| -------------------| -----| -----| -----| -----| -----| -----| -----| -----| -----| -----| -----| -----| -----|
| 01-ber                      | an-python         | 90  |     |     |     |     |     |     |     |     |     |     |     |     |
| 02-berinteraksi-dengan-data | 180               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 03-ekspresi                 | 117               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 04-aksi-sekuensial          | 95                |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 05-control-flow             | 118               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 06-array                    | 126               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 07-matriks                  | 100               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 08-subprogram               | 108               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 09-oop                      | 148               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| 10-style-guide              | 120               |     |     |     |     |     |     |     |     |     |     |     |     |     |
| **Total (10 modul)**        | **1.202**         |     |     |     |     |     |     |     |     |     |     |     |     |     |

Setiap modul memiliki distribusi tingkat kesulitan yang seimbang: **easy : medium : hard = 1 : 1 : 1**. Seluruh data point berjenis MCQ dan bersumber dari generasi sintetis LLM (`source: synthetic`).

Setiap modul menghasilkan tiga split file: `train.jsonl` (70%), `validation.jsonl` (15%), `test.jsonl` (15%), serta `dataset_info.json` berisi statistik lengkap.

---

## 6. Validasi Kualitas Dataset

Kedua dataset menerapkan proses validasi otomatis sebelum data disimpan ke file output. Validasi bertujuan memastikan konsistensi format dan kelengkapan informasi yang diperlukan selama proses *fine-tuning*.

**Dataset Domain Adaptation** — aturan validasi:

- Panjang `input`: 10–1024 token
- `target`: non-empty string (minimal 5 karakter)
- `metadata.format`: hanya `span_corruption`, `summarization`, atau `qa_generic`
- `metadata.source_file` dan `metadata.module_name`: non-empty

**Dataset AQG** — aturan validasi:

- Panjang `input`: 50–600 token
- `target`: harus mengandung substring `"Pertanyaan:"`, `"Jawaban benar:"`, `"Distraktor:"`
- `metadata`: harus memiliki semua field wajib (`difficulty`, `question_type`, `concept`, `misconception_tags`)
- `metadata.difficulty`: hanya `easy`, `medium`, `hard`
- `metadata.misconception_tags`: non-empty list

Data yang gagal validasi dicatat dalam file `validation_failures.jsonl` beserta alasan kegagalannya, sehingga dapat dianalisis untuk perbaikan pipeline.

---

## 7. Ringkasan

| Aspek | Dataset Domain Adaptation | Dataset AQG |
|-------|--------------------------|-------------|
| Tujuan | Pemahaman domain Python | Generasi soal kuis terstruktur |
| Sumber | Materi Markdown (11 modul) | Materi Markdown + LLM (GPT-4o) |
| Format data | Span Corruption, QA Generik | MCQ text-to-text |
| Total data point | 340 | 1.202 (10 modul) |
| Split ratio | 80/10/10 | 70/15/15 |
| Stratifikasi | Berdasarkan format | Berdasarkan difficulty |
| Biaya LLM | Zero (mode saat ini) | Ya (GPT-4o via OpenRouter) |
| Digunakan pada | Fine-tuning Tahap 1 | Fine-tuning Tahap 2 |
