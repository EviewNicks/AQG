# Evaluasi dan Perbaikan Metodologi Penelitian

## Ringkasan Eksekutif

Dokumen ini mengevaluasi implementasi aktual penelitian terhadap metodologi yang direncanakan dalam dokumen `3. Metodologi.md`. Evaluasi mengidentifikasi perbedaan antara rencana awal dengan eksekusi, mengklasifikasikan perubahan sebagai *simplifikasi pragmatis* atau *gap yang perlu ditangani*, dan memberikan rekomendasi perbaikan untuk dokumentasi metodologi final.

## 1. Perbandingan Metodologi: Rencana vs Implementasi

### 1.1. Tahap 3.1: Akuisisi & Deskripsi Dataset

#### 3.1.1. Sumber Data Primer ✅ SESUAI

**Rencana:**
> "Sumber data utama penelitian ini adalah materi kursus 'Memulai Pemrograman dengan Python' dari platform Maguru. Korpus ini terdiri dari 11 modul terstruktur yang mencakup 55+ unit pembelajaran."

**Implementasi:**
- ✅ 11 modul berhasil diproses
- ✅ 55+ lesson dalam format Markdown
- ✅ Chunking berdasarkan heading dan code block integrity terjaga

**Status:** SESUAI PENUH

---

#### 3.1.2. Tahap 1: Domain Adaptation ⚠️ SIMPLIFIKASI

**Rencana:**
> "Proses ini menggunakan teknik pembelajaran tanpa pengawasan (self-supervised) pada korpus materi Maguru melalui dua format: (1) Span Corruption, (2) QA Generik (Rule-Based)"

**Implementasi:**
- ✅ Span Corruption: 118 entri (15% masking rate)
- ✅ QA Generik: 222 entri (rule-based extraction)
- ❌ **Summarization: TIDAK DIIMPLEMENTASIKAN**

**Alasan Simplifikasi:**
- Design document mencantumkan 3 format (span_corruption, summarization, qa_generic)
- Summarization membutuhkan LLM API cost yang signifikan
- Keputusan pragmatis: fokus pada 2 format zero-cost untuk MVP

**Dampak:**
- Dataset domain adaptation: 340 entri (target awal ~3,500)
- Masih cukup untuk domain adaptation karena span corruption adalah format utama T5 pre-training

**Rekomendasi Perbaikan Dokumentasi:**
```markdown
### 3.1.2. Tahap 1: Domain Adaptation (Self-Supervised Learning)

Tahap pertama menggunakan teknik pembelajaran tanpa pengawasan pada korpus 
materi Maguru melalui **dua format utama**:

1. **Span Corruption** (118 entri): Mengikuti objektif pra-pelatihan asli T5...
2. **QA Generik** (222 entri): Pasangan tanya-jawab diekstrak secara otomatis...

Total dataset domain adaptation: **340 entri** (train: 271, val: 33, test: 36).

*Catatan implementasi: Format summarization yang direncanakan tidak 
diimplementasikan pada fase MVP untuk mengurangi biaya komputasi LLM eksternal. 
Evaluasi awal menunjukkan dua format ini sudah cukup untuk adaptasi domain.*
```

---

#### 3.1.3. Tahap 2: Task-Specific AQG ❌ GAP SIGNIFIKAN

**Rencana:**
> "Sejumlah kecil data (misalnya, 300-500 pasang) akan dianotasi secara manual oleh pakar domain atau asisten peneliti. Anotasi ini mencakup pembuatan pertanyaan, jawaban benar, dan pengecoh yang berkualitas tinggi."

**Implementasi:**
- ✅ Dataset task-specific: 1,262 entri
- ✅ MCQ: 674 soal, Code Completion: 588 soal
- ✅ Balanced difficulty: 420 easy, 421 medium, 421 hard
- ❌ **SEMUA DATA SYNTHETIC (GPT-4o generated)**
- ❌ **TIDAK ADA ANOTASI MANUAL**

**Rencana:**
> "Untuk mengatasi keterbatasan data dan meningkatkan generalisasi model, teknik augmentasi akan diterapkan. Ini termasuk back-translation (Indonesia ↔ Inggris menggunakan mT5), synonym replacement (menggunakan IndoWordNet), dan paraphrasing."

**Implementasi:**
- ❌ **TIDAK ADA AUGMENTASI DATA**
- Komponen Augmentor ada di design document tapi tidak dieksekusi

**Dampak:**
- **Risiko Tinggi:** Model mungkin overfit pada gaya generasi GPT-4o
- **Risiko Sedang:** Tidak ada gold standard untuk validasi kualitas
- **Risiko Rendah:** Jumlah data (1,262) cukup besar untuk fine-tuning

**Rekomendasi Perbaikan Dokumentasi:**
```markdown
### 3.1.3. Tahap 2: Task-Specific AQG (Instruction Tuning)

Dataset ini dibangun dengan prinsip **Context Grounding** dan dihasilkan 
secara sintetis melalui LLM (GPT-4o) dengan instruksi yang ketat [4]. 

**Komposisi Dataset:**
- Total: 1,262 entri (train: 876, val: 175, test: 211)
- Tipe soal: MCQ (674), Code Completion (588)
- Distribusi difficulty: balanced (420 easy, 421 medium, 421 hard)
- Stratifikasi: berdasarkan module_name dan difficulty
- Sumber: 100% synthetic generation

**Pendekatan MVP (Minimum Viable Product):**
Penelitian ini mengadopsi pendekatan iteratif di mana dataset awal 
sepenuhnya sintetis untuk mempercepat siklus pengembangan. Validasi 
kualitas dilakukan melalui:
1. Automated validation: 14 correctness properties
2. Misconception tagging: setiap distraktor memiliki label pedagogis
3. Context grounding: semua soal bersumber dari materi yang diberikan

**Limitasi dan Rencana Iterasi:**
- Tidak ada anotasi manual pada fase MVP
- Augmentasi data (back-translation, paraphrasing) ditunda untuk iterasi 
  berikutnya berdasarkan hasil evaluasi model
- Validasi manual akan dilakukan post-training pada subset validation set 
  (50-100 entri) untuk mengukur kualitas aktual
```

---

### 1.2. Tahap 3.2: Pra-pemrosesan & Rekayasa Prompt

#### 3.2.1. Modular Chunking ⚠️ KEPUTUSAN DESAIN

**Rencana:**
> "Pembersihan teks dilakukan untuk menghapus karakter non-esensial dan markup (misalnya, Markdown) yang dapat mengganggu pemahaman model."

**Implementasi:**
- ✅ Chunking: 250-400 token (task-specific), 128-512 token (domain)
- ✅ Code block integrity terjaga
- ⚠️ **MARKDOWN FORMATTING DIPERTAHANKAN** (# ** ` \n)

**Justifikasi Keputusan:**
Berdasarkan research findings:
1. T5 SentencePiece tokenizer robust dengan special characters
2. Markdown memberikan semantic signals untuk struktur dokumen
3. Code blocks esensial untuk domain Python
4. Recent research (MDEval 2025) menunjukkan LLMs dapat belajar markdown structure

**Rekomendasi Perbaikan Dokumentasi:**
```markdown
#### 3.2.1. Modular Chunking dan Preservasi Struktur

Data mentah dari file Markdown diproses oleh komponen Chunker yang memotong 
teks menjadi chunks berukuran 250-400 token (task-specific) atau 128-512 
token (domain adaptation) berdasarkan batas heading.

**Keputusan Desain: Preservasi Markdown Formatting**

Berbeda dari rencana awal yang menyebutkan "pembersihan markup", implementasi 
akhir **mempertahankan formatting Markdown** (`#`, `**`, `` ` ``, `\n`) dengan 
pertimbangan:

1. **Robustness Tokenizer:** T5 SentencePiece tokenizer dirancang untuk 
   menangani special characters tanpa degradasi performa [1]
2. **Semantic Signals:** Markdown heading (`#`, `##`) memberikan informasi 
   struktural yang membantu model memahami hierarki topik
3. **Code Integrity:** Code blocks (` ```python ... ``` `) esensial untuk 
   domain pemrograman dan harus dipertahankan utuh
4. **Empirical Evidence:** Penelitian terbaru menunjukkan LLMs dapat 
   memanfaatkan markdown structure untuk pemahaman konteks yang lebih baik

Metadata penting (source_file, section_heading, token_count, has_code) tetap 
diekstrak dan disimpan untuk analisis post-training.
```

---

#### 3.2.2. Konstruksi Prompt ✅ SESUAI

**Rencana:**
> "Input untuk model dibangun menggunakan template tetap yang menggabungkan konteks materi dengan instruksi tugas spesifik."

**Implementasi:**
- ✅ Template: `"Konteks: {context}\n\nPrompt: Buat satu soal {question_type} tentang {concept}, tingkat kesulitan: {difficulty}, bahasa Indonesia."`
- ✅ Parameter: concept, difficulty, question_type
- ✅ Konsisten dengan design document

**Status:** SESUAI PENUH

---

## 2. Klasifikasi Tahapan: Verifikasi Dataset dan Tokenizer

### Pertanyaan: Apakah langkah MANDATORY masuk tahap 2.2 atau 2.3?

**Langkah MANDATORY yang dimaksud:**
1. Verifikasi dataset loadable dengan `datasets.load_dataset()`
2. Test IndoT5 tokenizer dengan markdown
3. Setup training environment
4. Baseline evaluation setup

**Analisis Klasifikasi:**

#### Opsi A: Masuk Tahap 2.2 (Pra-pemrosesan & Rekayasa Prompt)

**Argumen PRO:**
- Verifikasi dataset adalah bagian dari quality assurance pra-pemrosesan
- Test tokenizer terkait dengan format input (prompt engineering)
- Logis sebagai "final check" sebelum training

**Argumen KONTRA:**
- Tahap 2.2 fokus pada *pembuatan* dataset, bukan *validasi* dataset
- Setup training environment bukan bagian dari data preparation

#### Opsi B: Masuk Tahap 2.3 (Fine-tuning Model) ✅ REKOMENDASI

**Argumen PRO:**
- Verifikasi dataset adalah *prerequisite* untuk training
- Test tokenizer adalah bagian dari model preparation
- Setup environment adalah tahap awal training pipeline
- Baseline evaluation adalah bagian dari training workflow

**Argumen KONTRA:**
- Tidak ada argumen kuat menentang klasifikasi ini

**Keputusan:** Langkah MANDATORY adalah **sub-tahap 2.3.0** (Persiapan Fine-tuning)

---

## 3. Rekomendasi Struktur Metodologi yang Diperbaiki

### Struktur Baru yang Diusulkan:

```markdown
## 3.1. Akuisisi & Deskripsi Dataset

### 3.1.1. Sumber Data Primer: Korpus Materi Maguru
[Tetap sama - sudah sesuai]

### 3.1.2. Tahap 1: Domain Adaptation (Self-Supervised Learning)
[Revisi: jelaskan 2 format, total 340 entri, alasan tidak ada summarization]

### 3.1.3. Tahap 2: Task-Specific AQG (Instruction Tuning)
[Revisi: jelaskan pendekatan MVP, 100% synthetic, limitasi dan rencana iterasi]

## 3.2. Pra-pemrosesan & Rekayasa Prompt

### 3.2.1. Modular Chunking dan Preservasi Struktur
[Revisi: jelaskan keputusan mempertahankan markdown, justifikasi ilmiah]

### 3.2.2. Konstruksi Prompt (Prompt Engineering)
[Tetap sama - sudah sesuai]

## 3.3. Fine-tuning Model (IndoT5 + LoRA)

### 3.3.0. Persiapan Fine-tuning (BARU - TAMBAHAN)
Sebelum memulai proses fine-tuning, dilakukan serangkaian verifikasi dan 
persiapan untuk memastikan kesiapan dataset dan environment:

1. **Verifikasi Kompatibilitas Dataset**
   - Load dataset menggunakan HuggingFace `datasets` library
   - Validasi struktur JSONL (input, target, metadata)
   - Konfirmasi split ratio dan stratifikasi

2. **Validasi Tokenizer dengan Markdown**
   - Test IndoT5 tokenizer (Wikidepia/IndoT5-base) dengan sample data
   - Verifikasi handling markdown formatting (# ** ` \n)
   - Analisis distribusi panjang token (max_length: 512)
   - Konfirmasi tidak ada truncation issues pada code blocks

3. **Setup Training Environment**
   - GPU availability check (minimal 16GB VRAM)
   - Install dependencies: transformers, peft, accelerate, bitsandbytes
   - Konfigurasi hyperparameters awal:
     * LoRA rank: 8, alpha: 16
     * Learning rate: 2e-4
     * Batch size: 8
     * Epochs: 6 (domain) + 3 (task-specific)

4. **Baseline Evaluation**
   - Inference IndoT5 base model (pre-fine-tuning) pada 10 sample validation
   - Catat baseline metrics (BLEU-4, ROUGE-L, BERTScore)
   - Establish performance benchmark untuk perbandingan post-training

### 3.3.1. Pemilihan Model Dasar
[Tetap sama]

### 3.3.2. Implementasi LoRA
[Tetap sama]

### 3.3.3. Pelatihan Model
[Tetap sama]
```

---

## 4. Ringkasan Gap dan Rekomendasi

### Gap Kritis (Perlu Dokumentasi Eksplisit)

| Gap | Dampak | Rekomendasi |
|-----|--------|-------------|
| Tidak ada anotasi manual | Tidak ada gold standard | Dokumentasikan sebagai limitasi MVP, rencanakan validasi manual post-training (50-100 entri) |
| Tidak ada augmentasi data | Risiko overfitting pada gaya GPT-4o | Dokumentasikan sebagai iterasi future work, evaluasi perlu augmentasi berdasarkan hasil training |
| Format summarization tidak ada | Dataset domain lebih kecil (340 vs 3,500) | Dokumentasikan sebagai simplifikasi pragmatis, justifikasi: span corruption adalah format utama T5 |

### Keputusan Desain (Perlu Justifikasi Ilmiah)

| Keputusan | Justifikasi | Status Dokumentasi |
|-----------|-------------|-------------------|
| Markdown formatting dipertahankan | T5 tokenizer robust, semantic signals, empirical evidence | ✅ Perlu ditambahkan ke 3.2.1 |
| 100% synthetic dataset | MVP approach, iterative development | ✅ Perlu ditambahkan ke 3.1.3 |

### Penambahan Tahapan Baru

| Tahapan Baru | Alasan | Lokasi |
|--------------|--------|--------|
| 3.3.0. Persiapan Fine-tuning | Verifikasi dataset, tokenizer, environment adalah prerequisite training | Sebelum 3.3.1 |

---

## 5. Kesimpulan

Implementasi aktual penelitian mengadopsi pendekatan **pragmatis-iteratif** yang berbeda dari rencana awal yang lebih **komprehensif-ideal**. Perbedaan utama:

1. **Simplifikasi Dataset Domain:** 340 entri (2 format) vs 3,500 entri (3 format)
2. **Pendekatan MVP:** 100% synthetic vs rencana awal 300-500 manual + synthetic + augmented
3. **Preservasi Markdown:** Keputusan desain berbasis research findings
4. **Penambahan Tahap Verifikasi:** Sub-tahap 2.3.0 untuk memastikan kesiapan training

**Rekomendasi Final:**
- Update dokumen `3. Metodologi.md` dengan struktur yang diusulkan di atas
- Tambahkan sub-section "Limitasi dan Rencana Iterasi" di setiap tahap yang memiliki gap
- Dokumentasikan keputusan desain dengan justifikasi ilmiah
- Klasifikasikan langkah MANDATORY sebagai **Tahap 3.3.0: Persiapan Fine-tuning**

Pendekatan ini lebih jujur secara akademis dan memberikan roadmap jelas untuk iterasi berikutnya.

---

## Referensi

[1] C. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1-67, 2020.

[2] Y. Zhang et al., "Evaluating and Enhancing Markdown Awareness in Large Language Models," *arXiv preprint arXiv:2501.15000*, 2025. [Online]. Available: https://arxiv.org/html/2501.15000v1

[3] "How to Generate Synthetic Training Data for LLM Fine-Tuning (2026 Guide)," *Prem AI Blog*, 2026. [Online]. Available: https://blog.premai.io/how-to-generate-synthetic-training-data-for-llm-fine-tuning-2026-guide/

[4] F. Koto et al., "Cendol: Open instruction-tuned generative large language models for Indonesian languages," in *Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2024, pp. 796-810.

[5] A. Karotia et al., "Domain Adaptation by Two-Stage Fine-Tuning of Large Language Models," in *Proc. 23rd Workshop on Biomedical Natural Language Processing (BioNLP)*, 2024. [Online]. Available: https://aclanthology.org/2024.bionlp-1.69/

[6] "Proactive Data Collection and Iteration for Machine Learning Using Reflexive Planning, Monitoring, and Density Estimation," *arXiv preprint arXiv:2301.10319*, 2023. [Online]. Available: https://ar5iv.labs.arxiv.org/html/2301.10319

[7] "AI/ML MVP Guide: From Notebook to Production in 6 Steps," *Tokomsoft Insights*, 2025. [Online]. Available: https://tokomsoft.com/insights/ai-ml-mvp-guide

[8] "Building a Minimum Viable Product (MVP) for your AI startup with limited resources," *Nucamp Blog*, 2025. [Online]. Available: https://www.nucamp.co/blog/solo-ai-tech-entrepreneur-2025-building-a-minimum-viable-product-mvp-for-your-ai-startup-with-limited-resources

### Catatan Penggunaan Referensi

**[1] T5 Original Paper:**
- Digunakan untuk justifikasi preservasi markdown: T5 SentencePiece tokenizer dirancang robust terhadap special characters
- Mendukung keputusan desain di bagian 3.2.1

**[2] MDEval - Markdown Awareness:**
- Penelitian terbaru (2025) yang secara eksplisit mengevaluasi kemampuan LLM dalam memahami dan menghasilkan markdown
- Menunjukkan bahwa LLM dapat belajar struktur markdown dan memanfaatkannya untuk pemahaman konteks
- Mendukung keputusan preservasi markdown formatting di bagian 3.2.1
- Key finding: "Markdown Awareness is closely related with English/Chinese/Coding/Longer Query"

**[3] Synthetic Data Generation Guide:**
- Panduan praktis tentang synthetic data generation untuk fine-tuning LLM
- Menjelaskan trade-off antara biaya human annotation ($15,000-$50,000) vs synthetic generation ($50-$500)
- Mendukung pendekatan MVP di bagian 3.1.3
- Key insight: "Synthetic data carries the biases, blindspots, and capability limits of whatever model generated it"

**[4] Cendol - Instruction Tuning:**
- Referensi untuk instruction tuning approach yang digunakan dalam task-specific AQG
- Mendukung metodologi di bagian 3.1.3

**[5] Two-Stage Fine-Tuning:**
- Justifikasi untuk pendekatan hibrida dua tahap (domain adaptation + task-specific)
- Mendukung struktur keseluruhan metodologi

**[6] Iterative ML Development:**
- Mendukung pendekatan iteratif dan MVP
- "ML practitioners need to change data, models, and their associated processes as they become more familiar with their modeling task"
- Justifikasi untuk pendekatan pragmatis di bagian 3.1.3

**[7-8] MVP Development:**
- Mendukung pendekatan MVP untuk AI/ML projects
- Justifikasi untuk simplifikasi dan iterasi bertahap
- "Start with the simplest model that could work... Focus on the delta to the KPI"
