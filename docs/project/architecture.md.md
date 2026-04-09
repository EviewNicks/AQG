# Pembaruan Arsitektur dan Alur Sistem Proyek
**"Automatic Generation of Python Programming Quiz Questions and Distractors Using IndoT5 with LoRA for Indonesian Educational Content"**

Dokumen ini menyajikan pembaruan arsitektur dan alur sistem proyek yang telah disesuaikan berdasarkan hasil analisis dan evaluasi sebelumnya. Tujuan dari pembaruan ini adalah untuk memberikan kejelasan metodologis, mengintegrasikan praktik terbaik dari literatur, dan memastikan kelengkapan tahapan dari akuisisi data hingga evaluasi akhir.

## 1. Alur Sistem yang Diperbarui
Alur sistem proyek kini direvisi menjadi enam tahapan utama, yang mencerminkan proses end-to-end yang lebih komprehensif dan terstruktur:

1.  **Akuisisi & Deskripsi Dataset (Data Acquisition & Description)**
2.  **Pra-pemrosesan & Rekayasa Prompt (Preprocessing & Prompt Engineering)**
3.  **Fine-tuning Model (IndoT5 + LoRA)**
4.  **Generasi Soal & Pengecoh Bersama (Joint Question & Distractor Generation)**
5.  **Validasi & Penyaringan (Validation & Filtering)**
6.  **Evaluasi Komprehensif (Comprehensive Evaluation)**

## 2. Deskripsi Tahapan yang Diperbarui

### 2.1. Akuisisi & Deskripsi Dataset
Tahap ini merupakan fondasi utama proyek, berfokus pada pengumpulan, anotasi, dan augmentasi data pelatihan. Kualitas model IndoT5 yang di-*fine-tune* sangat bergantung pada representasi data yang akurat dan relevan.

-   **Pengumpulan Data (Data Sourcing):** Materi pembelajaran Python (misalnya, dari platform Maguru) dalam format Markdown/YAML akan dikumpulkan. Data ini akan menjadi sumber utama untuk ekstraksi konteks.
-   **Anotasi Data (Data Annotation):** Sejumlah kecil data (misalnya, 300-500 pasang) akan dianotasi secara manual oleh pakar domain atau asisten peneliti. Anotasi ini mencakup pembuatan pertanyaan, jawaban benar, dan pengecoh yang berkualitas tinggi, serta pelabelan metadata seperti tingkat kesulitan, konsep, dan *misconception* yang diuji. Data anotasi ini akan berfungsi sebagai *gold standard* dan data pelatihan awal.
-   **Augmentasi Data (Data Augmentation):** Untuk mengatasi keterbatasan data dan meningkatkan generalisasi model, teknik augmentasi akan diterapkan. Ini termasuk *back-translation* (Indonesia ↔ Inggris menggunakan mT5), *synonym replacement* (menggunakan IndoWordNet), dan *paraphrasing* dengan model LLM lain (misalnya, GPT-4o) untuk menghasilkan variasi pertanyaan dan pengecoh.

### 2.2. Pra-pemrosesan & Rekayasa Prompt
Tahap ini mempersiapkan data teks agar sesuai dengan format input yang diharapkan oleh model IndoT5 dan mengkonstruksi prompt yang efektif.

-   **Ekstraksi Konteks:** Dari materi pembelajaran Python, bagian-bagian relevan (chunk teks 250-400 token) akan diekstraksi. Ini bisa berupa penjelasan konsep, contoh kode, atau deskripsi fungsi.
-   **Pembersihan Teks (Text Cleaning):** Teks akan dibersihkan dari karakter yang tidak relevan, *markup* (misalnya, Markdown), dan *noise* lainnya.
-   **Konstruksi Prompt (Prompt Construction):** Prompt input untuk model IndoT5 akan dibangun. Prompt ini akan menggabungkan konteks materi dengan instruksi spesifik untuk menghasilkan soal kuis, jawaban, dan pengecoh. Contoh prompt dapat mencakup informasi tentang topik, tingkat kesulitan yang diinginkan, dan jenis soal (MCQ atau *Code Completion*).

### 2.3. Fine-tuning Model (IndoT5 + LoRA)

Model IndoT5 akan di-*fine-tune* menggunakan teknik *Low-Rank Adaptation* (LoRA) untuk mengadaptasinya secara efisien pada tugas pembuatan soal kuis pemrograman Python dalam bahasa Indonesia. Proses fine-tuning mengadopsi strategi **hibrida dua tahap** untuk memaksimalkan adaptasi domain dan performa task-specific.

#### 2.3.0. Persiapan Fine-tuning (Pre-Training Setup)

Sebelum memulai training, dilakukan serangkaian verifikasi dan persiapan untuk memastikan kesiapan sistem:

**A. Verifikasi Kompatibilitas Dataset**
- **Load Dataset:** Menggunakan HuggingFace `datasets` library untuk memuat dataset dari format JSONL
  - Domain adaptation: 340 entri (train: 271, val: 33, test: 36)
  - Task-specific AQG: 1,262 entri (train: 876, val: 175, test: 211)
- **Validasi Struktur:** Memastikan setiap entri memiliki field `input`, `target`, dan `metadata` yang valid
- **Konfirmasi Split:** Verifikasi stratifikasi berdasarkan format (domain) dan module_name + difficulty (task-specific)
- **Data Integrity Check:** Deteksi missing values, duplicate entries, atau data corruption

**B. Validasi Tokenizer dengan Markdown**
- **Tokenizer Testing:** Test IndoT5 tokenizer (`Wikidepia/IndoT5-base`) dengan sample data yang mengandung markdown formatting
- **Markdown Handling:** Verifikasi bahwa special characters (`#`, `**`, `` ` ``, `\n`) tidak menyebabkan tokenization errors
- **Length Analysis:** Analisis distribusi panjang token untuk memastikan mayoritas sample berada dalam batas `max_length=512`
  - Jika > 5% sample melebihi 512 tokens → perlu adjustment chunking strategy
- **Code Block Verification:** Konfirmasi tidak ada truncation issues pada code blocks yang panjang
- **Vocabulary Coverage:** Check apakah ada token OOV (out-of-vocabulary) yang signifikan

**C. Setup Training Environment**
- **Hardware Requirements:**
  - GPU: Minimal 16GB VRAM (NVIDIA T4, V100, atau A100)
  - RAM: Minimal 32GB untuk data loading
  - Storage: ~10GB untuk model checkpoints
- **Software Dependencies:**
  ```bash
  pip install transformers==4.36.0
  pip install peft==0.7.0
  pip install datasets==2.16.0
  pip install accelerate==0.25.0
  pip install bitsandbytes==0.41.0
  pip install evaluate==0.4.1
  pip install rouge-score==0.1.2
  ```
- **Environment Configuration:**
  - Mixed precision training (FP16) untuk efisiensi memory
  - Gradient checkpointing untuk mengurangi memory footprint
  - Distributed training setup (jika multi-GPU available)

**D. Baseline Evaluation**
- **Pre-Fine-tuning Inference:** Run IndoT5 base model (tanpa fine-tuning) pada 10 sample dari validation set
- **Baseline Metrics:** Catat performa awal:
  - BLEU-4: Expected ~0.05-0.10 (very low, karena model belum dilatih untuk task ini)
  - ROUGE-L: Expected ~0.10-0.15
  - BERTScore: Expected ~0.60-0.65
- **Qualitative Analysis:** Dokumentasikan contoh output untuk analisis kualitatif
- **Performance Benchmark:** Establish baseline untuk perbandingan post-training

#### 2.3.1. Pemilihan Model Dasar

**Model:** IndoT5 (`Wikidepia/IndoT5-base`)
- **Arsitektur:** T5 (Text-to-Text Transfer Transformer) encoder-decoder
- **Parameters:** ~250M parameters
- **Pre-training:** Korpus monolingual Indonesia (Wikipedia, news, web crawl)
- **Vocabulary Size:** 32,000 tokens (SentencePiece)
- **Context Length:** 512 tokens (input dan output)

**Alasan Pemilihan:**
1. **Bahasa Indonesia:** Pre-trained pada korpus Indonesia, memahami struktur bahasa dan idiom lokal
2. **Text-to-Text Framework:** Cocok untuk task generatif seperti question generation
3. **Ukuran Reasonable:** 250M parameters memungkinkan fine-tuning dengan resource terbatas
4. **Community Support:** Model populer dengan dokumentasi dan contoh implementasi yang baik

#### 2.3.2. Implementasi LoRA (Low-Rank Adaptation)

LoRA adalah teknik PEFT (Parameter-Efficient Fine-Tuning) yang hanya melatih sebagian kecil parameter tambahan, mengurangi kebutuhan komputasi dan memory secara signifikan.

**Konfigurasi LoRA:**
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Sequence-to-sequence language modeling
    r=8,                               # LoRA rank (low-rank matrices dimension)
    lora_alpha=16,                     # Scaling factor (typically 2*r)
    lora_dropout=0.1,                  # Dropout for regularization
    target_modules=["q", "v"],         # Apply LoRA to attention query and value
    bias="none",                       # Don't train bias parameters
    inference_mode=False               # Training mode
)
```

**Parameter Explanation:**
- **rank (r=8):** Dimensi matriks low-rank. Semakin besar, semakin banyak parameter trainable (trade-off: capacity vs efficiency)
- **alpha (α=16):** Scaling factor untuk LoRA weights. Formula: `scaling = α / r = 16/8 = 2`
- **target_modules:** Hanya melatih attention layers (q, v), bukan seluruh model
- **Trainable Parameters:** ~0.5% dari total model parameters (~1.25M dari 250M)

**Memory Savings:**
- Full fine-tuning: ~40GB VRAM (untuk 250M model)
- LoRA fine-tuning: ~16GB VRAM (60% reduction)

#### 2.3.3. Pelatihan Model (Two-Stage Training)

**Stage 1: Domain Adaptation (6 epochs)**

*Objective:* Adaptasi terminologi teknis Python dan gaya instruksional Indonesia

- **Dataset:** 340 entri (span corruption + QA generik)
- **Training Configuration:**
  ```python
  training_args = Seq2SeqTrainingArguments(
      output_dir="./checkpoints/domain_adaptation",
      num_train_epochs=6,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      learning_rate=2e-4,
      weight_decay=0.01,
      warmup_steps=50,              # 10% of total steps
      evaluation_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=3,           # Keep only last 3 checkpoints
      logging_steps=50,
      fp16=True,                    # Mixed precision training
      gradient_checkpointing=True,  # Memory optimization
      predict_with_generate=True,
      generation_max_length=512,
      load_best_model_at_end=True,
      metric_for_best_model="eval_loss"
  )
  ```
- **Monitoring Metrics:**
  - Training loss: Expected to decrease from ~3.0 to ~1.5
  - Validation loss: Expected to decrease from ~2.8 to ~1.8
  - Perplexity: Expected to decrease from ~20 to ~6
  - Reconstruction accuracy (for span corruption): Expected ~70-80%
- **Early Stopping:** Jika validation loss tidak improve selama 2 epochs
- **Output:** Model checkpoint `indot5-python-domain` (~500MB)

**Stage 2: Task-Specific AQG (3 epochs)**

*Objective:* Pembelajaran pola generasi soal MCQ dan Code Completion dengan distraktor

- **Dataset:** 1,262 entri (MCQ + Code Completion)
- **Initial Weights:** Load dari checkpoint Stage 1 (`indot5-python-domain`)
- **Training Configuration:**
  ```python
  training_args = Seq2SeqTrainingArguments(
      output_dir="./checkpoints/task_specific",
      num_train_epochs=3,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      learning_rate=1e-4,           # Lower LR for fine-tuning
      weight_decay=0.01,
      warmup_steps=30,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=2,
      logging_steps=100,
      fp16=True,
      gradient_checkpointing=True,
      predict_with_generate=True,
      generation_max_length=512,
      generation_num_beams=4,       # Beam search for better quality
      load_best_model_at_end=True,
      metric_for_best_model="eval_bleu"
  )
  ```
- **Monitoring Metrics:**
  - BLEU-4: Expected to increase from ~0.10 to ~0.35-0.45
  - ROUGE-L: Expected to increase from ~0.15 to ~0.45-0.55
  - BERTScore: Expected to increase from ~0.65 to ~0.75-0.80
  - Diversity metrics: Distinct-1, Distinct-2 (untuk mengukur variasi output)
- **Validation Strategy:** Evaluate setiap epoch pada validation set
- **Output:** Final model `indot5-python-aqg` (~500MB)

**Training Time Estimation:**
- Stage 1 (6 epochs, 340 samples): ~30-45 minutes (on T4 GPU)
- Stage 2 (3 epochs, 1,262 samples): ~1.5-2 hours (on T4 GPU)
- **Total:** ~2-2.5 hours

**Checkpointing Strategy:**
- Save checkpoint setiap epoch
- Keep only best 2-3 checkpoints (berdasarkan validation metrics)
- Save final model dengan LoRA weights merged untuk inference

**Troubleshooting Common Issues:**
1. **CUDA Out of Memory:**
   - Reduce batch size: `per_device_train_batch_size=4`
   - Enable gradient accumulation: `gradient_accumulation_steps=2`
   - Use 8-bit quantization: `load_in_8bit=True`

2. **Overfitting:**
   - Increase dropout: `lora_dropout=0.2`
   - Add weight decay: `weight_decay=0.05`
   - Early stopping based on validation loss

3. **Slow Convergence:**
   - Increase learning rate: `learning_rate=3e-4`
   - Adjust warmup steps: `warmup_steps=100`
   - Check data quality and preprocessing

### 2.4. Generasi Soal & Pengecoh Bersama (Joint Question & Distractor Generation)
Pada tahap ini, model IndoT5 yang telah di-*fine-tune* akan menghasilkan pertanyaan, jawaban benar, dan pengecoh secara simultan dalam satu *output* terstruktur.

-   **Input Model:** Konteks materi Python yang telah dipra-proses dan prompt yang telah dikonstruksi akan diberikan sebagai input ke model.
-   **Output Generatif:** Model akan menghasilkan *output* teks yang berisi pertanyaan kuis, jawaban yang benar, dan 3-4 pengecoh yang relevan dan pedagogis. Format *output* ini akan konsisten dengan format *target* yang digunakan selama pelatihan.

### 2.5. Validasi & Penyaringan (Validation & Filtering)
Output generatif dari model akan melalui serangkaian validasi dan penyaringan untuk memastikan kualitas dan relevansinya.

-   **Validasi Sintaksis Kode:** Untuk soal *Code Completion* atau soal yang melibatkan potongan kode, validasi sintaksis akan dilakukan untuk memastikan kode yang dihasilkan valid dan dapat dieksekusi.
-   **Penyaringan Semantik (Semantic Filtering):** Menggunakan metrik kesamaan semantik (misalnya, *cosine similarity* dengan ambang batas < 0.65) untuk memastikan pengecoh tidak terlalu mirip dengan jawaban benar atau tidak terlalu jauh dari konteks soal.
-   **Penyaringan Pedagogis (Pedagogical Filtering):** Ini adalah langkah krusial untuk memastikan kualitas edukasi dari soal dan pengecoh. Filter ini akan memeriksa:
    -   **Plausibilitas Pengecoh:** Apakah pengecoh masuk akal dan dapat mengecoh siswa yang memiliki *misconception* umum.
    -   **Kejelasan & Ambigu:** Memastikan soal dan pengecoh tidak ambigu atau memiliki jawaban benar ganda.
    -   **Relevansi Konsep:** Memastikan soal menguji konsep yang relevan dengan konteks materi.

### 2.6. Evaluasi Komprehensif
Tahap akhir melibatkan evaluasi menyeluruh terhadap kualitas soal dan pengecoh yang dihasilkan, baik secara otomatis maupun melalui penilaian manusia.

-   **Evaluasi Otomatis:** Menggunakan metrik NLP standar seperti BLEU-4, ROUGE-L, dan BERTScore untuk mengukur kualitas generatif model dibandingkan dengan *gold standard*.
-   **Evaluasi Manusia (Human Evaluation):** Pakar domain (misalnya, pengajar pemrograman) akan mengevaluasi soal dan pengecoh berdasarkan kriteria seperti relevansi, kejelasan, tingkat kesulitan, nilai pedagogis, dan kemampuan pengecoh untuk mengidentifikasi *misconception* siswa. Hasil evaluasi manusia ini akan menjadi metrik kualitas paling penting.

## 3. Diagram Alur Sistem (Flowchart)

```mermaid
graph TD
    A[Materi Pembelajaran Python (Markdown/YAML)] --> B(Data Acquisition & Description)
    B --> C(Preprocessing & Prompt Engineering)
    C --> D{Fine-tuning Model IndoT5 + LoRA}
    D --> E(Joint Question & Distractor Generation)
    E --> F(Validation & Filtering)
    F --> G(Comprehensive Evaluation)
    G --> H[Soal Kuis Python Siap Pakai]

    subgraph Data Preparation
        B
        C
    end

    subgraph Model Training & Generation
        D
        E
    end

    subgraph Output Refinement & Assessment
        F
        G
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
```

Dokumen ini diharapkan dapat memberikan panduan yang jelas dan terstruktur untuk pengembangan proyek Anda, memastikan setiap tahapan dipertimbangkan dengan cermat dan sesuai dengan praktik terbaik dalam bidang NLP dan AQG.


![Diagram Alur Sistem Proyek](https://private-us-east-1.manuscdn.com/sessionFile/miHkxzNAUkl1djnq8d3uuI/sandbox/AoB1iHEkxastmeJf5gKtaQ-images_1775570216389_na1fn_L2hvbWUvdWJ1bnR1L2FyY2hpdGVjdHVyZV9mbG93.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvbWlIa3h6TkFVa2wxZGpucThkM3V1SS9zYW5kYm94L0FvQjFpSEVreGFzdG1lSmY1Z0t0YVEtaW1hZ2VzXzE3NzU1NzAyMTYzODlfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnlZMmhwZEdWamRIVnlaVjltYkc5My5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=fGgGgDjBHw8lMjDW0s2mqcjyCnFuI7FlUAKw-vZtHte3U0r6SAS3F2VAYG7KJGhvyUpN5Cld3vIfVXqGvPzckPXKA7olqpQcICDOLWP98a3Z8XSEWbmn1fNbtZd-bZNpPaQE4IcTQGTFgy4jdCNYanmD~XLC458S~TzWAv~XpBNwg1Pf8C~fBwK1jDC-qm-3Qiszt36rYcKdjpRbZTIBDTs-eiaip7NH-rQ5wMa49Hxzz5P6Rbqb9NbAdQ0dLCCCzTJ2XQJUYO-ernvoB~rAx~IP9MvUXSyaDtur~w~nTLm8RKxSStwxyyb0qN3OtHGwSz7f0ESqXoVJuSkjUXKyYQ__)
