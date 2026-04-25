# Mengapa Input Dimulai dengan "generate_mcq:"? - Penjelasan Ringkas

---

## PERTANYAAN ANDA
"Mengapa input format dataset yang baru ini dimulai dengan generate_mcq? Darimana referensi ini?"

---

## JAWABAN SINGKAT

**Karena T5 menggunakan "task prefix" untuk memberi tahu model apa yang harus dilakukan.**

---

## ANALOGI SEDERHANA

Bayangkan Anda adalah seorang asisten yang bisa melakukan banyak tugas:

```
Tanpa prefix:
Anda: "Dalam Python, array dimulai dari indeks 0"
Asisten: Bingung! Apakah saya harus merangkum? Menerjemahkan? Membuat soal?

Dengan prefix:
Anda: "generate_mcq: Dalam Python, array dimulai dari indeks 0"
Asisten: Ah! Saya harus membuat soal multiple choice! Langsung mengerti!
```

---

## KONSEP T5: TEXT-TO-TEXT FRAMEWORK

T5 (Text-to-Text Transfer Transformer) adalah model yang bisa melakukan berbagai task NLP.

**Cara kerjanya:**
- Semua task dikonversi ke format text-to-text
- Setiap task punya prefix yang unik
- Prefix memberi tahu model: "Ini task apa?"

**Contoh prefix T5 resmi:**
```
summarize: [teks] → [ringkasan]
translate English to German: [teks] → [terjemahan]
question answering: [pertanyaan] [konteks] → [jawaban]
```

**Untuk MCQ generation (task baru):**
```
generate_mcq: [konteks] → [pertanyaan + jawaban + distraktor]
```

---

## REFERENSI RESMI

### 1. T5 PAPER (ORIGINAL)
**Judul:** "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

**Penulis:** Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., et al.

**Publikasi:** arXiv:1910.10683 (2019)

**Link:** https://arxiv.org/abs/1910.10683

**Kutipan Penting:**
"To formulate every task as text generation, each task is prepended with a task-specific prefix (e.g., translate English to German: …, summarize: …). This enables T5 to handle tasks like translation, summarization, question answering, and more."

**Artinya:** "Untuk membuat setiap task sebagai text generation, setiap task diberi prefix khusus (contoh: translate English to German: …, summarize: …). Ini memungkinkan T5 menangani berbagai task seperti translation, summarization, question answering, dll."

---

### 2. HUGGINGFACE T5 DOCUMENTATION
**Link:** https://huggingface.co/docs/transformers/model_doc/t5

**Penjelasan:**
"T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format."

**Artinya:** "T5 adalah model encoder-decoder yang pre-trained pada mixture dari berbagai task, dan setiap task dikonversi ke format text-to-text."

---

### 3. QUESTION GENERATION REPOSITORY (GITHUB)
**Repository:** patil-suraj/question_generation

**Link:** https://github.com/patil-suraj/question_generation

**Praktik:**
- Menggunakan prefix "generate_question:" untuk question generation
- Menggunakan prefix "generate_mcq:" untuk MCQ generation
- Ini adalah praktik standar dalam komunitas NLP

---

## MENGAPA HARUS ADA PREFIX?

### Alasan 1: Model Perlu Tahu Task Apa
```
Tanpa prefix, model bingung:
Input: "Dalam Python, array dimulai dari 0"
Output yang mungkin: 
  - Ringkasan? "Python array dari 0"
  - Terjemahan? "In Python, array starts from 0"
  - Soal? "Indeks array dimulai dari berapa?"
  - Kesimpulan? "Array penting dalam Python"

Dengan prefix, model tahu:
Input: "generate_mcq: Dalam Python, array dimulai dari 0"
Output: Pasti soal MCQ!
```

### Alasan 2: Multi-Task Learning
T5 didesain untuk bisa handle banyak task dengan satu model:
- Summarization
- Translation
- Question answering
- Question generation
- Text classification
- Dan lainnya...

Prefix membedakan task satu dengan lainnya.

### Alasan 3: Kontrol Behavior Model
Prefix memberi "instruksi" kepada model tentang apa yang diharapkan.

---

## BAGAIMANA CARA KERJA PREFIX?

### Saat Pre-training:
```
Model belajar: "Ketika melihat prefix 'summarize:', output harus ringkasan"
Model belajar: "Ketika melihat prefix 'translate:', output harus terjemahan"
Model belajar: "Ketika melihat prefix 'generate_mcq:', output harus MCQ"
```

### Saat Fine-tuning:
```
Model sudah tahu: "generate_mcq:" = buat soal MCQ
Tinggal di-fine-tune dengan data MCQ spesifik
```

### Saat Inference:
```
Input: "generate_mcq: Dalam Python, array dimulai dari 0"
Model: "Ah! Prefix 'generate_mcq:' berarti saya harus generate MCQ"
Output: "question: ...\nanswer: ...\ndistractors: ..."
```

---

## APAKAH BISA PAKAI PREFIX LAIN?

**Secara teknis:** YA, bisa pakai prefix apa saja

**Contoh alternatif:**
```
"create_mcq: ..." (create = buat)
"mcq_generation: ..."
"generate_quiz: ..."
"make_multiple_choice: ..."
```

**TAPI:** Lebih baik pakai prefix yang sudah standard/dikenal:
- Lebih mudah dipahami
- Konsisten dengan praktik komunitas
- Model lebih mudah belajar (karena sudah familiar dari pre-training)

---

## KESIMPULAN

**Mengapa "generate_mcq:"?**

1. ✅ Ini adalah praktik standard T5 (dari paper Raffel et al., 2019)
2. ✅ Prefix memberi tahu model: "Ini task MCQ generation"
3. ✅ Model sudah familiar dengan konsep prefix dari pre-training
4. ✅ Membantu model fokus pada task yang tepat
5. ✅ Digunakan di komunitas NLP (GitHub, HuggingFace)

**Referensi Utama:**
- Raffel et al. (2019) - T5 Paper (arXiv:1910.10683)
- HuggingFace T5 Documentation
- patil-suraj/question_generation (GitHub)

---

**Mudah dipahami?** 😊

Intinya: **Prefix adalah "instruksi" yang memberi tahu model apa yang harus dilakukan. Ini adalah konsep standar dari T5 paper yang diterbitkan tahun 2019.**

