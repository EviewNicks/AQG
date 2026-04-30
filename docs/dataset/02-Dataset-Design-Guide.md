# Dataset Design Guide: IndoNanoT5 MCQ Generation

**Status:** FINAL DESIGN  
**Task Type:** Multiple Choice Question Generation (MCQ-G)  
**Model:** LazarusNLP/IndoNanoT5-base  
**Format:** JSONL (JSON Lines)  
**Date:** 22 April 2026

---

## 1. TASK NLP CLASSIFICATION

### Task Type: **Multiple Choice Question Generation (MCQ-G)**

**Definition:**
```
Input:  Context/passage (plain text, preserve code blocks)
Output: Structured MCQ (question + correct answer + distractors)
```

**Why MCQ-G (Not Just QG)?**

| Aspect | Question Generation | MCQ Generation |
|--------|-------------------|-----------------|
| **Output** | Only question | Q + Answer + Distractors |
| **Complexity** | Simple seq2seq | Structured generation |
| **Use Case** | General QA | Educational quizzes |
| **Your Project** | ❌ | ✅ Perfect fit |

**Architecture:** Encoder-Decoder (T5 style)
```
Encoder: Processes context
Decoder: Generates structured MCQ output
```

---

## 2. FORMAT DATASET FINAL

### 2.1 File Format: **JSONL** (JSON Lines)

**Why JSONL, not JSON?**

| Criteria | JSON | JSONL |
|----------|------|-------|
| **Structure** | Single array | One JSON per line |
| **Streaming** | Load all at once | Load line-by-line |
| **Memory** | High (all data in RAM) | Low (streaming) |
| **HuggingFace** | ❌ Not preferred | ✅ Preferred |
| **Large datasets** | Problematic | Efficient |

**Your Choice:** ✅ **JSONL** (better for large datasets)

---

### 2.2 Input Format (Source Text)

```
Plain text ONLY - NO markdown formatting
- ✅ Keep code blocks with triple backticks (```)
- ❌ Remove ## headers
- ❌ Remove ** bold markers
- ❌ Remove * italics markers
- ✅ Keep newlines for readability
```

**Why?**
- IndoNanoT5 trained on plain text
- Markdown tokens confuse the model
- Code blocks are content, not formatting

---

### 2.3 Output Format (Target Text)

**Format: Structured Plain Text**

```
question: [QUESTION TEXT]
answer: [CORRECT ANSWER]
distractors: [DISTRACTOR1] | [DISTRACTOR2] | [DISTRACTOR3]
```

**Example:**
```
question: Indeks array dalam Python dimulai dari berapa?
answer: 0
distractors: 1 | -1 | n
```

example :

```

{"input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_mat = [[10, 20],\n  [30, 40],\n [50, 60]]\nprint(var_mat[0][1] + var_mat[2][1])\n```\n Kode ini menjumlahkan elemen kolom kedua dari baris pertama dan baris terakhir.", "output": "question: Perhatikan kode berikut:\n```python\nvar_mat = [[10, 20],\n           [30, 40],\n           [50, 60]]\nprint(var_mat[0][1] + var_mat[2][1])\n```\nApa output dari kode tersebut?\nanswer: 80\ndistractors: 70 | 90 | 60", "metadata": {"difficulty": "Sedang", "source_section": "07-matriks", "source_file": "materi2.jsonl"}}

{"input": "buat_soal_pilihan_ganda: Fungsi min() dan max() dapat digunakan pada tuple.", "output": "question: Perhatikan kode berikut:\n```python\ndata = (5, 2, 8, 1)\nprint(max(data))\n```\nApa output dari kode di atas?\nanswer: 8\ndistractors: 5 | 1 | Error", "metadata": {"difficulty": "Mudah", "source_section": "02-berinteraksi-dengan-data", "source_file": "5_operasi_list_set_string.jsonl"}}

{"input": "buat_soal_pilihan_ganda: Variadic function adalah fungsi yang dapat menerima jumlah argumen yang bervariasi. Dalam Python, variadic function dibuat dengan *args dan **kwargs.", "output": "question: Apa yang dimaksud dengan variadic function?\nanswer: Fungsi yang dapat menerima jumlah argumen yang bervariasi\ndistractors: Fungsi yang hanya menerima satu argumen | Fungsi yang tidak menerima argumen | Fungsi yang selalu error", "metadata": {"difficulty": "Sedang", "source_section": "08-subprogram", "source_file": "materi4.jsonl"}}



```

**Why This Format?**
- Structured but parseable
- No special tokens needed
- Easy to split during inference
- Compatible with T5 text-to-text framework

---

### 2.4 Task Prefix (T5 Convention)

**Prefix to prepend to input:**

```
buat_soal_pilihan_ganda:
```

**Full Input Format:**
```
buat_soal_pilihan_ganda: [PLAIN TEXT CONTEXT]
```

**Why This Prefix?**
- Uses Indonesian language for better alignment with IndoNanoT5
- "buat_soal_pilihan_ganda" = "create multiple choice question"
- T5 uses prefixes to specify tasks
- Helps model understand what to do
- Standard practice in T5 fine-tuning
- Reference: Raffel et al., T5 paper (2019)

---

## 3. CONTOH DATASET (3 SAMPLES)

### Sample 1: Array Indexing

```json
{
  "input": "buat_soal_pilihan_ganda: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1. Contoh: jika Anda memiliki list = [10, 20, 30], maka list[0] adalah 10, list[1] adalah 20, dan list[2] adalah 30.",
  "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1"
}
```

### Sample 2: Code Execution

```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(var_list[1][2])\n```\nKode ini mengakses elemen pada baris kedua (indeks 1) dan kolom ketiga (indeks 2) dari nested list.",
  "output": "question: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(var_list[1][2])\n```\nApa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9"
}
```

**PENTING:** Question harus menyertakan code block agar self-contained!

### Sample 3: String Operations

```json
{
  "input": "buat_soal_pilihan_ganda: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat. Jika Anda mencoba mengubah karakter dalam string dengan indexing seperti s[0] = 'A', Python akan menampilkan error TypeError. Untuk mengubah string, Anda harus membuat string baru.",
  "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string dengan s[0] = 'A'?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError"
}
```


---

## 5. REFERENSI

### 5.1 T5 Text-to-Text Framework

**Reference:** Raffel, C., Shazeer, N., Roberts, A., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." arXiv:1910.10683

**Key Points:**
- T5 converts all NLP tasks to text-to-text format
- Task prefixes guide model behavior
- No task-specific architectures needed
- Format: `[prefix]: [input] → [output]`



---

### 5.2 IndoNanoT5 Documentation

**Reference:** LazarusNLP/IndoNanoT5-base (HuggingFace Hub)

**Model Details:**
- Architecture: T5 (Encoder-Decoder)
- Language: Indonesian
- Pretraining: 4B tokens from CulturaX corpus
- Input length: 512 tokens
- License: Apache 2.0

**URL:** https://huggingface.co/LazarusNLP/IndoNanoT5-base

---

### 5.3 Question Generation with T5

**Reference:** Patil, S. (2021). "Question Generation using Transformers." GitHub Repository

**Key Insights:**
- T5 can handle multiple QG variants (answer-aware, end-to-end, MCQ)
- Task prefixes: `generate_question:`, `generate_mcq:`, etc.
- Output format: Structured text with separators
- JSONL format recommended for large datasets

**URL:** https://github.com/patil-suraj/question_generation

---

### 5.4 MCQ Generation Literature

**Reference:** Automatic Generation of Multiple-Choice Questions (arXiv:2303.14576)

**Key Findings:**
- Two-stage approach: QG + Distractor generation
- T5 can be fine-tuned for end-to-end MCQ generation
- Output format: Structured text (question + answer + distractors)
- Evaluation: BLEU, ROUGE, human evaluation

---

### 5.5 HuggingFace Question Answering Documentation

**Reference:** HuggingFace Transformers - Seq2Seq QA

**Best Practices:**
- JSONL format preferred for streaming
- Separate input/output fields
- Tokenization: max_length=512 for input, 256 for output
- Batch size: 16-32 for fine-tuning

**URL:** https://huggingface.co/docs/transformers/tasks/question_answering



---

## 7. EXAMPLE JSONL FILE

**File: train.jsonl**

```jsonl
{"input": "buat_soal_pilihan_ganda: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1.", "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1"}
{"input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```", "output": "question: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```\nApa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9"}
{"input": "buat_soal_pilihan_ganda: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat.", "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError"}
```

---

## 4. ATURAN PENTING: CODE BLOCKS DALAM QUESTION

### 4.1 Prinsip Self-Contained Output

**CRITICAL RULE:** Output (question) harus **self-contained** dan tidak bergantung pada input!

**❌ SALAH:**
```json
{
  "input": "buat_soal_pilihan_ganda: Kode berikut: ```python\nx = 6\n```",
  "output": "question: Apa tipe data dari variabel x pada kode di atas?\nanswer: int"
}
```
**Masalah:** "kode di atas" tidak ada dalam question!

**✅ BENAR:**
```json
{
  "input": "buat_soal_pilihan_ganda: Kode berikut: ```python\nx = 6\n```",
  "output": "question: Perhatikan kode berikut:\n```python\nx = 6\n```\nApa tipe data dari variabel x?\nanswer: int"
}
```

### 4.2 Kapan Menyertakan Code Block?

**Sertakan code block jika:**
- Question merujuk "kode di atas/berikut"
- Question meminta output/hasil eksekusi
- Question meminta analisis code

**Tidak perlu jika:**
- Question hanya menyebutkan syntax inline: `print()`
- Question konseptual tanpa code spesifik

### 4.3 Input Context Harus Lengkap dan Deskriptif

**CRITICAL RULE:** Input context harus memberikan penjelasan yang cukup, tidak hanya topik singkat!

**❌ SALAH (terlalu ringkas):**
```json
{
  "input": "buat_soal_pilihan_ganda: Aksi sekuensial",
  "output": "question: Apa yang dimaksud dengan aksi sekuensial?\nanswer: ..."
}
```

**✅ BENAR (lengkap dan deskriptif):**
```json
{
  "input": "buat_soal_pilihan_ganda: Aksi sekuensial adalah sederetan instruksi yang akan dijalankan oleh komputer berdasarkan urutan penulisannya. Dalam Python, program dijalankan secara sekuensial dari atas ke bawah.",
  "output": "question: Apa yang dimaksud dengan aksi sekuensial?\nanswer: ..."
}
```

**Untuk code blocks, SELALU sertakan penjelasan:**

**❌ SALAH:**
```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nx = 5\nprint(x)\n```",
  "output": "question: Perhatikan kode berikut:\n```python\nx = 5\nprint(x)\n```\nApa output dari kode di atas?\nanswer: 5"
}
```

**✅ BENAR:**
```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nx = 5\nprint(x)\n```\nKode di atas menyimpan nilai 5 ke variabel x, kemudian menampilkan nilai x ke layar.",
  "output": "question: Perhatikan kode berikut:\n```python\nx = 5\nprint(x)\n```\nApa output dari kode di atas?\nanswer: 5"
}
```

**Prinsip Input Context:**
- Minimal 1-2 kalimat penjelasan
- Jelaskan konsep atau proses yang terjadi
- Untuk code: jelaskan apa yang dilakukan kode tersebut
- Berikan konteks yang cukup untuk memahami topik

### 4.4 Keseimbangan Soal Konseptual dan Code Blocks

**CRITICAL RULE:** Dataset harus seimbang antara soal konseptual (text/penjelasan) dan soal code blocks!

**Target Rasio:**
- Soal konseptual (text): ≥ 50%
- Soal code blocks: ≤ 50%
- Idealnya: 60% konseptual, 40% code blocks

**Mengapa Penting?**
- Model perlu memahami konsep teori, bukan hanya code
- Mencegah overfitting pada pattern code
- Meningkatkan kemampuan generalisasi
- Dataset lebih seimbang dan komprehensif

**Contoh Distribusi yang BAIK (120 samples):**
- 70 soal konseptual (definisi, konsep, teori)
- 50 soal code blocks (output, analisis, eksekusi)

**Contoh Distribusi yang BURUK:**
- 30 soal konseptual
- 90 soal code blocks ❌ (terlalu banyak code)

**Tips:**
- Buat soal tentang definisi, konsep, dan teori terlebih dahulu
- Tambahkan soal code blocks untuk validasi praktis
- Jangan membuat terlalu banyak variasi code yang mirip
- Fokus pada pemahaman konsep, bukan hanya sintaks

### 4.5 Knowledge-Only Dataset (Khusus Materi Tertentu)

**SPECIAL CASE:** Untuk materi rangkuman/review ATAU materi yang secara eksplisit diminta knowledge-only, dataset dapat dibuat 100% konseptual tanpa code blocks.

**Karakteristik:**
- 100% soal konseptual/knowledge-based
- Tidak ada code blocks dalam input maupun output
- Fokus pada pemahaman teori, definisi, dan konsep
- Cocok untuk materi rangkuman yang merangkum konsep-konsep sebelumnya

**Contoh Knowledge-Only Sample:**
```json
{
  "input": "buat_soal_pilihan_ganda: Matriks dalam matematika merupakan himpunan bilangan atau elemen yang disusun berdasarkan baris dan kolom. Setiap elemen dapat diakses melalui metode indexing jika kedua indeks diketahui.",
  "output": "question: Apa yang dimaksud dengan matriks dalam matematika?\nanswer: Himpunan bilangan atau elemen yang disusun berdasarkan baris dan kolom\ndistractors: Kumpulan data yang disusun secara acak | Array satu dimensi dengan banyak elemen | Struktur data yang hanya memiliki baris"
}
```

**Kapan Menggunakan:**
- Materi rangkuman/review
- Materi yang fokus pada teori dan konsep
- Materi pengenalan tanpa implementasi praktis
- Materi yang secara eksplisit diminta knowledge-only oleh pengguna (misalnya: inheritance, duck typing, prosedur, dll.)

**Daftar Materi yang Menggunakan Knowledge-Only (Confirmed):**
- `09-oop/03-inheritence.md` → Knowledge-only (konsep pewarisan, override, super)
- `09-oop/01-duck-typing.md` → Knowledge-only (konsep duck typing)

**Aturan Tambahan untuk Knowledge-Only:**
- Input context harus menjelaskan konsep secara lengkap dalam 2-3 kalimat
- Distractor harus plausibel secara konseptual, bukan hanya kata-kata acak
- Soal harus menguji pemahaman konsep, bukan hafalan istilah semata
- Variasikan tipe pertanyaan: definisi, perbandingan, tujuan/manfaat, mekanisme

---

## 8. COMPARISON: JSON vs JSONL vs CSV

| Criteria | JSON | JSONL | CSV |
|----------|------|-------|-----|
| **Format** | Single array | Line-delimited | Comma-separated |
| **Streaming** | ❌ No | ✅ Yes | ✅ Yes |
| **Memory** | High | Low | Low |
| **HuggingFace** | ⚠️ Limited | ✅ Preferred | ⚠️ Limited |
| **Nested data** | ✅ Yes | ✅ Yes | ❌ No |
| **Large datasets** | ❌ Problematic | ✅ Efficient | ✅ Efficient |
| **Your choice** | ❌ | ✅ | ⚠️ |

**Recommendation:** Use **JSONL** for IndoNanoT5

---

## 9. TROUBLESHOOTING

### Problem: Model generates repetitive output

**Cause:** Markdown formatting in input confuses model
**Solution:** Remove all markdown (##, **, __, etc.)

### Problem: Training loss = 0

**Cause:** Duplicate samples or incorrect label masking
**Solution:** Deduplicate dataset, verify DataCollator

### Problem: Eval loss = NaN

**Cause:** Numerical instability or incorrect output format
**Solution:** Check output format, verify tokenization

### Problem: Model doesn't understand task

**Cause:** Missing or incorrect task prefix
**Solution:** Ensure all inputs start with `buat_soal_pilihan_ganda:`

---

**Version:** 1.0  
**Last Updated:** 22 April 2026  
**Status:** READY FOR IMPLEMENTATION

