# Dataset Design Guide: IndoNanoT5 MCQ Generation

**Status:** FINAL DESIGN  
**Task Type:** Multiple Choice Question Generation (MCQ-G)  
**Model:** LazarusNLP/IndoNanoT5-base  
**Format:** JSONL (JSON Lines)  
**Date:** 22 April 2026  
**Version:** 2.0

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

### Sample 1: Knowledge-Based Question

```json
{
  "input": "buat_soal_pilihan_ganda: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1. Contoh: jika Anda memiliki list = [10, 20, 30], maka list[0] adalah 10, list[1] adalah 20, dan list[2] adalah 30.",
  "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1",
  "metadata": {"difficulty": "Mudah", "type": "knowledge", "source_section": "06-array", "source_file": "materi1.jsonl"}
}
```

### Sample 2: Code Execution Question

```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(var_list[1][2])\n```\nKode ini mengakses elemen pada baris kedua (indeks 1) dan kolom ketiga (indeks 2) dari nested list.",
  "output": "question: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(var_list[1][2])\n```\nApa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9",
  "metadata": {"difficulty": "Sedang", "type": "code", "source_section": "07-matriks", "source_file": "materi1.jsonl"}
}
```

**PENTING:** Question harus menyertakan code block agar self-contained!

### Sample 3: Conceptual Question

```json
{
  "input": "buat_soal_pilihan_ganda: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat. Jika Anda mencoba mengubah karakter dalam string dengan indexing seperti s[0] = 'A', Python akan menampilkan error TypeError. Untuk mengubah string, Anda harus membuat string baru.",
  "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string dengan s[0] = 'A'?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError",
  "metadata": {"difficulty": "Sedang", "type": "knowledge", "source_section": "05-control-flow", "source_file": "materi2.jsonl"}
}
```

---

## 4. METADATA STRUCTURE

### 4.1 Required Metadata Fields

Every dataset sample MUST include metadata with the following fields:

```json
{
  "input": "...",
  "output": "...",
  "metadata": {
    "difficulty": "Mudah|Sedang|Sulit",
    "type": "knowledge|code",
    "source_section": "section-name",
    "source_file": "filename.jsonl"
  }
}
```

### 4.2 Metadata Field Definitions

**difficulty** (Required)
- Values: `"Mudah"`, `"Sedang"`, `"Sulit"`
- Indicates cognitive complexity of the question
- Guidelines:
  - **Mudah**: Direct recall, simple concepts, basic syntax
  - **Sedang**: Application, analysis, multi-step reasoning
  - **Sulit**: Synthesis, complex scenarios, edge cases

**type** (Required)
- Values: `"knowledge"`, `"code"`
- Indicates question category
- Guidelines:
  - **knowledge**: Conceptual questions, definitions, theory, explanations
  - **code**: Questions with code blocks, output prediction, code analysis

**source_section** (Required)
- Format: kebab-case string
- Example: `"06-array"`, `"08-subprogram"`, `"09-oop"`
- Tracks which learning module the question comes from

**source_file** (Required)
- Format: filename with extension
- Example: `"materi1.jsonl"`, `"materi2.jsonl"`
- Tracks source file for traceability

### 4.3 Type Distribution Requirements

**CRITICAL RULE:** Dataset MUST maintain balanced type distribution!

**Target Ratio:**
- `type: "knowledge"` → **≥ 60%** of total samples
- `type: "code"` → **≤ 40%** of total samples

**Example Distribution (100 samples):**
- 60+ knowledge questions ✅
- 40 or fewer code questions ✅

**Why This Ratio?**
- Model needs strong conceptual understanding
- Prevents overfitting on code patterns
- Improves generalization across question types
- Balances theory and practice

### 4.4 Metadata Examples

**Knowledge-Type Sample:**
```json
{
  "input": "buat_soal_pilihan_ganda: Variadic function adalah fungsi yang dapat menerima jumlah argumen yang bervariasi. Dalam Python, variadic function dibuat dengan *args dan **kwargs.",
  "output": "question: Apa yang dimaksud dengan variadic function?\nanswer: Fungsi yang dapat menerima jumlah argumen yang bervariasi\ndistractors: Fungsi yang hanya menerima satu argumen | Fungsi yang tidak menerima argumen | Fungsi yang selalu error",
  "metadata": {
    "difficulty": "Sedang",
    "type": "knowledge",
    "source_section": "08-subprogram",
    "source_file": "materi4.jsonl"
  }
}
```

**Code-Type Sample:**
```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_mat = [[10, 20], [30, 40]]\nprint(var_mat[0][1])\n```\nKode ini mengakses elemen pada baris pertama, kolom kedua.",
  "output": "question: Perhatikan kode berikut:\n```python\nvar_mat = [[10, 20], [30, 40]]\nprint(var_mat[0][1])\n```\nApa output dari kode tersebut?\nanswer: 20\ndistractors: 10 | 30 | 40",
  "metadata": {
    "difficulty": "Mudah",
    "type": "code",
    "source_section": "07-matriks",
    "source_file": "materi2.jsonl"
  }
}
```

---

## 5. ATURAN PENTING: CODE BLOCKS DALAM QUESTION

### 5.1 Prinsip Self-Contained Output

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

### 5.2 Kapan Menyertakan Code Block?

**Sertakan code block jika:**
- Question merujuk "kode di atas/berikut"
- Question meminta output/hasil eksekusi
- Question meminta analisis code

**Tidak perlu jika:**
- Question hanya menyebutkan syntax inline: `print()`
- Question konseptual tanpa code spesifik

### 5.3 Input Context Harus Lengkap dan Deskriptif

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

### 5.4 Type Distribution Validation

**CRITICAL RULE:** Always validate type distribution before finalizing dataset!

**Validation Steps:**
1. Count total samples in dataset
2. Count samples with `type: "knowledge"`
3. Count samples with `type: "code"`
4. Calculate percentages:
   - Knowledge % = (knowledge_count / total) × 100
   - Code % = (code_count / total) × 100
5. Verify: Knowledge % ≥ 60% AND Code % ≤ 40%

**Example Validation (120 samples):**
```
Total: 120
Knowledge: 72 (60%) ✅
Code: 48 (40%) ✅
Status: VALID
```

**Example Invalid (120 samples):**
```
Total: 120
Knowledge: 50 (42%) ❌
Code: 70 (58%) ❌
Status: INVALID - Too many code questions!
Action: Add 22 knowledge questions, remove 22 code questions
```

**Tips for Maintaining Ratio:**
- Create knowledge questions first (definitions, concepts, theory)
- Add code questions for practical validation
- Don't create too many code variations
- Focus on conceptual understanding over syntax

---

## 6. EXAMPLE JSONL FILE

**File: train.jsonl**

```jsonl
{"input": "buat_soal_pilihan_ganda: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1.", "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1", "metadata": {"difficulty": "Mudah", "type": "knowledge", "source_section": "06-array", "source_file": "materi1.jsonl"}}
{"input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```\nKode ini mengakses elemen baris kedua, kolom ketiga.", "output": "question: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```\nApa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9", "metadata": {"difficulty": "Sedang", "type": "code", "source_section": "07-matriks", "source_file": "materi2.jsonl"}}
{"input": "buat_soal_pilihan_ganda: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat. Untuk mengubah string, Anda harus membuat string baru.", "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError", "metadata": {"difficulty": "Sedang", "type": "knowledge", "source_section": "05-control-flow", "source_file": "materi2.jsonl"}}
```

---

## 7. REFERENSI

### 7.1 T5 Text-to-Text Framework

**Reference:** Raffel, C., Shazeer, N., Roberts, A., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." arXiv:1910.10683

**Key Points:**
- T5 converts all NLP tasks to text-to-text format
- Task prefixes guide model behavior
- No task-specific architectures needed
- Format: `[prefix]: [input] → [output]`

---

### 7.2 IndoNanoT5 Documentation

**Reference:** LazarusNLP/IndoNanoT5-base (HuggingFace Hub)

**Model Details:**
- Architecture: T5 (Encoder-Decoder)
- Language: Indonesian
- Pretraining: 4B tokens from CulturaX corpus
- Input length: 512 tokens
- License: Apache 2.0

**URL:** https://huggingface.co/LazarusNLP/IndoNanoT5-base

---

### 7.3 Question Generation with T5

**Reference:** Patil, S. (2021). "Question Generation using Transformers." GitHub Repository

**Key Insights:**
- T5 can handle multiple QG variants (answer-aware, end-to-end, MCQ)
- Task prefixes: `generate_question:`, `generate_mcq:`, etc.
- Output format: Structured text with separators
- JSONL format recommended for large datasets

**URL:** https://github.com/patil-suraj/question_generation

---

### 7.4 MCQ Generation Literature

**Reference:** Automatic Generation of Multiple-Choice Questions (arXiv:2303.14576)

**Key Findings:**
- Two-stage approach: QG + Distractor generation
- T5 can be fine-tuned for end-to-end MCQ generation
- Output format: Structured text (question + answer + distractors)
- Evaluation: BLEU, ROUGE, human evaluation

---

### 7.5 HuggingFace Question Answering Documentation

**Reference:** HuggingFace Transformers - Seq2Seq QA

**Best Practices:**
- JSONL format preferred for streaming
- Separate input/output fields
- Tokenization: max_length=512 for input, 256 for output
- Batch size: 16-32 for fine-tuning

**URL:** https://huggingface.co/docs/transformers/tasks/question_answering

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

### Problem: Type distribution imbalanced

**Cause:** Too many code questions, not enough knowledge questions
**Solution:** Follow validation steps in Section 5.4, maintain 60/40 ratio

---

## 10. CHECKLIST VALIDASI DATASET

Before finalizing dataset, verify:

- [ ] All inputs start with `buat_soal_pilihan_ganda:`
- [ ] Input context minimal 1-2 kalimat penjelasan
- [ ] Questions are self-contained (tidak bergantung input)
- [ ] Code blocks di-copy ke question jika dirujuk
- [ ] All samples have complete metadata (difficulty, type, source_section, source_file)
- [ ] Type distribution: knowledge ≥ 60%, code ≤ 40%
- [ ] Plain text format (hapus markdown kecuali code blocks)
- [ ] Output format: `question:`, `answer:`, `distractors:`
- [ ] Distractors plausibel dan relevan
- [ ] No duplicate samples

---

**Version:** 2.0  
**Last Updated:** 1 May 2026  
**Status:** READY FOR IMPLEMENTATION
