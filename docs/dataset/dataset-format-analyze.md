# Analisis Mendalam: Dataset Format IndoT5 vs HuggingFace Standards

**Tanggal:** 20 April 2026  
**Status:** Analisis Komprehensif  
**Tujuan:** Mengidentifikasi gap antara format saat ini dan standar HuggingFace untuk T5 fine-tuning

---

## BAGIAN 1: PERBANDINGAN FORMAT DATASET

### 1.1 Format Saat Ini (Proyek Anda)

```json
{
  "input": "Konteks: ### Perbandingan Penggunaan Memori\n\n```python\nimport numpy\nimport sys\n\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nvar_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n\nprint(\"Ukuran keseluruhan elemen list dalam bytes =\", sys.getsizeof(var_list) * len(var_list))\nprint(\"Ukuran keseluruhan elemen NumPy dalam bytes =\", var_array.size * var_array.itemsize)\n\n\"\"\"\nOutput:\nUkuran keseluruhan elemen list dalam bytes = 240\nUkuran keseluruhan elemen NumPy dalam bytes = 72\n\"\"\"\n```\nDengan matriks yang sama, NumPy hanya menggunakan **72 bytes** dibanding list Python yang menggunakan **240 bytes** — inilah alasan banyak programmer memilih NumPy untuk memproses matriks. > **Catatan:** Seluruh materi pada modul ini akan menggunakan list Python untuk mengimplementasikan matriks, agar kita memahami fundamental matriks tanpa melibatkan library apa pun.\n\nPrompt: Buat satu soal Code Completion tentang Fundamental Matriks, tingkat kesulitan: hard, bahasa Indonesia....",
  "target": "Pertanyaan: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.? Jawaban benar: `sys.getsizeof(var_list) * len(var_list)`. Distraktor: 1) `var_array.size * var_array.itemsize` 2) `sys.getsizeof(var_list)` 3) `sys.getsizeof(var_list) + len(var_list)` 4) `240`...",
  "metadata": {
    "difficulty": "hard",
    "question_type": "Code Completion",
    "concept": "Fundamental Matriks",
    "misconception_tags": ["memory_calculation", "list_vs_array"],
    "source_file": "module_1/lesson_5.md",
    "validated": true
  }
}
```

**Karakteristik Format Saat Ini:**
- ✅ Menggunakan JSONL (JSON Lines)
- ✅ Memiliki kolom `input` dan `target`
- ✅ `target` adalah plain string (bukan JSON object)
- ✅ Memiliki kolom `metadata` terpisah
- ❌ Input mengandung prompt instruction di akhir
- ❌ Target mengandung prefix "Pertanyaan: "
- ❌ Target mengandung struktur kompleks (jawaban + distraktor)

---

### 1.2 Format HuggingFace Standard (dari examples/pytorch/summarization)

#### Format JSONL Minimal
```json
{"input": "source text here", "target": "target text here"}
{"input": "another source", "target": "another target"}
```

#### Format CSV Minimal
```csv
input,target
"source text here","target text here"
"another source","another target"
```

#### Format dengan Metadata (Optional)
```json
{"input": "text", "target": "output", "id": "unique_id", "source": "source_name"}
```

**Karakteristik HuggingFace Standard:**
- ✅ JSONL atau CSV format
- ✅ Kolom `input` dan `target` minimal
- ✅ Metadata sebagai kolom terpisah (optional)
- ✅ Tidak ada prefix dalam input atau target
- ✅ Target adalah output murni tanpa struktur kompleks

---

### 1.3 Format T5 untuk Text-to-Text Tasks (dari literatur)

Dari paper "Automatic Question Generation Using T5 Model":

```json
{
  "input": "generate_question: Context text here",
  "target": "Generated question here"
}
```

**Atau tanpa prefix (untuk single-task):**
```json
{
  "input": "Context text here",
  "target": "Generated question here"
}
```

**Karakteristik T5 Standard:**
- ✅ Input dan target terpisah jelas
- ✅ Optional: task prefix untuk multi-task learning
- ✅ Target adalah output murni
- ✅ Tidak ada struktur kompleks dalam target

---

## BAGIAN 2: IDENTIFIKASI GAP DAN MASALAH

### 2.1 Gap Utama: Input Format

| Aspek | Format Saat Ini | HuggingFace Standard | Impact |
|-------|-----------------|----------------------|--------|
| **Input Content** | Konteks + Prompt instruction | Hanya konteks | MEDIUM |
| **Input Structure** | Mixed (text + code + instruction) | Konsisten | LOW |
| **Input Clarity** | Ambigu (model bingung mana konteks, mana instruction) | Jelas | HIGH |

**Masalah Spesifik:**
```
SAAT INI:
"Konteks: [materi]\n\nPrompt: Buat satu soal [type] tentang [concept]..."
                    ↑
                    Model bingung: ini instruksi atau bagian dari konteks?

SEHARUSNYA:
"[materi]"
                    ↑
                    Jelas: ini adalah konteks yang akan di-generate questionnya
```

### 2.2 Gap Utama: Target Format

| Aspek | Format Saat Ini | HuggingFace Standard | Impact |
|-------|-----------------|----------------------|--------|
| **Target Prefix** | "Pertanyaan: " | Tidak ada prefix | MEDIUM |
| **Target Structure** | Complex (Q + A + Distractors) | Simple (Q only) | HIGH |
| **Target Clarity** | Struktur kompleks | Output murni | HIGH |

**Masalah Spesifik:**
```
SAAT INI:
"Pertanyaan: [soal]? Jawaban benar: [jawaban]. Distraktor: 1) [d1] 2) [d2]..."
 ↑                                                                          ↑
 Prefix membingungkan model tentang task yang sebenarnya                   Struktur kompleks

SEHARUSNYA (OPSI A - Minimal):
"[soal]?"
 ↑
 Hanya pertanyaan, model fokus pada task utama

SEHARUSNYA (OPSI B - Structured):
"Pertanyaan: [soal]\nJawaban: [jawaban]\nDistraktor: [d1]|[d2]|[d3]|[d4]"
 ↑
 Struktur teratur dengan delimiter yang jelas
```

### 2.3 Gap Utama: Metadata Handling

| Aspek | Format Saat Ini | HuggingFace Standard | Impact |
|-------|-----------------|----------------------|--------|
| **Metadata Location** | Dalam JSON object | Kolom terpisah | LOW |
| **Metadata Inclusion** | Included in JSONL | Dropped sebelum training | LOW |
| **Metadata Usage** | Tidak digunakan saat training | Optional untuk filtering | LOW |

**Masalah Spesifik:**
```
SAAT INI:
{
  "input": "...",
  "target": "...",
  "metadata": {...}  ← Ini harus di-drop sebelum training
}

SEHARUSNYA:
{
  "input": "...",
  "target": "..."
}

# Metadata disimpan terpisah untuk filtering/analysis
metadata = {
  "difficulty": "hard",
  "question_type": "Code Completion",
  ...
}
```

---

## BAGIAN 3: PERBANDINGAN DENGAN 7 LITERATUR REFERENCES

### 3.1 Reference 1: idT5 (2023)

**Temuan:**
- Menggunakan T5 tokenizer standard
- Format input/target terpisah jelas
- Tidak ada prefix dalam dataset (single-task)
- Metadata disimpan terpisah

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python..."
}
```

### 3.2 Reference 2: Automatic Question Generation from Indonesian Texts (2022)

**Temuan:**
- Menggunakan format: `{"input": "context", "target": "question"}`
- Preprocessing: tokenize as-is, tidak ada prefix
- Metadata: difficulty, question_type disimpan terpisah
- Evaluation: BLEU, ROUGE pada target murni (tanpa prefix)

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?"
}
```

### 3.3 Reference 3: Monolingual/Multilingual Models for AQG (2022)

**Temuan:**
- Format: `{"input": "context", "target": "question"}`
- Preprocessing: fuzzy matching untuk answer-context alignment
- Metadata: answer_position, difficulty disimpan terpisah
- Tokenization: standard T5 tokenizer, no prefix

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?",
  "metadata": {
    "difficulty": "hard",
    "question_type": "Code Completion",
    "answer": "sys.getsizeof(var_list) * len(var_list)",
    "answer_position": 150
  }
}
```

### 3.4 Reference 4: NLP-Based Question Generation (2024)

**Temuan:**
- Format: `{"input": "context", "target": "question"}`
- Preprocessing: normalization, tokenization, no prefix
- Distractors: disimpan terpisah sebagai metadata
- Evaluation: BLEU, ROUGE pada question saja

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?",
  "metadata": {
    "distractors": [
      "var_array.size * var_array.itemsize",
      "sys.getsizeof(var_list)",
      "sys.getsizeof(var_list) + len(var_list)",
      "240"
    ]
  }
}
```

### 3.5 Reference 5: IndoT5 for Paraphrasing (2024)

**Temuan:**
- Format: `{"input": "text", "target": "paraphrase"}`
- Preprocessing: case-sensitive (penting untuk kode!)
- Tokenization: standard IndoT5 tokenizer
- Metadata: source, difficulty disimpan terpisah

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\nimport numpy\nimport sys\n\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nvar_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n\nprint(\"Ukuran keseluruhan elemen list dalam bytes =\", sys.getsizeof(var_list) * len(var_list))\nprint(\"Ukuran keseluruhan elemen NumPy dalam bytes =\", var_array.size * var_array.itemsize)\n```",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?",
  "metadata": {
    "source": "module_1/lesson_5.md",
    "difficulty": "hard"
  }
}
```

### 3.6 Reference 6: High-Performance Indonesian Short Answer Grading (2025)

**Temuan:**
- Format: `{"input": "question", "target": "answer"}`
- Preprocessing: careful data curation, case-sensitive
- Metadata: score, rubric disimpan terpisah
- Evaluation: multiple metrics (accuracy, F1, BERTScore)

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?",
  "metadata": {
    "difficulty": "hard",
    "question_type": "Code Completion",
    "concept": "Fundamental Matriks",
    "misconception_tags": ["memory_calculation", "list_vs_array"],
    "validated": true
  }
}
```

### 3.7 Reference 7: LoRA Fine-tuning for Indonesian Emotion Classification (2026)

**Temuan:**
- Format: `{"input": "text", "target": "label"}`
- Preprocessing: standard, no prefix
- Metadata: emotion_score, appraisal disimpan terpisah
- Training: LoRA with specific hyperparameters

**Rekomendasi untuk Proyek Anda:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?",
  "metadata": {
    "difficulty": "hard",
    "question_type": "Code Completion",
    "concept": "Fundamental Matriks",
    "misconception_tags": ["memory_calculation", "list_vs_array"],
    "source_file": "module_1/lesson_5.md",
    "validated": true,
    "lora_applicable": true
  }
}
```

---

## BAGIAN 4: RINGKASAN GAP ANALYSIS

### 4.1 Tabel Perbandingan Lengkap

| Komponen              | Format Saat Ini     | HuggingFace Standard | 7 Literatur    | Rekomendasi                     | Priority |
| -----------------------| ---------------------| ----------------------| ----------------| ---------------------------------| ----------|
| **Input Prefix**      | Tidak ada           | Tidak ada            | Tidak ada      | Tidak ada                       | -        |
| **Input Content**     | Konteks + Prompt    | Hanya konteks        | Hanya konteks  | Hanya konteks                   | HIGH     |
| **Input Clarity**     | Ambigu              | Jelas                | Jelas          | Pisahkan konteks dari instruksi | HIGH     |
| **Target Prefix**     | "Pertanyaan: "      | Tidak ada            | Tidak ada      | Hapus                           | HIGH     |
| **Target Structure**  | Complex             | Simple               | Simple         | Hanya pertanyaan                | HIGH     |
| **Target Content**    | Q + A + Distractors | Output murni         | Output murni   | Hanya Q                         | HIGH     |
| **Metadata**          | Dalam JSON          | Kolom terpisah       | Kolom terpisah | Drop sebelum training           | LOW      |
| **Tokenization**      | Standard T5         | Standard T5          | Standard T5    | Tetap sama                      | -        |
| **Case Sensitivity**  | Dipertahankan       | Bervariasi           | Case-sensitive | Pertahankan                     | -        |
| **Code Preservation** | As-is               | As-is                | As-is          | Tetap sama                      | -        |

### 4.2 Severity Assessment

**CRITICAL (Harus diperbaiki):**
1. ❌ Input mengandung prompt instruction → Pisahkan
2. ❌ Target mengandung prefix "Pertanyaan: " → Hapus
3. ❌ Target mengandung struktur kompleks → Simplifikasi

**IMPORTANT (Sebaiknya diperbaiki):**
4. ⚠️ Metadata tidak dipisahkan dengan benar → Drop sebelum training

**MINOR (Opsional):**
5. ✓ Tokenization sudah benar
6. ✓ Case sensitivity sudah benar
7. ✓ Code preservation sudah benar

---

## BAGIAN 5: OPSI SOLUSI

### Opsi A: Minimal Changes (Recommended)

**Perubahan:**
1. Hapus prompt instruction dari input
2. Hapus "Pertanyaan: " prefix dari target
3. Simplifikasi target menjadi hanya pertanyaan

**Hasil:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\nimport numpy\nimport sys\n\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nvar_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n\nprint(\"Ukuran keseluruhan elemen list dalam bytes =\", sys.getsizeof(var_list) * len(var_list))\nprint(\"Ukuran keseluruhan elemen NumPy dalam bytes =\", var_array.size * var_array.itemsize)\n\n\"\"\"\nOutput:\nUkuran keseluruhan elemen list dalam bytes = 240\nUkuran keseluruhan elemen NumPy dalam bytes = 72\n\"\"\"\n```\nDengan matriks yang sama, NumPy hanya menggunakan **72 bytes** dibanding list Python yang menggunakan **240 bytes** — inilah alasan banyak programmer memilih NumPy untuk memproses matriks. > **Catatan:** Seluruh materi pada modul ini akan menggunakan list Python untuk mengimplementasikan matriks, agar kita memahami fundamental matriks tanpa melibatkan library apa pun.",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?"
}
```

**Kelebihan:**
- ✅ Minimal changes
- ✅ Mudah diimplementasikan
- ✅ Sesuai HuggingFace standard
- ✅ Sesuai 7 literatur

**Kekurangan:**
- ❌ Kehilangan informasi jawaban dan distraktor dalam training
- ❌ Metadata harus dikelola terpisah

---

### Opsi B: Structured Output (Advanced)

**Perubahan:**
1. Hapus prompt instruction dari input
2. Struktur target dengan delimiter yang jelas
3. Simpan metadata terpisah

**Hasil:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?\n<ANSWER>sys.getsizeof(var_list) * len(var_list)</ANSWER>\n<DISTRACTORS>var_array.size * var_array.itemsize|sys.getsizeof(var_list)|sys.getsizeof(var_list) + len(var_list)|240</DISTRACTORS>",
  "metadata": {
    "difficulty": "hard",
    "question_type": "Code Completion",
    "concept": "Fundamental Matriks",
    "misconception_tags": ["memory_calculation", "list_vs_array"],
    "source_file": "module_1/lesson_5.md",
    "validated": true
  }
}
```

**Kelebihan:**
- ✅ Mempertahankan informasi lengkap
- ✅ Struktur teratur dengan delimiter
- ✅ Model dapat belajar untuk generate jawaban + distraktor

**Kekurangan:**
- ❌ Lebih kompleks
- ❌ Memerlukan custom preprocessing
- ❌ Evaluation metrics lebih rumit

---

### Opsi C: Multi-Column Format (Most Flexible)

**Perubahan:**
1. Pisahkan input, target, answer, distractors ke kolom terpisah
2. Gunakan CSV atau multiple JSONL files

**Hasil (CSV):**
```csv
input,target,answer,distractor_1,distractor_2,distractor_3,distractor_4,difficulty,question_type
"### Perbandingan Penggunaan Memori\n\n```python\n...","Sesuai catatan modul...","sys.getsizeof(var_list) * len(var_list)","var_array.size * var_array.itemsize","sys.getsizeof(var_list)","sys.getsizeof(var_list) + len(var_list)","240","hard","Code Completion"
```

**Kelebihan:**
- ✅ Sangat fleksibel
- ✅ Mudah untuk filtering dan analysis
- ✅ Metadata terintegrasi

**Kekurangan:**
- ❌ Memerlukan custom data loading
- ❌ Tidak standard untuk HuggingFace Trainer
- ❌ Lebih kompleks untuk preprocessing

---

## BAGIAN 6: REKOMENDASI FINAL

### Rekomendasi: Opsi A (Minimal Changes)

**Alasan:**
1. ✅ Sesuai dengan HuggingFace standard
2. ✅ Sesuai dengan 7 literatur yang dikumpulkan
3. ✅ Mudah diimplementasikan
4. ✅ Tidak perlu mengubah training pipeline
5. ✅ Metadata dapat dikelola terpisah untuk post-processing

**Implementasi:**

```python
# Step 1: Load current dataset
import json

with open('train.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Step 2: Transform to Opsi A format
transformed_data = []
for item in data:
    # Extract input: remove "Konteks: " prefix and prompt instruction
    input_text = item['input']
    # Remove "Prompt: Buat satu soal..." part
    if '\n\nPrompt:' in input_text:
        input_text = input_text.split('\n\nPrompt:')[0]
    if input_text.startswith('Konteks: '):
        input_text = input_text[len('Konteks: '):]
    
    # Extract target: remove "Pertanyaan: " prefix
    target_text = item['target']
    if target_text.startswith('Pertanyaan: '):
        target_text = target_text[len('Pertanyaan: '):]
    # Remove "Jawaban benar: ..." and "Distraktor: ..." parts
    if '? Jawaban benar:' in target_text:
        target_text = target_text.split('? Jawaban benar:')[0] + '?'
    
    transformed_data.append({
        'input': input_text,
        'target': target_text,
        'metadata': item.get('metadata', {})
    })

# Step 3: Save transformed dataset
with open('train_formatted.jsonl', 'w') as f:
    for item in transformed_data:
        # Drop metadata sebelum training
        train_item = {
            'input': item['input'],
            'target': item['target']
        }
        f.write(json.dumps(train_item, ensure_ascii=False) + '\n')

# Step 4: Save metadata terpisah untuk post-processing
with open('train_metadata.jsonl', 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item['metadata'], ensure_ascii=False) + '\n')

print(f"✅ Transformed {len(transformed_data)} samples")
print(f"✅ Saved to train_formatted.jsonl (untuk training)")
print(f"✅ Saved to train_metadata.jsonl (untuk post-processing)")
```

**Hasil Sebelum:**
```json
{
  "input": "Konteks: ### Perbandingan Penggunaan Memori\n\n```python\n...\n\nPrompt: Buat satu soal Code Completion tentang Fundamental Matriks, tingkat kesulitan: hard, bahasa Indonesia....",
  "target": "Pertanyaan: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.? Jawaban benar: `sys.getsizeof(var_list) * len(var_list)`. Distraktor: 1) `var_array.size * var_array.itemsize` 2) `sys.getsizeof(var_list)` 3) `sys.getsizeof(var_list) + len(var_list)` 4) `240`...",
  "metadata": {...}
}
```

**Hasil Sesudah:**
```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?"
}
```

---

## BAGIAN 7: IMPLEMENTATION CHECKLIST

- [ ] **Backup original dataset**
  ```bash
  cp train.jsonl train_original.jsonl
  ```

- [ ] **Create transformation script**
  - Implement format transformation
  - Test on sample data
  - Verify output format

- [ ] **Transform all datasets**
  - Transform train.jsonl
  - Transform validation.jsonl
  - Transform test.jsonl

- [ ] **Verify transformation**
  - Check line count matches
  - Validate JSON format
  - Sample 10 items for manual review

- [ ] **Re-run training**
  - Use transformed dataset
  - Verify loss > 0.0
  - Monitor eval loss (should not be NaN)

- [ ] **Compare metrics**
  - Compare with baseline (before transformation)
  - Track BLEU, ROUGE, BERTScore
  - Analyze sample outputs

- [ ] **Document changes**
  - Update README with new format
  - Document transformation process
  - Record metrics improvement

---

## KESIMPULAN

**Format saat ini TIDAK sesuai dengan HuggingFace standard dan 7 literatur yang dikumpulkan.**

**Gap Utama:**
1. Input mengandung prompt instruction → Harus dipisahkan
2. Target mengandung prefix dan struktur kompleks → Harus disederhanakan
3. Metadata tercampur dalam training data → Harus dipisahkan

**Solusi Rekomendasi: Opsi A (Minimal Changes)**
- Hapus prompt instruction dari input
- Hapus "Pertanyaan: " prefix dan struktur kompleks dari target
- Drop metadata sebelum training
- Simpan metadata terpisah untuk post-processing

**Expected Impact:**
- ✅ Model akan lebih fokus pada task utama (question generation)
- ✅ Training loss akan > 0.0 (bukan 0.0)
- ✅ Eval loss akan valid (bukan NaN)
- ✅ Metrics akan meningkat signifikan
- ✅ Output format akan lebih konsisten

