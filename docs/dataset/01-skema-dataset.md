# Skema Dataset AQG

**Tanggal:** 21 April 2026  
**Status:** Production Ready  
**Format:** HuggingFace Standard (JSONL)

---

## Overview

Dataset untuk fine-tuning model IndoT5 dalam task Automatic Question Generation (AQG) dari materi pembelajaran Python. Dataset menggunakan format standar HuggingFace dengan pemisahan data dan metadata.

---

## Struktur Direktori

```
dataset_aqg/
├── dataset-task-v2/              # ✅ Format HuggingFace (CURRENT)
│   ├── train.jsonl               # Data training (876 samples)
│   ├── validation.jsonl          # Data validasi (175 samples)
│   ├── test.jsonl                # Data testing (211 samples)
│   ├── train_metadata.jsonl      # Metadata training
│   ├── validation_metadata.jsonl # Metadata validasi
│   └── test_metadata.jsonl       # Metadata testing
│
└── dataset-task-spesifc/         # ❌ Format lama (DEPRECATED)
    └── ...
```

**Total:** 1,262 samples (876 train / 175 val / 211 test)

---

## Format Data

### 1. File Data Utama (`*.jsonl`)

Format standar HuggingFace untuk sequence-to-sequence tasks:

```jsonl
{
  "input": "<konteks_materi>",
  "target": "<pertanyaan>?"
}
```

**Karakteristik:**
- ✅ Plain text dengan markdown formatting
- ✅ Input: konteks materi pembelajaran (tanpa prefix "Konteks:")
- ✅ Target: pertanyaan saja (tanpa "Pertanyaan:", "Jawaban benar:", atau distraktor)
- ✅ Encoding: UTF-8
- ✅ Format: JSONL (satu JSON object per baris)

**Contoh:**

```json
{
  "input": "### Perbandingan Penggunaan Memori\n\n```python\nimport numpy\nimport sys\n\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nvar_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n\nprint(\"Ukuran keseluruhan elemen list dalam bytes =\", sys.getsizeof(var_list) * len(var_list))\nprint(\"Ukuran keseluruhan elemen NumPy dalam bytes =\", var_array.size * var_array.itemsize)\n```",
  "target": "Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?"
}
```

### 2. File Metadata (`*_metadata.jsonl`)

Metadata terpisah untuk analisis dan post-processing:

```jsonl
{
  "difficulty": "easy|medium|hard",
  "question_type": "Multiple Choice|Code Completion|Conceptual|...",
  "concept": "<konsep_utama>",
  "misconception_tags": ["<tag1>", "<tag2>", ...],
  "source_file": "<path_file_materi>",
  "section": "<judul_section>",
  "source": "synthetic",
  "validated": true
}
```

**Contoh:**

```json
{
  "difficulty": "hard",
  "question_type": "Code Completion",
  "concept": "Fundamental Matriks",
  "misconception_tags": ["`miskonsepsi_numpy_formula`", "`lupa_operasi_perkalian`"],
  "source_file": "dataset_aqg/materi/07-matriks/01-fundamental-matriks.md",
  "section": "### Perbandingan Penggunaan Memori",
  "source": "synthetic",
  "validated": true
}
```

---

## Transformasi Format

Dataset ditransformasi dari format custom ke HuggingFace standard:

### Format Lama (DEPRECATED)

```json
{
  "input": "Konteks: <text>\n\nPrompt: Buatlah pertanyaan...",
  "target": "Pertanyaan: <question>? Jawaban benar: <answer>. Distraktor: ...",
  "difficulty": "...",
  "question_type": "...",
  ...
}
```

### Format Baru (CURRENT)

**Data:**
```json
{
  "input": "<text>",
  "target": "<question>?"
}
```

**Metadata (terpisah):**
```json
{
  "difficulty": "...",
  "question_type": "...",
  ...
}
```

**Transformasi dilakukan oleh:** `scripts/transform_dataset.py`

---

## Karakteristik Dataset

### Distribusi Data

| Split      | Samples | Persentase |
|------------|---------|------------|
| Train      | 876     | 69.4%      |
| Validation | 175     | 13.9%      |
| Test       | 211     | 16.7%      |
| **Total**  | **1,262** | **100%** |

### Tipe Pertanyaan

- Multiple Choice
- Code Completion
- Conceptual
- True/False
- Error Identification
- Code Output Prediction

### Tingkat Kesulitan

- Easy: ~30%
- Medium: ~45%
- Hard: ~25%

### Domain

- Fundamental Python (variabel, tipe data, operator)
- Control Flow (if-else, loop)
- Fungsi dan Subprogram
- Struktur Data (list, dictionary, set, tuple)
- Matriks dan NumPy
- File I/O

---

## Markdown Formatting

Dataset **mempertahankan** markdown formatting untuk menjaga struktur semantik:

**Preserved Elements:**
- ✅ Headers (`###`, `##`)
- ✅ Code blocks (` ``` `)
- ✅ Inline code (`` ` ``)
- ✅ Bold (`**text**`)
- ✅ Lists (`-`, `1.`)
- ✅ Links (`[text](url)`)
- ✅ Images (`![alt](path)`)

**Catatan:** Beberapa markdown warnings (unclosed code blocks, unbalanced bold) mungkin ada karena berasal dari data original.

---

## Penggunaan

### Loading Dataset

```python
from datasets import load_dataset

# Load dari file lokal
dataset = load_dataset(
    'json',
    data_files={
        'train': 'dataset_aqg/dataset-task-v2/train.jsonl',
        'validation': 'dataset_aqg/dataset-task-v2/validation.jsonl',
        'test': 'dataset_aqg/dataset-task-v2/test.jsonl'
    }
)

# Akses data
print(dataset['train'][0])
# {'input': '...', 'target': '...'}
```

### Loading Metadata

```python
import json

# Load metadata
with open('dataset_aqg/dataset-task-v2/train_metadata.jsonl', 'r') as f:
    metadata = [json.loads(line) for line in f]

# Filter by difficulty
hard_samples = [m for m in metadata if m['difficulty'] == 'hard']
```

### Preprocessing untuk T5

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoT5-base")

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

---

## Validasi

### Kriteria Kualitas

**CRITICAL (harus pass):**
- ✅ Tidak ada prefix "Konteks:", "Prompt:", "Pertanyaan:", "Jawaban benar:"
- ✅ Target hanya berisi pertanyaan (tanpa jawaban/distraktor)
- ✅ Format JSON valid
- ✅ Encoding UTF-8

**WARNINGS (acceptable):**
- ⚠️ Unclosed code blocks (dari data original)
- ⚠️ Unbalanced bold markers (dari data original)
- ⚠️ Very long targets (>1000 chars)

### Verifikasi

```bash
# Run transformation dengan validasi
python scripts/transform_dataset.py

# Check untuk prefix yang tidak diinginkan
grep -i "Konteks:" dataset_aqg/dataset-task-v2/*.jsonl
grep -i "Jawaban benar:" dataset_aqg/dataset-task-v2/*.jsonl
```

---

## Changelog

### v2 (21 April 2026) - CURRENT
- ✅ Format HuggingFace standard
- ✅ Pemisahan data dan metadata
- ✅ Pembersihan prefix dan suffix
- ✅ Validasi komprehensif
- ✅ Error handling dan edge cases

### v1 (DEPRECATED)
- ❌ Format custom dengan metadata embedded
- ❌ Prefix "Konteks:", "Prompt:", "Pertanyaan:"
- ❌ Target berisi jawaban dan distraktor

---

## Referensi

- **Transformation Script:** `scripts/transform_dataset.py`
- **Training Script:** `src/finetuned/training/task_trainer.py`
- **Training Notebook:** `src/finetuned/notebooks/03_task_specific_training_v2.ipynb`
- **Evaluation Plan:** `docs/plan.md`
- **Phase 2 Report:** `docs/phase2-completion-report.md`
