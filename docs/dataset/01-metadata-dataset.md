# Metadata Dataset AQG

**Tanggal:** 21 April 2026  
**Status:** Available for Analysis  
**Format:** JSONL (JSON Lines)

---

## Overview

File metadata menyimpan informasi tambahan tentang setiap sample dalam dataset yang tidak digunakan saat training, tetapi sangat berguna untuk evaluasi mendalam, error analysis, dan reporting. Metadata dipisahkan dari data utama untuk menjaga format HuggingFace standard yang clean.

---

## Struktur File

```
dataset_aqg/dataset-task-v2/
├── train_metadata.jsonl      # Metadata untuk 876 training samples
├── validation_metadata.jsonl # Metadata untuk 175 validation samples
└── test_metadata.jsonl        # Metadata untuk 211 test samples
```

**Korespondensi 1:1:**
- Baris ke-N di `train.jsonl` → Baris ke-N di `train_metadata.jsonl`
- Baris ke-N di `test.jsonl` → Baris ke-N di `test_metadata.jsonl`

---

## Format Metadata

### Schema

```json
{
  "difficulty": "easy|medium|hard",
  "question_type": "Multiple Choice|Code Completion|Conceptual|True/False|Error Identification|Code Output Prediction",
  "concept": "<konsep_pembelajaran_utama>",
  "misconception_tags": ["<tag1>", "<tag2>", ...],
  "source_file": "<path_ke_file_materi_asli>",
  "section": "<judul_section_dalam_materi>",
  "source": "synthetic",
  "validated": true
}
```

### Field Descriptions

| Field                | Type    | Description                        | Example                     |
| ----------------------| ---------| ------------------------------------| -----------------------------|
| `difficulty`         | string  | Tingkat kesulitan pertanyaan       | `"hard"`                    |
| `question_type`      | string  | Jenis pertanyaan                   | `"Code Completion"`         |
| `concept`            | string  | Konsep pembelajaran utama          | `"Fundamental Matriks"`     |
| `misconception_tags` | array   | Tag misconception yang ditargetkan | `["`miskonsepsi_numpy`"]`   |
| `source_file`        | string  | Path ke file materi asli           | `"dataset_aqg/materi/..."`  |
| `section`            | string  | Section dalam materi               | `"### Perbandingan Memori"` |
| `source`             | string  | Sumber data (selalu "synthetic")   | `"synthetic"`               |
| `validated`          | boolean | Status validasi (selalu true)      | `true`                      |

### Contoh Lengkap

```json
{
  "difficulty": "hard",
  "question_type": "Code Completion",
  "concept": "Fundamental Matriks",
  "misconception_tags": [
    "`miskonsepsi_numpy_formula`",
    "`lupa_operasi_perkalian`",
    "`hafalan_nilai_output`"
  ],
  "source_file": "dataset_aqg/materi/07-matriks/01-fundamental-matriks.md",
  "section": "### Perbandingan Penggunaan Memori",
  "source": "synthetic",
  "validated": true
}
```

---

## Tujuan & Kegunaan

### 1. Error Analysis & Debugging

**Tujuan:** Identifikasi pola kesalahan model

**Cara Kerja:**
```python
import json

# Load metadata dan predictions
with open('test_metadata.jsonl') as f:
    metadata = [json.loads(line) for line in f]

# Analisis: Di difficulty mana model sering salah?
errors_by_difficulty = {}
for meta, pred, ref in zip(metadata, predictions, references):
    if pred != ref:  # Jika prediksi salah
        diff = meta['difficulty']
        errors_by_difficulty[diff] = errors_by_difficulty.get(diff, 0) + 1

print(errors_by_difficulty)
# Output: {'easy': 5, 'medium': 15, 'hard': 30}
# Insight: Model struggle dengan pertanyaan hard
```

### 2. Stratified Evaluation

**Tujuan:** Evaluasi performa model per kategori

**Cara Kerja:**
```python
from collections import defaultdict

# Group samples by difficulty
samples_by_diff = defaultdict(list)
for i, meta in enumerate(metadata):
    samples_by_diff[meta['difficulty']].append(i)

# Evaluate per difficulty
results = {}
for diff, indices in samples_by_diff.items():
    preds = [predictions[i] for i in indices]
    refs = [references[i] for i in indices]
    results[diff] = {
        'count': len(indices),
        'bleu': calculate_bleu(preds, refs),
        'rouge': calculate_rouge(preds, refs)
    }

# Output:
# easy:   BLEU=0.85, ROUGE=0.88 (90 samples)
# medium: BLEU=0.72, ROUGE=0.75 (105 samples)
# hard:   BLEU=0.58, ROUGE=0.62 (16 samples)
```

### 3. Misconception Analysis

**Tujuan:** Evaluasi kemampuan model menangani misconception

**Cara Kerja:**
```python
# Filter samples dengan misconception tertentu
numpy_samples = [
    i for i, meta in enumerate(metadata)
    if any('numpy' in tag.lower() for tag in meta['misconception_tags'])
]

# Evaluasi khusus untuk misconception ini
numpy_preds = [predictions[i] for i in numpy_samples]
numpy_refs = [references[i] for i in numpy_samples]
numpy_score = calculate_metrics(numpy_preds, numpy_refs)

print(f"Model performance on NumPy misconceptions: {numpy_score}")
```

### 4. Detailed Reporting

**Tujuan:** Membuat laporan evaluasi komprehensif

**Cara Kerja:**
```python
# Generate comprehensive report
report = {
    'overall_metrics': calculate_metrics(predictions, references),
    'by_difficulty': {},
    'by_question_type': {},
    'by_concept': {}
}

# Breakdown by difficulty
for diff in ['easy', 'medium', 'hard']:
    indices = [i for i, m in enumerate(metadata) if m['difficulty'] == diff]
    report['by_difficulty'][diff] = {
        'count': len(indices),
        'metrics': calculate_metrics(
            [predictions[i] for i in indices],
            [references[i] for i in indices]
        )
    }

# Breakdown by question type
for qtype in set(m['question_type'] for m in metadata):
    indices = [i for i, m in enumerate(metadata) if m['question_type'] == qtype]
    report['by_question_type'][qtype] = {
        'count': len(indices),
        'metrics': calculate_metrics(
            [predictions[i] for i in indices],
            [references[i] for i in indices]
        )
    }

# Save report
with open('evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

### 5. Dataset Filtering & Experimentation

**Tujuan:** Membuat subset dataset untuk eksperimen

**Cara Kerja:**
```python
# Eksperimen: Training hanya dengan pertanyaan easy & medium
easy_medium_indices = [
    i for i, meta in enumerate(metadata)
    if meta['difficulty'] in ['easy', 'medium']
]

# Filter dataset
filtered_dataset = dataset.select(easy_medium_indices)

# Train dengan subset ini
trainer.train(filtered_dataset)
```

### 6. Traceability

**Tujuan:** Trace kembali ke source material untuk investigasi

**Cara Kerja:**
```python
# Jika menemukan error menarik
error_index = 42
error_meta = metadata[error_index]

print(f"Error found in:")
print(f"  Source: {error_meta['source_file']}")
print(f"  Section: {error_meta['section']}")
print(f"  Concept: {error_meta['concept']}")
print(f"  Difficulty: {error_meta['difficulty']}")

# Bisa buka file asli untuk investigasi lebih lanjut
with open(error_meta['source_file']) as f:
    content = f.read()
    # Analisis konteks asli
```

---

## Kapan Menggunakan Metadata?

### ✅ GUNAKAN Metadata Untuk:

1. **Post-Training Evaluation**
   - Setelah model selesai training
   - Ingin analisis mendalam tentang performa model
   - Membuat laporan evaluasi komprehensif

2. **Error Analysis**
   - Model performa tidak sesuai ekspektasi
   - Ingin tahu di kategori mana model lemah
   - Debugging untuk improvement

3. **Research & Publication**
   - Membuat paper atau laporan penelitian
   - Butuh breakdown metrics yang detail
   - Perlu visualisasi performa per kategori

4. **Model Comparison**
   - Membandingkan beberapa model
   - Ingin tahu model mana yang lebih baik di kategori tertentu

5. **Dataset Analysis**
   - Memahami distribusi dataset
   - Identifikasi bias atau imbalance
   - Planning untuk data augmentation

### ❌ TIDAK Perlu Metadata Untuk:

1. **Training Process**
   - Model training tidak butuh metadata
   - Hanya butuh `input` dan `target`

2. **Basic Evaluation**
   - Jika hanya butuh overall BLEU/ROUGE score
   - Quick check apakah model belajar

3. **Inference/Prediction**
   - Saat menggunakan model untuk generate pertanyaan
   - Production deployment

---

## Workflow Penggunaan

### Phase 1: Training (TIDAK Pakai Metadata)

```python
from datasets import load_dataset

# Load dataset (tanpa metadata)
dataset = load_dataset(
    'json',
    data_files={
        'train': 'dataset_aqg/dataset-task-v2/train.jsonl',
        'validation': 'dataset_aqg/dataset-task-v2/validation.jsonl',
        'test': 'dataset_aqg/dataset-task-v2/test.jsonl'
    }
)

# Training (hanya pakai input & target)
trainer.train(dataset['train'])
```

### Phase 2: Basic Evaluation (TIDAK Pakai Metadata)

```python
# Generate predictions
predictions = model.generate(test_dataset)

# Calculate overall metrics
bleu_score = calculate_bleu(predictions, references)
rouge_score = calculate_rouge(predictions, references)

print(f"Overall BLEU: {bleu_score}")
print(f"Overall ROUGE: {rouge_score}")
```

### Phase 3: Deep Analysis (PAKAI Metadata)

```python
import json

# Load metadata
with open('dataset_aqg/dataset-task-v2/test_metadata.jsonl') as f:
    metadata = [json.loads(line) for line in f]

# Stratified evaluation
for difficulty in ['easy', 'medium', 'hard']:
    indices = [i for i, m in enumerate(metadata) if m['difficulty'] == difficulty]
    diff_preds = [predictions[i] for i in indices]
    diff_refs = [references[i] for i in indices]
    
    bleu = calculate_bleu(diff_preds, diff_refs)
    print(f"{difficulty}: BLEU={bleu:.3f} ({len(indices)} samples)")

# Error analysis
errors = []
for i, (pred, ref, meta) in enumerate(zip(predictions, references, metadata)):
    if pred != ref:
        errors.append({
            'index': i,
            'difficulty': meta['difficulty'],
            'question_type': meta['question_type'],
            'concept': meta['concept'],
            'prediction': pred,
            'reference': ref
        })

# Analyze error patterns
error_by_diff = {}
for err in errors:
    diff = err['difficulty']
    error_by_diff[diff] = error_by_diff.get(diff, 0) + 1

print("\nError distribution:")
for diff, count in error_by_diff.items():
    print(f"  {diff}: {count} errors")
```

---

## Rekomendasi

### Untuk Sekarang (Training Phase)

**TIDAK PERLU** update notebook 03 untuk metadata karena:
- Focus pada training dengan dataset format baru
- Cukup evaluasi basic (overall BLEU/ROUGE)
- Metadata untuk analisis lanjutan nanti

### Untuk Nanti (Analysis Phase)

**UPDATE notebook** jika:
- Model sudah selesai training
- Ingin analisis mendalam tentang performa
- Perlu breakdown metrics untuk laporan/paper
- Ada masalah yang perlu di-debug

### Workflow yang Disarankan

```
1. Training dengan dataset v2 (tanpa metadata)
   ↓
2. Basic evaluation (overall metrics)
   ↓
3. Jika metrics bagus → Selesai ✓
   Jika metrics kurang → Lanjut ke step 4
   ↓
4. Deep analysis dengan metadata
   ↓
5. Identifikasi masalah spesifik
   ↓
6. Improvement (data augmentation, model tuning, etc.)
```

---

## Loading Metadata

### Python

```python
import json

# Load metadata
def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Usage
train_meta = load_metadata('dataset_aqg/dataset-task-v2/train_metadata.jsonl')
test_meta = load_metadata('dataset_aqg/dataset-task-v2/test_metadata.jsonl')

# Access
print(train_meta[0]['difficulty'])  # 'hard'
print(train_meta[0]['concept'])     # 'Fundamental Matriks'
```

### Pandas

```python
import pandas as pd

# Load as DataFrame
train_meta_df = pd.read_json(
    'dataset_aqg/dataset-task-v2/train_metadata.jsonl',
    lines=True
)

# Analysis
print(train_meta_df['difficulty'].value_counts())
print(train_meta_df['question_type'].value_counts())
print(train_meta_df['concept'].value_counts())

# Filter
hard_questions = train_meta_df[train_meta_df['difficulty'] == 'hard']
code_completion = train_meta_df[train_meta_df['question_type'] == 'Code Completion']
```

---

## Statistik Metadata

### Distribusi Difficulty

| Difficulty | Train | Validation | Test | Total |
|------------|-------|------------|------|-------|
| Easy       | ~263  | ~53        | ~63  | ~379  |
| Medium     | ~394  | ~79        | ~95  | ~568  |
| Hard       | ~219  | ~43        | ~53  | ~315  |

### Distribusi Question Type

- Multiple Choice: ~35%
- Code Completion: ~30%
- Conceptual: ~20%
- True/False: ~8%
- Error Identification: ~4%
- Code Output Prediction: ~3%

### Top Concepts

1. Fundamental Python (variabel, tipe data, operator)
2. Control Flow (if-else, loop)
3. Fungsi dan Subprogram
4. Struktur Data (list, dictionary, set, tuple)
5. Matriks dan NumPy

---

## Referensi

- **Dataset Schema:** `docs/dataset/01-skema-dataset.md`
- **Transformation Script:** `scripts/transform_dataset.py`
- **Training Notebook:** `src/finetuned/notebooks/03_task_specific_training_v2.ipynb`
       |            |      |       |
```