# Preprocessing & Tokenization Pipeline - Task-Specific AQG Training

## Overview

Dokumen ini menjelaskan tahapan preprocessing dan tokenization yang digunakan dalam notebook `03_task_specific_training_v2.ipynb` untuk fine-tuning IndoNanoT5 pada task Automatic Question Generation (AQG).

---

## Pipeline Preprocessing

### 1. Dataset Loading & Validation

**Apa itu**: Memuat dataset JSONL dan memvalidasi struktur data

**Tujuan**: 
- Memastikan dataset tersedia dan format valid
- Verifikasi field wajib (`input`, `target`, `metadata`)
- Deteksi duplikasi dan statistik dasar

**Cara Kerja**:
```python
# Di notebook Section 3
loader = DatasetLoader()
train_dataset = loader.load_dataset(TASK_DIR, split='train')
validation_results = loader.validate_dataset(train_dataset)
```

**Lokasi Code**:
- Module: `src/finetuned/data/dataset_loader.py`
- Method: `load_dataset()`, `validate_dataset()`
- Notebook: Section 3 - Load Dataset

**Output**:
- HuggingFace Dataset object
- Validation report (total entries, duplicates, avg length)

---

### 2. Tokenizer Loading

**Apa itu**: Load tokenizer yang sesuai dengan model IndoNanoT5

**Tujuan**: 
- Menyediakan tool untuk konversi text → token IDs
- Memastikan compatibility dengan model architecture

**Cara Kerja**:
```python
# Di notebook Section 2
from src.finetuned.utils.model_loader import load_model_with_lora

peft_model, tokenizer = load_model_with_lora(
    model_name='LazarusNLP/IndoNanoT5-base'
)
```

**Lokasi Code**:
- Module: `src/finetuned/utils/model_loader.py`
- Line: `tokenizer = AutoTokenizer.from_pretrained(model_name)`
- Notebook: Section 2 - Load Model with LoRA

**Detail Tokenizer**:
- Type: `AutoTokenizer` (auto-detect T5Tokenizer)
- Vocab size: ~32,000 tokens
- Special tokens: `<pad>`, `</s>` (EOS)

---

### 3. Tokenization (NO Padding)

**Apa itu**: Konversi text menjadi token IDs **tanpa padding**

**Tujuan**:
- Transform text → numerical format untuk model
- Truncate sequences yang > 512 tokens
- Hemat memory dengan NO padding di tahap ini

**Cara Kerja**:
```python
# Di task_trainer.py → preprocess_dataset()
def tokenize_function(examples):
    # Tokenize INPUT (NO PADDING)
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True
        # ❌ TIDAK ada padding=True
    )
    
    # Tokenize TARGET (T5-specific)
    labels = tokenizer(
        text_target=examples["target"],  # ⭐ T5 parameter
        max_length=512,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

**Lokasi Code**:
- Module: `src/finetuned/training/task_trainer.py`
- Method: `preprocess_dataset()` → inner function `tokenize_function()`
- Dipanggil di: `trainer.train()` (Section 6 notebook)

**Key Points**:
- ✅ `text_target` parameter untuk T5 model
- ✅ NO padding (memory efficient)
- ✅ Truncation di 512 tokens
- ✅ Batch processing via `dataset.map(batched=True)`

**Output**:
```python
{
    "input_ids": [234, 567, 1234, ...],      # Variable length
    "attention_mask": [1, 1, 1, ...],        # No padding yet
    "labels": [890, 456, 789, ...]           # Variable length
}
```

---

### 4. Dynamic Padding (DataCollator)

**Apa itu**: Padding dilakukan per-batch saat training, bukan saat preprocessing

**Tujuan**:
- Pad hanya ke max length dalam batch (bukan global max)
- Hemat memory 40-60%
- Automatic label masking dengan `-100`

**Cara Kerja**:
```python
# Di task_trainer.py → train()
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # ⭐ Mask padding di labels
    padding=True,              # Dynamic padding
    pad_to_multiple_of=8       # GPU optimization
)
```

**Lokasi Code**:
- Module: `src/finetuned/training/task_trainer.py`
- Method: `train()` (line ~175-181)
- Digunakan di: `Seq2SeqTrainer` initialization

**Contoh**:
```python
# Batch 1: max_len dalam batch = 128
# → Semua sequences di-pad ke 128

# Batch 2: max_len dalam batch = 256  
# → Semua sequences di-pad ke 256

# ✅ Lebih efisien daripada pad semua ke 512!
```

**Output per Batch**:
```python
{
    "input_ids": [[234, 567, ..., 0, 0]],      # Padded
    "attention_mask": [[1, 1, ..., 0, 0]],     # 1=real, 0=pad
    "labels": [[890, 456, ..., -100, -100]]    # -100=ignored in loss
}
```

---

### 5. Training Execution

**Apa itu**: Eksekusi training loop dengan preprocessed data

**Cara Kerja**:
```python
# Di notebook Section 6
results = trainer.train(
    train_dataset=train_dataset,  # Raw dataset (akan di-preprocess)
    eval_dataset=val_dataset,
    early_stopping=True
)
```

**Lokasi Code**:
- Notebook: Section 6 - Start Training
- Module: `src/finetuned/training/task_trainer.py` → `train()`

**Flow Internal**:
1. Preprocess datasets (tokenization)
2. Setup DataCollator (dynamic padding)
3. Initialize Seq2SeqTrainer
4. Run training loop

---

## Perbandingan: Implementasi vs Spec Design

| Aspek | Spec Design | Implementasi | Status |
|-------|-------------|--------------|--------|
| **Tokenizer** | AutoTokenizer | ✅ `AutoTokenizer.from_pretrained()` | ✅ |
| **T5-Specific** | `text_target` parameter | ✅ Implemented | ✅ |
| **Padding Strategy** | NO padding di tokenization | ✅ NO `padding=True` | ✅ |
| **Dynamic Padding** | DataCollator handles it | ✅ `DataCollatorForSeq2Seq` | ✅ |
| **Label Masking** | `-100` untuk padding | ✅ `label_pad_token_id=-100` | ✅ |
| **Truncation** | `max_length=512` | ✅ Implemented | ✅ |
| **Batch Processing** | `batched=True` | ✅ `dataset.map(batched=True)` | ✅ |

---

## Flow Diagram: Text → Model Input

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Raw JSONL Dataset                                        │
│    {"input": "Apa itu variabel?", "target": "Pertanyaan..."} │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Load Dataset (DatasetLoader)                             │
│    Dataset object: [input, target, metadata]                │
│    Location: Section 3 notebook                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Tokenization (NO Padding)                                │
│    {"input_ids": [234, 567, ...], "labels": [890, ...]}     │
│    Location: task_trainer.py → tokenize_function()          │
│    ⚠️  Variable length, NO padding yet                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Batch Collation (Dynamic Padding)                        │
│    Batch 1: pad to 128 tokens                               │
│    Batch 2: pad to 256 tokens                               │
│    Location: DataCollatorForSeq2Seq                         │
│    ⚠️  Padding per-batch, memory efficient                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Training Batch (Ready for Model)                         │
│    {                                                         │
│      "input_ids": [[234, 567, ..., 0, 0]],                  │
│      "attention_mask": [[1, 1, ..., 0, 0]],                 │
│      "labels": [[890, 456, ..., -100, -100]]                │
│    }                                                         │
│    Location: Seq2SeqTrainer training loop                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### ✅ Best Practices Implemented

1. **AutoTokenizer**: Flexible, auto-detect tokenizer type
2. **T5-Specific**: `text_target` parameter untuk target sequences
3. **Memory Efficient**: NO padding di tokenization, dynamic padding per-batch
4. **Label Masking**: `-100` untuk padding tokens (ignored in loss calculation)
5. **GPU Optimization**: `pad_to_multiple_of=8` untuk Tensor Cores

### 📊 Memory Savings

- **Static Padding** (pad semua ke 512): ~100% memory usage
- **Dynamic Padding** (pad per-batch): ~40-60% memory usage
- **Savings**: 40-60% memory reduction ✅

### 🎯 Kesesuaian dengan Spec

Implementasi preprocessing **100% sesuai** dengan spec design yang telah dibuat:
- Separation of Concerns (Tokenization ≠ Padding) ✅
- Dynamic Padding Strategy ✅
- T5-Specific Parameters ✅
- Label Masking ✅
- GPU Optimization ✅

---

## Referensi

- **HuggingFace T5 Documentation**: https://huggingface.co/docs/transformers/model_doc/t5
- **DataCollatorForSeq2Seq**: https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq
- **Spec Design**: `.kiro/specs/indot5-finetuning/design.md`
- **Implementation**: `src/finetuned/training/task_trainer.py`

====


Baik, saya akan jawab semua pertanyaan Anda secara langsung:

## 1. Apakah kita tidak melakukan preprocessing seperti lowercase, remove special characters, dll?

**TIDAK, kita TIDAK melakukan preprocessing tradisional seperti itu!** 

### Kenapa?

**Alasan 1: Model T5 Sudah Pre-trained dengan Text Asli**
- IndoNanoT5 di-train dengan text dalam format asli (mixed case, dengan punctuation)
- Jika kita lowercase/remove special chars → **MISMATCH** dengan training data model
- Model tidak akan recognize pattern yang sudah dipelajari

**Alasan 2: Informasi Penting Hilang**
```python
# ❌ SALAH - Informasi hilang
"Apa itu variabel?" → "apa itu variabel"  # Kehilangan tanda tanya!
"Python" → "python"  # Kehilangan kapitalisasi (proper noun)

# ✅ BENAR - Pertahankan format asli
"Apa itu variabel?" → tetap "Apa itu variabel?"
```

**Alasan 3: Code Blocks Rusak**
```python
# Dataset kita punya code Python
"print('Hello')" 

# Jika lowercase:
"print('hello')"  # ❌ Semantik berubah!

# Jika remove special chars:
"printhello"  # ❌ Code rusak total!
```

**Alasan 4: Tokenizer Sudah Handle Semuanya**
- T5Tokenizer sudah dilatih untuk handle:
  - Mixed case (uppercase/lowercase)
  - Special characters (!, ?, `, ", dll)
  - Code syntax (parentheses, quotes, dll)
  - Markdown formatting (#, **, dll)

### Preprocessing yang Kita Lakukan HANYA:
1. ✅ **Tokenization** (text → token IDs)
2. ✅ **Truncation** (potong jika > 512 tokens)
3. ✅ **Padding** (dynamic, per-batch)

**Tidak ada**: lowercase, remove punctuation, stemming, lemmatization, dll.

---

## 2. Perbedaan Tahap 2 (Tokenizer Loading) vs Tahap 3 (Tokenization)

### **Tahap 2: Tokenizer Loading** = Menyiapkan ALAT

```python
# Load tokenizer (ALAT untuk tokenize)
tokenizer = AutoTokenizer.from_pretrained('LazarusNLP/IndoNanoT5-base')

# Ini seperti membeli pisau, tapi belum memotong apa-apa
```

**Output**: Tokenizer object (alat)

---

### **Tahap 3: Tokenization** = MENGGUNAKAN ALAT

```python
# Gunakan tokenizer untuk convert text → token IDs
tokens = tokenizer("Apa itu variabel?")

# Ini seperti MENGGUNAKAN pisau untuk memotong sayuran
```

**Output**: Token IDs (hasil tokenisasi)

---

### Analogi Sederhana:

| Tahap | Analogi | Dalam Code |
|-------|---------|------------|
| **Tokenizer Loading** | Beli pisau | `tokenizer = AutoTokenizer.from_pretrained()` |
| **Tokenization** | Potong sayuran dengan pisau | `tokenizer("text")` |

**Kesimpulan**: 
- Tahap 2 = **Persiapan alat** (load tokenizer)
- Tahap 3 = **Pakai alat** (tokenize text)

---

## 3. Perbedaan Pakai Padding vs Tidak Pakai Padding

### **Tanpa Padding** (Variable Length)

```python
# Batch dengan 3 samples
[
  [234, 567, 890],           # Length: 3
  [123, 456],                # Length: 2
  [789, 012, 345, 678, 901]  # Length: 5
]

# ❌ TIDAK BISA di-process oleh GPU!
# GPU butuh tensor dengan shape yang sama
```

**Masalah**: GPU tidak bisa process array dengan panjang berbeda-beda.

---

### **Dengan Padding** (Fixed Length)

```python
# Pad semua ke length 5 (max dalam batch)
[
  [234, 567, 890, 0, 0],     # Length: 5 (padded)
  [123, 456, 0, 0, 0],       # Length: 5 (padded)
  [789, 012, 345, 678, 901]  # Length: 5 (original)
]

# ✅ BISA di-process oleh GPU!
# Semua array punya panjang yang sama
```

**Solusi**: Padding membuat semua sequences punya panjang yang sama.

---

### Kenapa Padding Dibutuhkan?

**GPU/Neural Network butuh tensor dengan shape konsisten**:
```python
# ❌ TIDAK BISA
input_shape = (batch_size, variable_length)  # Error!

# ✅ BISA
input_shape = (batch_size, fixed_length)  # OK!
```

---

## 4. Apa itu Dynamic Padding dan Kenapa Dibutuhkan?

### **Static Padding** (Buruk ❌)

```python
# Pad SEMUA sequences ke max_length global (512)
Batch 1:
  Sample 1: [234, 567, 890, 0, 0, 0, ..., 0]  # 509 padding tokens!
  Sample 2: [123, 456, 0, 0, 0, ..., 0]       # 510 padding tokens!
  
# Waste memory: 99% adalah padding!
```

**Masalah**: 
- Boros memory (banyak padding yang tidak perlu)
- Slow training (process banyak padding tokens)

---

### **Dynamic Padding** (Bagus ✅)

```python
# Pad hanya ke max_length DALAM BATCH

Batch 1 (max_len = 5):
  Sample 1: [234, 567, 890, 0, 0]      # 2 padding
  Sample 2: [123, 456, 0, 0, 0]        # 3 padding
  
Batch 2 (max_len = 10):
  Sample 3: [789, 012, ..., 0, 0, 0]   # 3 padding
  Sample 4: [456, 789, ..., 0]         # 1 padding

# Hemat memory: hanya pad seperlunya per batch!
```

**Keuntungan**:
- ✅ Hemat memory 40-60%
- ✅ Training lebih cepat
- ✅ Flexible (setiap batch beda panjang)

---

### Perbandingan Memory Usage:

| Strategy | Batch 1 | Batch 2 | Total Memory |
|----------|---------|---------|--------------|
| **Static Padding** | 512 × 8 = 4,096 | 512 × 8 = 4,096 | 8,192 tokens |
| **Dynamic Padding** | 128 × 8 = 1,024 | 256 × 8 = 2,048 | 3,072 tokens |
| **Savings** | - | - | **62% less!** |

---

## 5. Kenapa Perlu Label Masking dengan -100?

### **Masalah: Loss Calculation pada Padding**

```python
# Target sequence dengan padding
labels = [890, 456, 789, 0, 0, 0]  # 0 = padding token

# Model prediction
prediction = [890, 456, 789, 123, 456, 789]

# Loss calculation:
loss = CrossEntropy(prediction, labels)

# ❌ MASALAH: Model di-penalize karena predict padding!
# Padding bukan "real" target, tapi model dihukum karena salah predict padding
```

**Masalah**: Model belajar predict padding tokens (yang tidak penting!)

---

### **Solusi: Label Masking dengan -100**

```python
# Target dengan label masking
labels = [890, 456, 789, -100, -100, -100]  # -100 = ignore

# Loss calculation:
loss = CrossEntropy(prediction, labels)
# PyTorch IGNORE token dengan value -100

# ✅ SOLUSI: Model TIDAK di-penalize untuk padding
# Loss hanya dihitung untuk token yang real (890, 456, 789)
```

**Keuntungan**:
- ✅ Model fokus belajar predict real tokens
- ✅ Padding tidak mempengaruhi loss
- ✅ Training lebih akurat

---

### Visualisasi Loss Calculation:

```python
# Tanpa masking (-100)
labels:      [890, 456, 789, 0,    0,    0   ]
prediction:  [890, 456, 789, 123,  456,  789 ]
loss:        [0,   0,   0,   HIGH, HIGH, HIGH]  # ❌ Penalized!
total_loss = (0 + 0 + 0 + HIGH + HIGH + HIGH) / 6

# Dengan masking (-100)
labels:      [890, 456, 789, -100, -100, -100]
prediction:  [890, 456, 789, 123,  456,  789 ]
loss:        [0,   0,   0,   SKIP, SKIP, SKIP]  # ✅ Ignored!
total_loss = (0 + 0 + 0) / 3  # Hanya hitung real tokens
```

---

## Summary Jawaban:

1. **Preprocessing tradisional TIDAK dilakukan** karena:
   - Model sudah pre-trained dengan text asli
   - Informasi penting hilang (case, punctuation, code)
   - Tokenizer sudah handle semuanya

2. **Tokenizer Loading vs Tokenization**:
   - Loading = Persiapan alat (load tokenizer object)
   - Tokenization = Pakai alat (convert text → tokens)

3. **Padding vs No Padding**:
   - No padding = Variable length (GPU tidak bisa process)
   - Padding = Fixed length (GPU bisa process)

4. **Dynamic Padding**:
   - Pad hanya ke max length dalam batch (bukan global)
   - Hemat memory 40-60%
   - Setiap batch bisa beda panjang

5. **Label Masking -100**:
   - Ignore padding tokens dalam loss calculation
   - Model tidak di-penalize untuk predict padding
   - Training lebih akurat dan fokus pada real tokens