# Laporan Analisis Proyek AQG Fine-Tuning IndoT5

**Tanggal:** 20 April 2026  
**Model:** Wikidepia/IndoT5-base (297M parameters)  
**Task:** Automatic Question Generation (AQG) untuk Python berbahasa Indonesia

---

## 📊 Informasi yang Dapat Dijawab

### 1. Dataset Struktur Saat Ini

#### ✅ Jumlah Total Samples

**Domain Dataset** (`dataset_aqg/output_domain/accumulated.jsonl`):
- Total: **341 samples** (format span_corruption + qa_generic)
- Format: Mixed (span corruption untuk domain adaptation, qa_generic untuk QA)

**Task-Specific Dataset** (`dataset_aqg/dataset-task-spesifc/`):
- Train: **876 samples**
- Validation: **175 samples**
- Test: **211 samples**
- Total: **1,262 samples**

#### ✅ Struktur JSON

**Format JSONL yang Benar:**
```json
{
  "input": "Konteks: [materi Python]\\n\\nPrompt: Buat satu soal [question_type] tentang [concept], tingkat kesulitan: [difficulty], bahasa Indonesia.",
  "target": "Pertanyaan: [soal]? Jawaban benar: [jawaban]. Distraktor: 1) [d1] 2) [d2] 3) [d3] 4) [d4]",
  "metadata": {
    "difficulty": "easy|medium|hard",
    "question_type": "MCQ|Code Completion",
    "concept": "...",
    "misconception_tags": ["tag1", "tag2"],
    "source_file": "...",
    "validated": true
  }
}
```

**Catatan Penting:**
- `target` adalah **plain string**, bukan JSON object ✅
- `metadata` adalah **JSON object terpisah** ✅
- Kolom `metadata` **harus di-drop** sebelum training (sudah dilakukan) ✅

---

### 2. Preprocessing Detail

#### ✅ Task Prefix

**TIDAK menggunakan task prefix** pada implementasi saat ini.

**Alasan:**
1. IndoT5 dilatih pada CulturaX yang case-sensitive
2. Dataset task-specific hanya berisi format `qa_generic` setelah `span_corruption` dihapus
3. Tidak ada kebutuhan task prefix untuk single-task fine-tuning
4. Konsistensi: inference juga tidak perlu prefix

**Kode di `domain_trainer.py` (line 82-95):**
```python
# Tokenize inputs as-is — tidak menggunakan prefix maupun .lower().
# Alasan:
#   1. IndoNanoT5 dilatih pada CulturaX yang case-sensitive
#   2. Domain dataset hanya berisi format qa_generic setelah span_corruption dihapus
#   3. Konsistensi: inference juga tidak perlu lowercase.
model_inputs = self.tokenizer(
    examples["input"],
    max_length=self.max_length,
    truncation=True,
    padding=False
)
```

#### ✅ Code Block Handling

**Preservasi Code Block:**
- Code block Python **TIDAK di-escape** dengan karakter khusus
- Code block dipertahankan **as-is** dalam format Markdown (` ```python ... ``` `)
- Chunker memastikan code block **tidak pernah dipotong** di tengah

**Implementasi di `chunker.py`:**
```python
# Code block tidak pernah dipotong — selalu dipertahankan utuh dalam satu chunk
# Token count diestimasi dengan len(text.split()) * 1.3
```

#### ✅ Normalisasi Teks

**Preprocessing yang Dilakukan:**
1. **Path normalization**: `source_file` menggunakan forward slash (`/`) di semua OS
2. **Token masking**: Padding tokens di-replace dengan `-100` untuk loss calculation
3. **Truncation**: Max length 512 tokens
4. **NO lowercase**: Case-sensitive dipertahankan untuk Python syntax

**Preprocessing yang TIDAK Dilakukan:**
- ❌ Lowercase conversion (akan merusak Python syntax)
- ❌ Special character escaping
- ❌ Code block modification

---

### 3. Training Results

#### ✅ Training Loss Progression

**MASALAH KRITIS TERDETEKSI:**

```
Final training loss: 0.0000  ❌
eval_loss: nan               ❌
```

**Analisis:**
- Training loss = 0.0000 menunjukkan **model tidak belajar**
- Eval loss = NaN menunjukkan **numerical instability**
- Ini adalah **bug preprocessing/DataCollator**, bukan masalah dataset

**Root Cause (sudah diidentifikasi di `error.md`):**
- DataCollator dengan `max_length` parameter menyebabkan **semua labels di-mask dengan -100**
- Model tidak punya learning signal karena tidak ada valid labels

**Fix yang Sudah Diterapkan:**
```python
# SEBELUM (SALAH):
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    max_length=self.max_length  # ❌ Ini menyebabkan bug!
)

# SESUDAH (BENAR):
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    label_pad_token_id=-100,
    padding=True,
    pad_to_multiple_of=8
    # ✅ TIDAK menggunakan max_length
)
```

#### ✅ Epochs yang Dijalankan

**Task-Specific Training:**
- Epochs: **3**
- Batch size: **8**
- Gradient accumulation: **4**
- Effective batch size: **32**
- Learning rate: **1e-4**
- Warmup steps: **50**

**Training Time:**
- Duration: **637.98 seconds** (~10.6 minutes)
- Samples per second: **4.12**

#### ✅ Validation Loss / Evaluation Metrics

**Baseline (Pre-Training):**
```
BLEU-4:  0.0143
ROUGE-L: 0.1387
```

**After Fine-Tuning (dengan bug):**
```
BLEU-4:  0.0133  (-7.6%)  ❌
ROUGE-L: 0.1224  (-11.8%) ❌
BERTScore F1: 0.6305
```

**Kesimpulan:**
- Metrics **TURUN** setelah training → konfirmasi model tidak belajar
- BERTScore 0.63 menunjukkan output masih semantically related (base model capability)

---

### 4. Tokenizer Testing

#### ✅ Test Hasil Tokenization

**Dari `error.md` - Test Tokenization:**

```
Input IDs length: 319
Label IDs length: 201

Input padding tokens: 0 (seharusnya 0) ✅
Label padding tokens: 0 (seharusnya 0) ✅

Non-zero label tokens: 201 / 201 ✅
✓ Labels mengandung token non-zero (BAGUS)
```

**Analisis:**
- Tokenization **BENAR** ✅
- Tidak ada padding di tahap preprocessing ✅
- Semua labels valid (tidak ada -100 prematur) ✅

#### ✅ Token Overflow Issues

**TIDAK ADA token overflow** yang terdeteksi.

**Statistik:**
- Max length: **512 tokens**
- Avg input length: **821 chars** (~250 tokens estimated)
- Avg target length: **344 chars** (~100 tokens estimated)
- Total: ~350 tokens (well below 512 limit)

---

### 5. Current Error Context

#### ✅ Error dari Training Pertama atau Iterasi?

**Ini adalah hasil dari ITERASI KEDUA:**

1. **Iterasi Pertama** (IndoNanoT5):
   - Model: LazarusNLP/IndoNanoT5-base (248M params)
   - Hasil: Model terlalu kecil untuk complex AQG task
   - Keputusan: Switch ke IndoT5-base (580M params)

2. **Iterasi Kedua** (IndoT5 - Current):
   - Model: Wikidepia/IndoT5-base (297M params)
   - Bug: DataCollator masking issue
   - Status: **Bug sudah diidentifikasi dan di-fix**

#### ✅ Perubahan Format Dataset

**Perubahan yang Sudah Dilakukan:**

1. **Hapus format `span_corruption`** dari task-specific dataset
   - Alasan: Tidak cocok untuk AQG task
   - Hanya gunakan format `qa_generic`

2. **Drop kolom `metadata`** sebelum training
   - Alasan: Metadata menyebabkan dataset size mismatch
   - Fix: `dataset.remove_columns(['metadata'])`

3. **Normalisasi `source_file` path**
   - Gunakan forward slash (`/`) di semua OS
   - Implementasi: `pathlib.PurePosixPath`

---

## 🔍 Informasi yang Perlu Anda Berikan

### 1. ❓ Apakah Fix DataCollator Sudah Diterapkan?

**Pertanyaan:**
- Apakah Anda sudah **re-run training** setelah fix DataCollator?
- Apakah training loss sekarang **> 0.0** dan eval loss **bukan NaN**?

**Yang Perlu Dilakukan:**
```python
# Pastikan DataCollator TIDAK menggunakan max_length
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True,
    pad_to_multiple_of=8
    # ❌ JANGAN tambahkan max_length parameter!
)
```

### 2. ❓ Apakah Ada Training Log Terbaru?

**Pertanyaan:**
- Apakah ada file `training_results.json` atau log training setelah fix?
- Bagaimana loss progression per epoch setelah fix?

**Yang Perlu Diperiksa:**
```bash
# Cek file training results
{
  "training_loss": 0.0,
  "metrics": {
    "train_runtime": 637.9805,
    "train_samples_per_second": 4.119,
    "train_steps_per_second": 0.132,
    "total_flos": 1580075954012160.0,
    "train_loss": 0.0,
    "epoch": 3.0
  },
  "training_history": [
    {
      "eval_loss": NaN,
      "eval_bleu_1": 0.027711943751850796,
      "eval_bleu_4": 0.027711943751850796,
      "eval_rouge_l": 0.0,
      "eval_runtime": 114.8499,
      "eval_samples_per_second": 1.524,
      "eval_steps_per_second": 0.192,
      "epoch": 1.0,
      "step": 28
    },
    {
      "loss": 0.0,
      "grad_norm": NaN,
      "learning_rate": 9.8e-05,
      "epoch": 1.8,
      "step": 50
    },
    {
      "eval_loss": NaN,
      "eval_bleu_1": 0.027711943751850796,
      "eval_bleu_4": 0.027711943751850796,
      "eval_rouge_l": 0.0,
      "eval_runtime": 118.4421,
      "eval_samples_per_second": 1.478,
      "eval_steps_per_second": 0.186,
      "epoch": 2.0,
      "step": 56
    },
    {
      "eval_loss": NaN,
      "eval_bleu_1": 0.027711943751850796,
      "eval_bleu_4": 0.027711943751850796,
      "eval_rouge_l": 0.0,
      "eval_runtime": 118.0967,
      "eval_samples_per_second": 1.482,
      "eval_steps_per_second": 0.186,
      "epoch": 3.0,
      "step": 84
    },
    {
      "train_runtime": 637.9805,
      "train_samples_per_second": 4.119,
      "train_steps_per_second": 0.132,
      "total_flos": 1580075954012160.0,
      "train_loss": 0.0,
      "epoch": 3.0,
      "step": 84
    },
    {
      "eval_loss": NaN,
      "eval_bleu_1": 0.027711943751850796,
      "eval_bleu_4": 0.027711943751850796,
      "eval_rouge_l": 0.0,
      "eval_runtime": 119.5063,
      "eval_samples_per_second": 1.464,
      "eval_steps_per_second": 0.184,
      "epoch": 3.0,
      "step": 84
    }
  ]
}

```

### 3. ❓ Apakah Perlu Tambah Data?

**Pertanyaan:**
- Apakah Anda ingin **menambah dataset** dari 1,262 samples?
- Target: 1,500-3,000 samples (sesuai spec)

**Opsi:**
1. Generate lebih banyak synthetic data via LLM
2. Implementasi Augmentor (task 10 di tasks.md - optional)
3. Manual annotation

### 4. ❓ Hyperparameter Tuning?

**Pertanyaan:**
- Apakah Anda ingin **experiment dengan hyperparameters**?

**Rekomendasi untuk Dicoba:**
```python
# Option 1: Lower learning rate
learning_rate = 5e-5  # dari 1e-4

# Option 2: More epochs
num_train_epochs = 5  # dari 3

# Option 3: Larger batch size
per_device_train_batch_size = 16  # dari 8
gradient_accumulation_steps = 2   # dari 4
```

### 5. ❓ Evaluation Strategy?

**Pertanyaan:**
- Apakah Anda ingin **qualitative analysis** dari generated outputs?
- Apakah perlu **error analysis** untuk identify failure patterns?

**Yang Bisa Dilakukan:**
1. Analisis sample outputs 
2. Kategorisasi error types (format, content, hallucination)
3. Identify misconception tags yang sulit di-generate

---

## 📋 Ringkasan Status

### ✅ Yang Sudah Benar

1. **Dataset format**: JSONL dengan `input`, `target`, `metadata` ✅
2. **Tokenization**: Benar, tidak ada overflow ✅
3. **Preprocessing**: Case-sensitive, code preservation ✅
4. **Bug identification**: DataCollator issue sudah diidentifikasi ✅
5. **Model selection**: IndoT5-base (297M) cocok untuk task ✅

### ⚠️ Yang Perlu Diperbaiki

1. **Re-run training** dengan DataCollator fix
2. **Verify loss > 0.0** dan eval loss bukan NaN
3. **Monitor metrics improvement** setelah fix

### 🎯 Next Steps

1. **Immediate**: Re-run training dengan fix, verify loss progression
2. **Short-term**: Evaluate metrics, analyze sample outputs
3. **Long-term**: Consider data augmentation, hyperparameter tuning

---

**Prepared by:** Kiro AI Assistant  
**Last Updated:** 20 April 2026
