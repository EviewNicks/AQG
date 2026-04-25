# IndoT5 Training Report #3: Dataset Format v2

**Tanggal:** 21 April 2026  
**Model:** Wikidepia/IndoT5-base (297.8M parameters)  
**Dataset:** HuggingFace Standard Format (v2)  
**Training Time:** 0.19 hours (~11 minutes)

---

## Executive Summary

Training dengan dataset format baru (v2) menunjukkan **KEGAGALAN KRITIS**. Model mengalami masalah yang sama dengan training sebelumnya: training loss = 0.0000, validation loss = NaN, dan metrics tidak menunjukkan pembelajaran yang bermakna.

**Status:** ❌ FAILED - Model tidak belajar

---

## Perubahan dari Report #2

### Dataset Format

**Report #2 (Format Lama):**
```json
{
  "input": "Konteks: <text>\n\nPrompt: Buatlah pertanyaan...",
  "target": "Pertanyaan: <question>? Jawaban benar: <answer>. Distraktor: ...",
  "difficulty": "...",
  ...
}
```

**Report #3 (Format Baru - HuggingFace Standard):**
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

### Perubahan Kunci

1. ✅ Removed "Konteks:" prefix dari input
2. ✅ Removed prompt instruction dari input  
3. ✅ Removed "Pertanyaan:" prefix dari target
4. ✅ Removed "Jawaban benar:" dan distractors dari target
5. ✅ Metadata dipisahkan ke file terpisah
6. ✅ Format clean dan standard

**Catatan:** Dataset di Google Drive diganti dengan format baru, path notebook tidak berubah (`dataset-task-spesifc/` tetap digunakan tetapi berisi data format v2).

---

## Training Configuration

### Model Setup

| Parameter | Value |
|-----------|-------|
| Base Model | Wikidepia/IndoT5-base |
| Total Parameters | 297,811,200 |
| Trainable Parameters | 884,736 (0.30%) |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target Modules | ['q', 'v'] |

### Dataset

### Dataset

| Split | Samples | Avg Input Length | Avg Target Length |
|-------|---------|------------------|-------------------|
| Train | 876 | 707.47 chars | 140.21 chars |
| Validation | 175 | - | - |
| Test | 211 | - | - |

**Dataset Issues:**
- ⚠️ 649 duplicate entries detected (74% of training data)
- ⚠️ No metadata in loaded dataset (expected - metadata separated)

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 32 |
| Learning Rate | 1e-4 |
| Warmup Steps | 50 |
| FP16 | True |
| Max Length | 512 |

---

## Training Results

### Loss Progression

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | No log | NaN |
| 2 | 0.000000 | NaN |
| 3 | 0.000000 | NaN |

**Critical Issues:**
- ❌ Training loss = 0.0000 (model tidak menerima learning signal)
- ❌ Validation loss = NaN (numerical instability)
- ❌ No improvement across epochs

---

## Evaluation Metrics

### Baseline vs Fine-tuned Comparison

**Baseline (Pre-training):**
- Evaluated on 10 validation samples
- Model: IndoT5-base (no fine-tuning)

**Fine-tuned (Post-training):**
- Evaluated on 10 validation samples  
- Model: IndoT5-base + LoRA (after 3 epochs)

### BLEU Scores

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| BLEU | 0.0336 | 0.0303 | -9.72% |
| BLEU-1 | 0.1578 | 0.1330 | -15.69% |
| BLEU-2 | 0.0504 | 0.0388 | -22.96% |
| BLEU-3 | 0.0198 | 0.0175 | -11.44% |
| BLEU-4 | 0.0081 | 0.0093 | **+15.50%** |

**Analysis:**
- BLEU-4 menunjukkan improvement kecil (+15.5%)
- BLEU-1, BLEU-2, BLEU-3 menurun signifikan
- Overall BLEU menurun (-9.72%)
- **Kesimpulan:** Improvement BLEU-4 tidak bermakna karena nilai absolut sangat rendah (0.0093)

### ROUGE Scores

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| ROUGE-1 | 0.1715 | 0.1638 | -4.49% |
| ROUGE-2 | 0.0573 | 0.0546 | -4.79% |
| ROUGE-L | 0.1379 | 0.1335 | -3.22% |

**Analysis:**
- Semua ROUGE metrics menurun
- Penurunan konsisten di semua level (1-gram, 2-gram, longest)
- Model tidak belajar generate output yang lebih baik

### Diversity Metrics

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Distinct-1 | 0.5610 | 0.2068 | -63.14% |
| Distinct-2 | 0.8646 | 0.5051 | -41.58% |

**Analysis:**
- Diversity menurun drastis (>40%)
- Model generate output yang lebih repetitive
- Indikasi model tidak belajar variasi output

### BERTScore

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| BERTScore F1 | 0.0000 | 0.6383 | +∞ |

**Analysis:**
- BERTScore baseline = 0 (tidak dihitung)
- BERTScore fine-tuned = 0.6383 (moderate)
- Tidak dapat dibandingkan karena baseline tidak ada

---

## Performance Summary

### Target vs Actual

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| BLEU-4 | ≥ 0.35 | 0.0093 | ❌ FAILED (97% below target) |
| ROUGE-L | ≥ 0.40 | 0.1335 | ❌ FAILED (67% below target) |
| Training Loss | Decreasing | 0.0000 (flat) | ❌ FAILED |
| Validation Loss | Decreasing | NaN | ❌ FAILED |

### Overall Assessment

**Training Status:** ❌ CRITICAL FAILURE

**Key Findings:**
1. Model tidak belajar (training loss = 0.0000)
2. Validation loss = NaN menunjukkan numerical instability
3. Metrics tidak menunjukkan improvement bermakna
4. Diversity menurun drastis (model lebih repetitive)
5. BLEU-4 jauh di bawah target (0.0093 vs 0.35)

---

## Comparison with Previous Reports

### Report #1 (IndoNanoT5)
- Model: IndoNanoT5-base (248M params)
- Status: FAILED (model terlalu kecil)
- BLEU-4: Not achieved

### Report #2 (IndoT5 - Format Lama)
- Model: IndoT5-base (297M params)
- Dataset: Format custom dengan metadata embedded
- Status: FAILED (training loss = 0.0000, eval loss = NaN)
- BLEU-4: 0.0081 baseline

### Report #3 (IndoT5 - Format Baru) - CURRENT
- Model: IndoT5-base (297M params)
- Dataset: HuggingFace standard format (clean)
- Status: FAILED (training loss = 0.0000, eval loss = NaN)
- BLEU-4: 0.0093 (slight improvement dari baseline 0.0081)

**Kesimpulan Perbandingan:**
- Format dataset baru TIDAK menyelesaikan masalah
- Masalah fundamental masih ada (loss = 0.0000, eval = NaN)
- Improvement BLEU-4 (+15.5%) tidak bermakna karena nilai absolut sangat rendah

---

## Technical Observations

### Training Behavior

1. **Loss = 0.0000:**
   - Model tidak menerima learning signal
   - Kemungkinan: semua labels di-mask dengan -100
   - Atau: gradient tidak mengalir ke trainable parameters

2. **Validation Loss = NaN:**
   - Numerical instability
   - Kemungkinan: division by zero atau log(0)
   - Atau: invalid loss computation

3. **Metrics Degradation:**
   - BLEU-1, BLEU-2, BLEU-3 menurun
   - ROUGE semua menurun
   - Diversity menurun drastis
   - Indikasi: model tidak belajar, malah memburuk

4. **Duplicate Data:**
   - 649 duplicates dari 876 samples (74%)
   - Dapat menyebabkan overfitting
   - Dapat mengurangi efektivitas training

### Dataset Format Impact

**Expected:** Format baru yang clean seharusnya improve training
**Actual:** Tidak ada perbaikan, masalah yang sama persist

**Possible Reasons:**
1. Masalah bukan di format dataset
2. Masalah di preprocessing/tokenization
3. Masalah di DataCollator (label masking)
4. Masalah di training loop
5. Masalah di model configuration

---

## Metrics Interpretation

### BLEU-4 = 0.0093

**Interpretation:**
- Sangat rendah (target: 0.35)
- Hanya 2.7% dari target
- Model hampir tidak generate n-gram yang match dengan reference
- Equivalent dengan random generation

### ROUGE-L = 0.1335

**Interpretation:**
- Rendah (target: 0.40)
- Hanya 33% dari target
- Longest common subsequence sangat pendek
- Model tidak capture struktur kalimat dengan baik

### Distinct-1 = 0.2068 (turun dari 0.5610)

**Interpretation:**
- Diversity menurun 63%
- Model generate output yang sangat repetitive
- Indikasi model "stuck" di pattern tertentu
- Tidak belajar variasi output

---

## Data Analysis

### Training Data Characteristics

| Characteristic | Value |
|----------------|-------|
| Total Samples | 876 |
| Unique Samples | 227 (26%) |
| Duplicate Samples | 649 (74%) |
| Avg Input Length | 707.47 chars |
| Avg Target Length | 140.21 chars |
| Input/Target Ratio | 5.04:1 |

**Observations:**
- High duplication rate (74%)
- Long input, short target (5:1 ratio)
- May cause training instability

### Sample Data

**Input (truncated):**
```
### Perbandingan Penggunaan Memori

```python
import numpy
import sys

var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
...
```

**Target (truncated):**
```
Sesuai catatan modul yang menggunakan list Python untuk matriks, 
lengkapi kode berikut untuk menghitung ukuran memori list: 
import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; 
ukuran_memori = ________________.?
```

**Format:** Clean, no prefixes, no suffixes ✓

---

## Resource Usage

| Resource | Value |
|----------|-------|
| GPU | NVIDIA T4 (Colab) |
| GPU Memory Allocated | 1.19 GB |
| Training Time | 0.19 hours (~11 minutes) |
| Samples/Second (Eval) | 1.72 |
| Steps/Second (Eval) | 0.22 |

**Observations:**
- Training sangat cepat (11 menit untuk 3 epochs)
- Kemungkinan: model tidak benar-benar training
- GPU utilization rendah

---

## Conclusion

### Summary

Training dengan dataset format HuggingFace standard (v2) **GAGAL** mencapai target. Model menunjukkan masalah yang sama dengan training sebelumnya:

1. ❌ Training loss = 0.0000 (tidak belajar)
2. ❌ Validation loss = NaN (numerical instability)
3. ❌ BLEU-4 = 0.0093 (97% di bawah target 0.35)
4. ❌ ROUGE-L = 0.1335 (67% di bawah target 0.40)
5. ❌ Diversity menurun drastis (-63%)

### Key Findings

1. **Format dataset bukan root cause** - masalah persist meskipun format sudah clean
2. **Model tidak menerima learning signal** - loss = 0.0000 konsisten
3. **Numerical instability** - validation loss = NaN
4. **High data duplication** - 74% duplicate samples
5. **Metrics degradation** - model memburuk, bukan membaik

### Status

**CRITICAL FAILURE** - Model tidak dapat digunakan untuk production. Masalah fundamental dalam training pipeline perlu diidentifikasi dan diperbaiki sebelum melanjutkan.

---

## Appendix

### Training Configuration File

```python
{
    'model': 'Wikidepia/IndoT5-base',
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'target_modules': ['q', 'v'],
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 3,
    'warmup_steps': 50,
    'max_length': 512,
    'fp16': True
}
```

### File Locations

- Model Checkpoint: `/content/drive/MyDrive/dataset_aqg/checkpoints/aqg/`
- Evaluation Report: `/content/drive/MyDrive/dataset_aqg/evaluation_results/evaluation_report.json`
- Sample Outputs: `/content/drive/MyDrive/dataset_aqg/evaluation_results/sample_outputs.json`
- Training Results: `/content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_results.json`

### References

- Previous Report: `docs/training-report/indot5-report2.md`
- Dataset Schema: `docs/dataset/01-skema-dataset.md`
- Metadata Documentation: `docs/dataset/01-metadata-dataset.md`
- Transformation Script: `scripts/transform_dataset.py`
