# Analisis Masalah Padding dan Label Masking

## Ringkasan Masalah

Training IndoT5 gagal total dengan:
- Training loss: **0.0000** (seharusnya 0.5-2.0)
- Eval loss: **NaN** (numerical instability)
- Semua metrics **lebih buruk** dari baseline
- Model output **gibberish** (copy-paste input)

## Root Cause: Label Masking Issue

### Hipotesis
Semua labels di-mask dengan `-100`, menyebabkan model tidak menerima learning signal.

### Bukti
1. Loss = 0.0 → tidak ada valid labels untuk dihitung
2. Eval loss = NaN → division by zero
3. Model tidak belajar → output gibberish

## Analisis DataCollator

### Dokumentasi Resmi HuggingFace

Dari [DataCollatorForSeq2Seq Documentation](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq):

```python
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, 
    as well as the labels.
    
    Parameters:
        tokenizer: The tokenizer used for encoding the data
        model: The model being trained (optional)
        padding: Strategy to pad sequences (default: True)
        max_length: Maximum length (optional)
        pad_to_multiple_of: Pad to multiple of value (optional)
        label_pad_token_id: ID for padding labels (default: -100)
        return_tensors: Type of tensor (default: "pt")
    """
```

### Perilaku yang Benar

**DataCollator seharusnya**:
1. Pad input sequences ke panjang maksimum dalam batch
2. Pad label sequences ke panjang maksimum dalam batch
3. **HANYA** mask padding tokens dengan `-100`
4. **TIDAK** mask valid label tokens

**Contoh**:
```python
# Sebelum collator
labels = [10, 20, 30, 40, 50]  # 5 tokens

# Setelah collator (batch max length = 8)
labels = [10, 20, 30, 40, 50, -100, -100, -100]
#         ↑  valid tokens  ↑   ↑  padding  ↑
```

### Masalah dalam Implementasi Lama

```python
# MASALAH: max_length parameter
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    label_pad_token_id=-100,
    padding=True,
    max_length=self.max_length,  # ❌ BERMASALAH
    pad_to_multiple_of=8
)
```

**Mengapa bermasalah?**

Dari dokumentasi:
> `max_length` (int, optional) — Maximum length of the returned list and optionally padding length (see above).

Ketika `padding=True` dan `max_length` diset:
- Collator akan pad SEMUA sequences ke `max_length` (512)
- Jika original labels pendek (misal 50 tokens), akan ada 462 padding tokens
- Ini menyebabkan **mayoritas labels di-mask**

**Warning yang muncul**:
```
UserWarning: `max_length` is ignored when `padding`=`True` 
and there is no truncation strategy.
```

Ini sebenarnya **PERINGATAN PENTING** bahwa parameter tidak digunakan dengan benar!

## Solusi yang Benar

### 1. Hapus max_length dari DataCollator

```python
# SOLUSI: Dynamic padding tanpa max_length
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    label_pad_token_id=-100,  # Mask padding dengan -100
    padding=True,  # Dynamic padding
    pad_to_multiple_of=8  # Untuk efisiensi GPU
    # TIDAK menggunakan max_length
)
```

**Keuntungan**:
- Pad hanya ke panjang maksimum dalam batch (bukan 512)
- Lebih sedikit padding tokens
- Lebih banyak valid labels untuk training
- Lebih efisien memory dan compute

### 2. Verifikasi Tokenization

Pastikan tokenization menghasilkan labels yang valid:

```python
def tokenize_function(examples):
    # Tokenize inputs
    model_inputs = self.tokenizer(
        examples["input"],
        max_length=self.max_length,
        truncation=True
        # TIDAK ada padding di sini
    )
    
    # Tokenize targets dengan text_target (PENTING untuk T5!)
    labels = self.tokenizer(
        text_target=examples["target"],  # ✓ Correct untuk T5
        max_length=self.max_length,
        truncation=True
        # TIDAK ada padding di sini
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs
```

**Penting**: 
- Gunakan `text_target` untuk T5 models
- JANGAN pad di tokenization
- Biarkan DataCollator handle padding

## Perbandingan: Sebelum vs Sesudah

### Sebelum (SALAH)

```python
# Tokenization: NO padding ✓
labels = [10, 20, 30, 40, 50]  # 5 tokens

# DataCollator: Pad ke max_length=512 ❌
labels = [10, 20, 30, 40, 50, -100, -100, ..., -100]
#         ↑  5 valid  ↑  ↑    507 masked    ↑

# Result: 99% labels di-mask → zero loss
```

### Sesudah (BENAR)

```python
# Tokenization: NO padding ✓
labels = [10, 20, 30, 40, 50]  # 5 tokens

# DataCollator: Pad ke batch max (misal 60) ✓
labels = [10, 20, 30, 40, 50, -100, -100, ..., -100]
#         ↑  5 valid  ↑  ↑    55 masked     ↑

# Result: 8% labels di-mask → valid loss
```

## Debugging Steps

### 1. Jalankan Debug Notebook

```bash
# Di Colab
src/finetuned/notebooks/debug_label_masking.ipynb
```

Notebook akan:
- ✓ Verifikasi tokenization menghasilkan labels valid
- ✓ Verifikasi DataCollator tidak mask semua labels
- ✓ Verifikasi forward pass menghasilkan loss valid
- ✓ Verifikasi generation menghasilkan output reasonable

### 2. Periksa Output

**Jika semua labels = 0**:
```python
# Masalah: tokenization
# Fix: Periksa text_target parameter
```

**Jika semua labels = -100**:
```python
# Masalah: DataCollator
# Fix: Hapus max_length parameter
```

**Jika loss = 0.0**:
```python
# Masalah: Tidak ada valid labels
# Fix: Ikuti langkah di atas
```

## Expected Results Setelah Fix

### Training Metrics
- Training loss: 2.0 → 0.5-1.0 (decreasing)
- Eval loss: 1.5-2.5 (valid number, not NaN)

### Test Metrics
- BLEU-4: 0.15-0.25 (+50-100% vs baseline)
- ROUGE-L: 0.20-0.30 (+50-100% vs baseline)
- Distinct-1: 0.40-0.50 (not decreasing)

### Sample Outputs
- Generate pertanyaan yang coherent
- TIDAK copy-paste input
- Follow format AQG yang benar

## Langkah Selanjutnya

1. **Fix DataCollator** (DONE ✓)
   - Hapus `max_length` parameter
   - Verifikasi dengan debug notebook

2. **Domain Adaptation** (TODO)
   - Train dengan 340 samples dari `output_domain/`
   - 2 epochs, learning rate 5e-5
   - Save adapted model

3. **Task-Specific Training** (TODO)
   - Load domain-adapted model
   - Increase LoRA r dari 8 ke 16
   - Lower learning rate dari 1e-4 ke 5e-5
   - More epochs dari 3 ke 5

4. **Dataset Augmentation** (TODO)
   - Current: 876 samples
   - Target: 2,000-5,000 samples
   - Generate more synthetic questions

## Referensi

- [DataCollatorForSeq2Seq Docs](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq)
- [T5 Tokenizer Docs](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)
- [Training Report](../training-report/indot5-report2.md)
- [Quick Fix Guide](../training-report/QUICK_FIX_GUIDE.md)
