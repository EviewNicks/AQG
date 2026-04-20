# 1 Load Model dan Tokenizer 

/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:103: UserWarning: 
Error while fetching `HF_TOKEN` secret value from your vault: 'Requesting secret HF_TOKEN timed out. Secrets can only be fetched when running from the Colab UI.'.
You are not authenticated with the Hugging Face Hub in this notebook.
If the error persists, please let us know by opening an issue on GitHub (https://github.com/huggingface/huggingface_hub/issues/new).
  warnings.warn(
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

✓ Base model loaded
✓ LoRA applied: r=8, alpha=16, target=['q', 'v']
  Trainable: 884,736 (0.30%)
  Total:     297,811,200
✓ Model device: cuda:0
  GPU allocated: 1.19 GB
Tokenizer pad_token_id: 0
Tokenizer eos_token_id: 1
Model device: cuda:0

# 2 Load Sample Data

✓ Dataset already exists
✓ Loaded 876 entries from /content/dataset_aqg/dataset-task-spesifc/train.jsonl

Loaded 876 samples

Sample input length: 973 chars
Sample target length: 422 chars

Input preview : Konteks: ### Perbandingan Penggunaan Memori

```python
import numpy
import sys

var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Ukuran keseluruhan elemen list dalam bytes =", sys.getsizeof(var_list) * len(var_list))
print("Ukuran keseluruhan elemen NumPy dalam bytes =", var_array.size * var_array.itemsize)

"""
Output:
Ukuran keseluruhan elemen list dalam bytes = 240
Ukuran keseluruhan elemen NumPy dalam bytes = 72
"""
```
Dengan matriks yang sama, NumPy hanya menggunakan **72 bytes** dibanding list Python yang menggunakan **240 bytes** — inilah alasan banyak programmer memilih NumPy untuk memproses matriks. > **Catatan:** Seluruh materi pada modul ini akan menggunakan list Python untuk mengimplementasikan matriks, agar kita memahami fundamental matriks tanpa melibatkan library apa pun.

Prompt: Buat satu soal Code Completion tentang Fundamental Matriks, tingkat kesulitan: hard, bahasa Indonesia....

Target preview : Pertanyaan: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.? Jawaban benar: `sys.getsizeof(var_list) * len(var_list)`. Distraktor: 1) `var_array.size * var_array.itemsize` 2) `sys.getsizeof(var_list)` 3) `sys.getsizeof(var_list) + len(var_list)` 4) `240`...

# 3 Test Tokenization

=== HASIL TOKENIZATION ===

Input IDs length: 319
Input IDs sample (20 pertama): [2777, 5561, 39, 2892, 18209, 18209, 926, 16135, 30, 12489, 24532, 11, 10347, 10347, 10347, 258, 22502, 7676, 120, 11]

Label IDs length: 201
Label IDs sample (20 pertama): [926, 369, 23, 30, 39, 15506, 1489, 10059, 10, 138, 19133, 211, 22502, 21, 12942, 8, 2046, 43, 1620, 614]

Input padding tokens: 0 (seharusnya 0)
Label padding tokens: 0 (seharusnya 0)

Non-zero label tokens: 201 / 201

✓ Labels mengandung token non-zero (BAGUS)

# 4 Test DataColator Behaviour 

=== SEBELUM DATACOLLATOR ===
Sample 1 - Input length: 319
Sample 1 - Label length: 201
Sample 2 - Input length: 50
Sample 2 - Label length: 30

=== SETELAH DATACOLLATOR ===
Batch input_ids shape: torch.Size([2, 320])
Batch labels shape: torch.Size([2, 208])

First sample labels (30 pertama): tensor([  926,   369,    23,    30,    39, 15506,  1489, 10059,    10,   138,
        19133,   211, 22502,    21, 12942,     8,  2046,    43,  1620,   614,
           21,  6983,  1547,  6273, 19133,    39,  7676,   120,  7809,    32])

=== ANALISIS MASKING ===
Masked tokens (-100): 7 (3.4%)
Non-masked tokens: 201 (96.6%)
Total tokens: 208

✓ DataCollator bekerja dengan benar!
   201 valid labels untuk training.

# 5  Test Forward Pass 

=== HASIL FORWARD PASS ===

Loss: 9.9250
Logits shape: torch.Size([2, 208, 32128])

✓ Loss valid: 9.9250
   Model dapat menerima learning signal.

# 6 Test Generation

=== TEST GENERATION ===

Input (200 char): Konteks: ### Perbandingan Penggunaan Memori

```python
import numpy
import sys

var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Ukuran k...

Reference (200 char): Pertanyaan: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran...

Prediction (200 char): Perbandingan Penggunaan Memori Konteks: ### Perbandingan Penggunaan Memori Konteks: ### Perbandingan Penggunaan Memori Konteks: ### Perbandingan Penggunaan Memori Konteks: ### Perbandingan Penggunaan ...

Prediction length: 414 chars

⚠️ PERINGATAN: Prediction tidak dimulai dengan 'Pertanyaan'
   Format output mungkin tidak sesuai.
```

# Hasil Evaluasi Report

## Evaluasi Hasil Debug - Ringkasan

Berdasarkan hasil debug, saya dapat memberikan **KABAR BAIK** dan analisis lengkap:

### ✅ MASALAH UTAMA SUDAH FIXED!

**DataCollator Fix BERHASIL**:
- ✓ Tokenization: **201/201 labels valid** (100%)
- ✓ DataCollator: **201/208 non-masked** (96.6% valid, 3.4% padding)
- ✓ Forward Pass: **Loss = 9.9250** (VALID, bukan 0.0 atau NaN!)
- ✓ Model menerima learning signal dengan benar

**Perbandingan**:
```
SEBELUM FIX:
- Loss: 0.0000 ❌
- Eval loss: NaN ❌
- Labels: >90% di-mask ❌

SESUDAH FIX:
- Loss: 9.9250 ✓
- Labels: 96.6% valid ✓
- DataCollator bekerja benar ✓
```

### ⚠️ Prediction Warning - BUKAN Masalah Kritis

**Output yang Dihasilkan**:
```
Prediction: "Perbandingan Penggunaan Memori Konteks: ### Perbandingan..."
Expected: "Pertanyaan: Sesuai catatan modul..."
```

**Analisis**:
1. **BUKAN masalah dataset** - Format dataset sudah benar
2. **BUKAN masalah preprocessing** - Tokenization 100% valid
3. **BUKAN masalah DataCollator** - Masking sudah benar

**Root Cause**: **Model belum di-train!**
- Ini adalah **base model** (pre-trained IndoT5)
- Model belum pernah lihat task AQG
- Model belum belajar format output yang diinginkan
- **NORMAL** untuk model yang belum di-fine-tune

### 📊 Kesimpulan

**Masalah Training Loss=0.0 SUDAH TERATASI**:
1. ✅ DataCollator fix berhasil (hapus `max_length`)
2. ✅ Labels tidak di-mask semua (96.6% valid)
3. ✅ Loss valid (9.9250, bukan 0.0)
4. ✅ Model siap untuk training

**Prediction Warning adalah EXPECTED**:
- Model base belum di-train untuk AQG
- Setelah training, output akan sesuai format
- Ini BUKAN indikasi masalah teknis

