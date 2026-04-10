# Panduan Menjalankan Fine-tuning di Google Colab

## Situasi Kamu Sekarang

Berdasarkan screenshot, kamu sudah:
- ✓ Terhubung ke Colab kernel dari VS Code (`Python 3 (ipykernel) - Colab`)
- ✓ Conda env lokal (`nlp_project`) tersedia sebagai alternatif

**Yang perlu dipahami:**
- Colab kernel = Linux server Google, **bukan** Windows lokal kamu
- Library di conda env **tidak otomatis tersedia** di Colab — harus `!pip install` ulang
- Dataset di `D:\2-Project\AQG\` **tidak bisa diakses langsung** dari Colab — harus di-upload

## Daftar Isi
1. [Perbedaan Conda vs Colab](#1-perbedaan-conda-vs-colab)
2. [Cara Upload Dataset ke Colab](#2-cara-upload-dataset-ke-colab)
3. [Persiapan Akun Google](#3-persiapan-akun-google)
4. [Urutan Menjalankan Notebooks](#4-urutan-menjalankan-notebooks)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Perbedaan Conda vs Colab

| Aspek | Conda Env Lokal | Colab Runtime |
|-------|----------------|---------------|
| OS | Windows | Linux (Ubuntu) |
| Python | 3.12.13 | 3.10.x |
| Library | Dari `requirements.txt` | Harus `!pip install` ulang |
| File akses | `D:\2-Project\AQG\` | `/content/` |
| GPU | Tidak ada (CPU only) | T4 15GB (gratis) |
| Persistent | Ya | Hilang saat session berakhir |

**Kesimpulan:** Kamu perlu install ulang library dan upload dataset setiap kali session Colab baru.

---

## 2. Cara Upload Dataset ke Colab

### Opsi A: Upload Langsung (Tanpa Google Drive) — Recommended

Ini cara paling cepat karena kamu sudah connect ke Colab dari VS Code.

**Langkah:**

1. Buka `01_setup_and_validation.ipynb` di VS Code
2. Pastikan kernel = `Python 3 (ipykernel) - Colab`
3. Jalankan cell "Install Dependencies"
4. Jalankan cell "Upload Dataset" — akan muncul tombol upload
5. Pilih file dari `D:\2-Project\AQG\dataset_aqg\output_domain\`:
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`
6. Ulangi untuk `dataset-task-spesifc/`

**Kelemahan:** File hilang saat session Colab berakhir. Tapi model checkpoint bisa disimpan ke Drive.

### Opsi B: Via Google Drive (Persistent)

Jika ingin dataset tersimpan permanen:

1. Upload folder `dataset_aqg` ke Google Drive
2. Di notebook, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Akses dataset dari `/content/drive/MyDrive/AQG/dataset_aqg/`

### Opsi C: Upload via Zip (Untuk Source Code)

Source code `src/finetuned/` perlu di-upload agar bisa di-import:

**Di Windows (PowerShell):**
```powershell
cd D:\2-Project\AQG
python -m zipfile -c src_finetuned.zip src/finetuned
```

Lalu upload `src_finetuned.zip` via cell upload di notebook.

---

## 3. Persiapan Akun Google

### Buat Akun Google (jika belum punya)
1. Buka [accounts.google.com/signup](https://accounts.google.com/signup)
2. Isi nama, username, password
3. Verifikasi nomor HP
4. Selesai — akun otomatis dapat akses ke Google Drive dan Colab

### Cek Google Drive Storage
- Free tier: **15 GB** (cukup untuk project ini)
- Model checkpoint ~1 GB per stage, total ~3-4 GB
- Jika kurang, hapus file lama atau upgrade ke Google One

---

## 2. Upload Project ke Google Drive

### Opsi A: Upload via Browser (Recommended)

1. Buka [drive.google.com](https://drive.google.com)
2. Klik **"+ New"** → **"Folder upload"**
3. Pilih folder `AQG` dari komputer kamu
4. Tunggu upload selesai (bisa 5-15 menit tergantung ukuran)

Struktur yang harus ada di Drive:
```
MyDrive/
└── AQG/
    ├── src/
    │   └── finetuned/
    │       ├── data/
    │       ├── model/
    │       ├── training/
    │       ├── evaluation/
    │       ├── utils/
    │       └── notebooks/
    ├── dataset_aqg/
    │   ├── output_domain/
    │   │   ├── train.jsonl
    │   │   ├── validation.jsonl
    │   │   └── test.jsonl
    │   └── dataset-task-spesifc/
    │       ├── train.jsonl
    │       ├── validation.jsonl
    │       └── test.jsonl
    └── checkpoints/  (akan dibuat otomatis)
```

### Opsi B: Upload via Google Drive Desktop App
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Sign in dengan akun Google
3. Copy folder `AQG` ke folder Google Drive di komputer
4. Tunggu sync selesai

---

## 3. Setup Google Colab

### Buka Google Colab
1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Sign in dengan akun Google yang sama

### Enable GPU (WAJIB)
1. Klik menu **Runtime** (atas)
2. Pilih **"Change runtime type"**
3. Di **"Hardware accelerator"**, pilih **T4 GPU**
4. Klik **Save**

> ⚠️ Tanpa GPU, training akan sangat lambat (10-20x lebih lama)

### Buka Notebook dari Drive
1. Di Colab, klik **File** → **Open notebook**
2. Pilih tab **Google Drive**
3. Navigate ke `AQG/src/finetuned/notebooks/`
4. Pilih notebook yang ingin dijalankan

---

## 4. Urutan Menjalankan Notebooks

Jalankan notebook **secara berurutan**:

```
01_setup_and_validation.ipynb  →  02_domain_adaptation.ipynb  →  03_task_specific_training.ipynb
```

---

### Notebook 01: Setup and Validation

**Tujuan:** Verifikasi semua komponen berjalan dengan benar sebelum training.

**Langkah:**

1. Buka `01_setup_and_validation.ipynb` di Colab
2. **Ubah path project** di cell ini:
   ```python
   PROJECT_ROOT = '/content/drive/MyDrive/AQG'  # Sesuaikan jika berbeda
   ```
3. Klik **Runtime** → **Run all** (atau Ctrl+F9)
4. Saat muncul popup "Mount Google Drive", klik **Connect to Google Drive**
5. Tunggu semua cell selesai (~5-10 menit)

**Yang diverifikasi:**
- GPU T4 tersedia
- Dataset domain (340 entries) dan task-specific (1,262 entries) ter-load
- Tokenizer berjalan normal
- Model IndoNanoT5 + LoRA ter-load (~248M params, ~0.5% trainable)
- Baseline BLEU-4 < 0.15 (expected)

**Output yang diharapkan:**
```
✓ GPU: Tesla T4 (15.78 GB)
✓ Domain Train: 272 entries
✓ Task Train: 883 entries
✓ Tokenizer loaded
✓ Model loaded with LoRA
Baseline BLEU-4: 0.0823
```

---

### Notebook 02: Domain Adaptation

**Tujuan:** Stage 1 training — adaptasi model ke domain Python.

**Estimasi waktu:** 2-3 jam

**Langkah:**

1. Buka `02_domain_adaptation.ipynb` di Colab
2. **Ubah path project** di cell setup:
   ```python
   PROJECT_ROOT = '/content/drive/MyDrive/AQG'  # Sesuaikan
   ```
3. Klik **Runtime** → **Run all**
4. Tunggu training selesai

> ⚠️ **Jangan tutup browser** selama training. Colab akan disconnect jika tab tidak aktif.
> 
> Tips: Buka tab Colab di foreground, sesekali scroll atau klik untuk mencegah timeout.

**Monitoring training:**

Saat training berjalan, kamu akan melihat progress bar seperti:
```
Epoch 1/6: 100%|████████| 34/34 [08:23<00:00]
{'loss': 2.8432, 'eval_loss': 2.6123, 'epoch': 1.0}

Epoch 2/6: 100%|████████| 34/34 [08:15<00:00]
{'loss': 2.1234, 'eval_loss': 2.1876, 'epoch': 2.0}
...
```

**Output yang diharapkan:**
```
Final training loss: ~1.5
Model saved: /content/drive/MyDrive/AQG/checkpoints/domain/indot5-python-domain
```

**Jika session disconnect:**
- Checkpoints sudah tersimpan di Drive setiap epoch
- Buka notebook lagi, jalankan ulang dari awal
- HuggingFace Trainer akan otomatis resume dari checkpoint terakhir

---

### Notebook 03: Task-Specific Training

**Tujuan:** Stage 2 training — fine-tune untuk AQG task.

**Estimasi waktu:** 1-2 jam

**Prerequisites:** Notebook 02 harus sudah selesai dan model `indot5-python-domain` tersimpan di Drive.

**Langkah:**

1. Buka `03_task_specific_training.ipynb` di Colab
2. **Ubah path project** di cell setup:
   ```python
   PROJECT_ROOT = '/content/drive/MyDrive/AQG'  # Sesuaikan
   STAGE1_MODEL_PATH = '/content/drive/MyDrive/AQG/checkpoints/domain/indot5-python-domain'
   ```
3. Klik **Runtime** → **Run all**
4. Tunggu training selesai

**Monitoring training:**

Setiap epoch akan menampilkan BLEU-4 score:
```
Epoch 1/3: {'eval_bleu_4': 0.1823, 'eval_rouge_l': 0.2341}
Epoch 2/3: {'eval_bleu_4': 0.3012, 'eval_rouge_l': 0.3876}
Epoch 3/3: {'eval_bleu_4': 0.3845, 'eval_rouge_l': 0.4123}
```

**Output yang diharapkan:**
```
BLEU-4:       0.0823 → 0.3845
ROUGE-L:      0.1234 → 0.4123
BERTScore F1: 0.4521 → 0.7834

✓ SUCCESS: BLEU-4 target achieved (>= 0.35)
✓ Fine-tuning pipeline complete!
```

**Files yang dihasilkan:**
```
checkpoints/aqg/
├── indot5-python-aqg/     ← Final model
└── training_curves.png    ← Training plots

evaluation_results/
├── evaluation_report.json ← Metrics report
└── sample_outputs.json    ← 20 sample predictions
```

---

## 5. Troubleshooting

### Error: "GPU not available"
**Solusi:**
1. Runtime → Change runtime type → T4 GPU → Save
2. Runtime → Disconnect and delete runtime
3. Runtime → Run all (ulang dari awal)

---

### Error: "FileNotFoundError: Dataset file not found"
**Solusi:**
1. Pastikan dataset sudah di-upload ke Drive
2. Cek path di cell setup:
   ```python
   PROJECT_ROOT = '/content/drive/MyDrive/AQG'  # Harus sesuai
   ```
3. Verifikasi file ada:
   ```python
   import os
   print(os.listdir('dataset_aqg/output_domain/'))
   # Harus ada: ['train.jsonl', 'validation.jsonl', 'test.jsonl']
   ```

---

### Error: "CUDA out of memory"
**Solusi:**
1. Kurangi batch size di training args:
   ```python
   training_args = trainer.get_training_args(
       per_device_train_batch_size=4,   # Dari 8 ke 4
       gradient_accumulation_steps=8,   # Dari 4 ke 8 (effective batch tetap 32)
   )
   ```
2. Enable gradient checkpointing:
   ```python
   peft_model.gradient_checkpointing_enable()
   ```
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Error: "ModuleNotFoundError: No module named 'src'"
**Solusi:**
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/AQG')
```

---

### Session Timeout / Disconnect
**Colab free tier** memiliki batas waktu ~12 jam per session.

**Pencegahan:**
- Buka tab Colab di foreground
- Sesekali scroll atau klik di notebook
- Gunakan ekstensi browser seperti "Colab Alive" (Chrome)

**Recovery:**
1. Buka notebook lagi
2. Run semua cell setup (install, mount drive, import)
3. Training akan resume dari checkpoint terakhir secara otomatis

---

### Training Sangat Lambat
**Cek GPU:**
```python
import torch
print(torch.cuda.get_device_name(0))  # Harus T4, bukan CPU
```

Jika CPU, ubah runtime ke T4 GPU (lihat bagian Setup).

---

## Tips Tambahan

### Cegah Timeout dengan JavaScript
Paste di browser console (F12 → Console):
```javascript
function KeepAlive() {
  document.querySelector('#top-toolbar').click();
  setTimeout(KeepAlive, 60000);
}
KeepAlive();
```

### Monitor GPU Usage
```python
!nvidia-smi
```

### Cek Disk Usage di Drive
```python
!df -h /content/drive/MyDrive/
```

### Download Model Setelah Training
```python
import shutil
from google.colab import files

# Zip model
shutil.make_archive('/content/indot5-python-aqg', 'zip',
                    '/content/drive/MyDrive/AQG/checkpoints/aqg/indot5-python-aqg')

# Download
files.download('/content/indot5-python-aqg.zip')
```

---

## Estimasi Waktu Total

| Tahap | Waktu |
|-------|-------|
| Upload project ke Drive | 10-20 menit |
| Notebook 01 (validation) | 5-10 menit |
| Notebook 02 (domain adaptation) | 2-3 jam |
| Notebook 03 (task-specific) | 1-2 jam |
| **Total** | **~4-6 jam** |

---

## Hasil Akhir

Setelah semua notebook selesai, kamu akan memiliki:

- `indot5-python-domain` — Model setelah domain adaptation
- `indot5-python-aqg` — Model final untuk AQG
- `evaluation_report.json` — Metrics lengkap
- `sample_outputs.json` — 20 contoh output soal yang dihasilkan
