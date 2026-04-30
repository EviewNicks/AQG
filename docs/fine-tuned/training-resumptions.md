# Training Resumptions: Melanjutkan Training yang Terputus

## Overview

Dokumentasi ini menjelaskan cara melanjutkan training IndoNanoT5 adapter yang terputus karena mati lampu, GPU habis, atau Colab disconnect. Sistem kami **otomatis menyimpan checkpoint setiap epoch**, sehingga Anda dapat melanjutkan dari epoch terakhir tanpa perlu training ulang dari awal.

**Benefit:**
- ⏱️ Hemat waktu: 50-75% lebih cepat
- 💰 Hemat biaya GPU: Tidak perlu training ulang
- 🧠 Tidak stress: Progress tersimpan otomatis

Cara Pakia 

```
# ✅ Auto-resume dari checkpoint terakhir
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint=True  # ✅ SEKARANG WORKS!
)

# ✅ Start fresh (no resume)
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint=False
)

# ✅ Resume dari checkpoint spesifik
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint="/path/to/checkpoint-1200"
)


```

---

## Konsep Dasar

### Apa itu Checkpoint?

Checkpoint adalah snapshot lengkap dari training state yang disimpan setiap epoch:

```
checkpoint-300/
├── pytorch_model.bin          # Model weights
├── optimizer.pt               # Optimizer state
├── scheduler.pt               # Learning rate scheduler
├── trainer_state.json         # Training state (epoch, step, metrics)
├── training_args.bin          # Training configuration
└── adapter_config.json        # Adapter configuration
```

### Bagaimana Sistem Kami Bekerja?

```
Training Epoch 1 → Save checkpoint-300
Training Epoch 2 → Save checkpoint-600
Training Epoch 3 → Save checkpoint-900
Training Epoch 4 → Save checkpoint-1200
    ⚠️ MATI LAMPU!
    
Restart Colab → Load checkpoint-1200 → Resume Epoch 5
Training Epoch 5 → Save checkpoint-1500
Training Epoch 6 → Save checkpoint-1800
... dst sampai Epoch 8
```

---

## Prosedur Resume Training

### Metode 1: Auto-Resume (PALING MUDAH) ✅

Ini adalah cara yang **paling direkomendasikan**. Trainer otomatis mendeteksi checkpoint terakhir.

#### Step 1: Setup Environment (5 menit)

```python
# Cell 1: Install dependencies
!pip install -q adapters transformers datasets accelerate evaluate rouge_score bert_score
print('✓ Dependencies installed')

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print('✓ Google Drive mounted')

# Cell 3: Extract source code
import os, sys, zipfile, shutil

DRIVE_ROOT = '/content/drive/MyDrive/dataset_aqg'
sys.path.insert(0, '/content')

if not os.path.exists('/content/src'):
    shutil.copy(f'{DRIVE_ROOT}/src_finetuned.zip', '/content/')
    with zipfile.ZipFile('/content/src_finetuned.zip', 'r') as z:
        z.extractall('/content/')
    print('✓ src extracted')
else:
    print('✓ src already exists')
```

#### Step 2: Load Model & Dataset (5 menit)

```python
# Cell 4: Load model with adapter
from src.finetuned.utils.adapter_loader import load_model_with_adapter, print_adapter_info

model, tokenizer = load_model_with_adapter(
    model_name='LazarusNLP/IndoNanoT5-base',
    adapter_name='mcq_generation',
    adapter_config='pfeiffer',
    reduction_factor=12,
    device='cuda'
)

trainable, total = print_adapter_info(model, tokenizer)

# Cell 5: Load dataset
from src.finetuned.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
TASK_DIR = '/content/dataset_aqg/dataset-task-spesifc/'

train_dataset = loader.load_dataset(TASK_DIR, split='train')
val_dataset = loader.load_dataset(TASK_DIR, split='validation')

print(f'✓ Model & dataset loaded')
print(f'  Train: {len(train_dataset)} samples')
print(f'  Val: {len(val_dataset)} samples')
```

#### Step 3: 🔑 Resume Training (3-4 jam)

```python
# Cell 6: Setup trainer & resume
from src.finetuned.training.adapter_trainer import AdapterTrainer
from src.finetuned.evaluation.metrics_calculator import MetricsCalculator

CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'

# Initialize trainer
trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    metrics_calculator=MetricsCalculator(),
    output_dir=CHECKPOINT_DIR,
    max_length=512
)

# Setup training configuration (SAMA seperti training pertama)
training_args = trainer.setup_training(
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01
)

# 🔑 RESUME TRAINING - Cukup tambahkan parameter ini!
print("🔄 Resuming training from last checkpoint...")
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    resume_from_checkpoint=True  # ✅ Otomatis detect checkpoint terakhir!
)

print(f'✓ Training completed!')
print(f'  Final loss: {results["training_loss"]:.4f}')
```

#### Step 4: Save & Evaluate (5 menit)

```python
# Cell 7: Save adapter
adapter_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 12,
        "num_train_epochs": 8,
        "learning_rate": 1e-4,
    }
)

print(f'✓ Adapter saved: {adapter_path}')

# Plot training curves
trainer.plot_training_curves(
    save_path=f'{CHECKPOINT_DIR}/training_curves.png'
)

print('✓ Training curves saved')
```

---

### Metode 2: Manual Resume (Jika Ingin Kontrol Lebih)

Gunakan metode ini jika ingin memilih checkpoint tertentu atau check dulu.

```python
# Check available checkpoints
import os

CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'
checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('checkpoint-')]

print(f"Available checkpoints: {sorted(checkpoints)}")
# Output: ['checkpoint-300', 'checkpoint-600', 'checkpoint-900', 'checkpoint-1200']

# Resume dari checkpoint tertentu
latest_checkpoint = sorted(checkpoints)[-1]  # Ambil yang terakhir
print(f"Resuming from: {latest_checkpoint}")

results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    resume_from_checkpoint=f"{CHECKPOINT_DIR}/{latest_checkpoint}"
)
```

---

## ⚠️ Hal Penting yang Perlu Diperhatikan

### 1. CHECKPOINT_DIR HARUS SAMA

**❌ SALAH:**
```python
# Training pertama
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints/adapter_v3'

# Resume (BERBEDA!)
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints/adapter_v3_new'
# → Trainer tidak akan menemukan checkpoint!
```

**✅ BENAR:**
```python
# Training pertama
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints/adapter_v3'

# Resume (SAMA!)
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints/adapter_v3'
# → Trainer akan menemukan checkpoint!
```

### 2. Checkpoint Harus Ada di Folder

Jika checkpoint tidak terdeteksi, copy dari Drive:

```python
# Cek dulu
import os
CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'

if not os.path.exists(CHECKPOINT_DIR):
    print("⚠️ Checkpoint directory not found!")
    print("Creating and copying from Drive...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    !cp -r /content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3/* {CHECKPOINT_DIR}/
    print("✓ Checkpoints copied")
```

### 3. Training Config HARUS Konsisten

**⚠️ Jangan ubah parameter ini saat resume:**
- `num_train_epochs` (tetap 8)
- `per_device_train_batch_size` (tetap 4)
- `learning_rate` (tetap 1e-4)
- `gradient_accumulation_steps` (tetap 2)

**✅ Boleh ubah:**
- `logging_steps` (bisa diubah)
- `save_total_limit` (bisa diubah)
- `warmup_steps` (bisa diubah)

---

## Troubleshooting

### Problem 1: "No checkpoint found"

**Penyebab:** Checkpoint directory kosong atau path salah

**Solusi:**
```python
# Cek path
import os
CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'
print(f"Checkpoint dir exists: {os.path.exists(CHECKPOINT_DIR)}")
print(f"Contents: {os.listdir(CHECKPOINT_DIR)}")

# Jika kosong, copy dari Drive
if not os.listdir(CHECKPOINT_DIR):
    print("Copying from Drive...")
    !cp -r /content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3/* {CHECKPOINT_DIR}/
```

### Problem 2: "Checkpoint corrupted"

**Penyebab:** File checkpoint tidak lengkap atau rusak

**Solusi:**
```python
# Hapus checkpoint yang rusak
import shutil
CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'
shutil.rmtree(CHECKPOINT_DIR)

# Copy ulang dari Drive
!cp -r /content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3 {CHECKPOINT_DIR}

# Resume training
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    resume_from_checkpoint=True
)
```

### Problem 3: "Out of Memory (OOM)"

**Penyebab:** GPU memory tidak cukup saat resume

**Solusi:**
```python
# Reduce batch size
training_args = trainer.setup_training(
    num_train_epochs=8,
    per_device_train_batch_size=2,  # Reduce dari 4 ke 2
    gradient_accumulation_steps=4,  # Increase dari 2 ke 4
    # ... config lainnya
)

# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Resume training
results = trainer.train(...)
```

---

## Expected Output

Ketika resume training berjalan dengan baik, Anda akan melihat:

```
📂 Found 4 checkpoint(s):
   - checkpoint-300
   - checkpoint-600
   - checkpoint-900
   - checkpoint-1200

🔄 Resuming training from last checkpoint...
Loading model from checkpoint-1200

Epoch 5/8: 100%|██████████| 300/300 [15:23<00:00]
{'loss': 3.245, 'learning_rate': 8.5e-05, 'epoch': 5.0}

Epoch 6/8: 100%|██████████| 300/300 [15:21<00:00]
{'loss': 2.987, 'learning_rate': 7.0e-05, 'epoch': 6.0}

Epoch 7/8: 100%|██████████| 300/300 [15:19<00:00]
{'loss': 2.756, 'learning_rate': 5.5e-05, 'epoch': 7.0}

Epoch 8/8: 100%|██████████| 300/300 [15:22<00:00]
{'loss': 2.543, 'learning_rate': 4.0e-05, 'epoch': 8.0}

✓ Training completed!
  Final loss: 2.543
✓ Adapter saved: /content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3/adapter_weights
✓ Training curves saved
```

---

## Skenario Praktis: Training Terputus di Epoch 4

### Situasi
- Training dimulai dengan 8 epochs
- Epoch 1-4 sudah selesai (checkpoint-1200 tersimpan)
- Tiba-tiba mati lampu / Colab disconnect
- Ingin melanjutkan dari epoch 5

### Solusi

**Jalankan cells berikut di Colab baru:**

```python
# ========================================
# CELL 1: Setup environment
# ========================================
!pip install -q adapters transformers datasets accelerate evaluate rouge_score bert_score
from google.colab import drive
drive.mount('/content/drive')

import os, sys, zipfile, shutil
sys.path.insert(0, '/content')

if not os.path.exists('/content/src'):
    shutil.copy('/content/drive/MyDrive/dataset_aqg/src_finetuned.zip', '/content/')
    with zipfile.ZipFile('/content/src_finetuned.zip', 'r') as z:
        z.extractall('/content/')

print('✓ Environment ready')

# ========================================
# CELL 2: Load model & dataset
# ========================================
from src.finetuned.utils.adapter_loader import load_model_with_adapter, print_adapter_info
from src.finetuned.data.dataset_loader import DatasetLoader
from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
from src.finetuned.training.adapter_trainer import AdapterTrainer

model, tokenizer = load_model_with_adapter(
    model_name='LazarusNLP/IndoNanoT5-base',
    adapter_name='mcq_generation',
    adapter_config='pfeiffer',
    reduction_factor=12,
    device='cuda'
)
print_adapter_info(model, tokenizer)

loader = DatasetLoader()
TASK_DIR = '/content/dataset_aqg/dataset-task-spesifc/'
train_dataset = loader.load_dataset(TASK_DIR, split='train')
val_dataset = loader.load_dataset(TASK_DIR, split='validation')

print(f'✓ Model & dataset loaded')

# ========================================
# CELL 3: 🔑 RESUME TRAINING
# ========================================
CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/adapter_v3'

trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    metrics_calculator=MetricsCalculator(),
    output_dir=CHECKPOINT_DIR,
    max_length=512
)

training_args = trainer.setup_training(
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01
)

print("🔄 Resuming training from last checkpoint...")
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    resume_from_checkpoint=True
)

print(f'✓ Training completed!')

# ========================================
# CELL 4: Save & Evaluate
# ========================================
adapter_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 12,
        "num_train_epochs": 8,
        "learning_rate": 1e-4,
    }
)

trainer.plot_training_curves(save_path=f'{CHECKPOINT_DIR}/training_curves.png')

print(f'✓ Adapter saved: {adapter_path}')
print('✓ Training curves saved')
```

---

## Checklist Resume Training

Sebelum resume training, pastikan:

- [ ] Google Drive sudah di-mount
- [ ] Source code sudah di-extract
- [ ] GPU tersedia (T4 atau lebih baik)
- [ ] CHECKPOINT_DIR path benar
- [ ] Checkpoint folder tidak kosong
- [ ] Training config sama dengan training pertama
- [ ] Dataset sudah di-load dengan benar

---

## FAQ

**Q: Berapa lama resume training?**
A: Tergantung epoch yang tersisa. Jika terputus di epoch 4 dari 8, maka 4 epoch × ~1 jam = ~4 jam.

**Q: Apakah metrics akan reset?**
A: Tidak. Metrics akan dilanjutkan dari checkpoint terakhir.

**Q: Bisa resume dari checkpoint yang lebih lama?**
A: Ya, gunakan Metode 2 (Manual Resume) dan specify checkpoint yang ingin digunakan.

**Q: Apakah perlu training ulang dari awal?**
A: Tidak. Sistem kami otomatis menyimpan checkpoint, jadi Anda hanya perlu resume.

**Q: Bagaimana jika ingin mengubah hyperparameter?**
A: Tidak disarankan. Jika ingin mengubah, lebih baik training dari awal dengan config baru.

---

## Referensi

- **Checkpoint System:** HuggingFace Trainer (built-in)
- **Adapter Framework:** https://docs.adapterhub.ml/
- **Training Configuration:** `.kiro/specs/adapter-training/design.md`
- **Notebook Implementation:** `src/finetuned/notebooks/04_task_specific_training.ipynb`

---

**Last Updated:** April 2026
**Status:** ✅ Production Ready
