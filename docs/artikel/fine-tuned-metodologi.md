
## 3.3. Fine-tuning Model (IndoT5 + LoRA)

Model IndoT5 akan di-*fine-tune* menggunakan teknik *Low-Rank Adaptation* (LoRA) untuk mengadaptasinya secara efisien pada tugas pembuatan soal kuis pemrograman Python dalam bahasa Indonesia.

### 3.3.0. Persiapan Fine-tuning

Sebelum memulai proses fine-tuning, dilakukan serangkaian verifikasi dan persiapan untuk memastikan kesiapan dataset dan environment:

**1. Verifikasi Kompatibilitas Dataset**
- Load dataset menggunakan HuggingFace `datasets` library untuk memastikan format JSONL valid
- Validasi struktur setiap entri: field `input`, `target`, dan `metadata` harus ada
- Konfirmasi split ratio dan stratifikasi sesuai spesifikasi (domain: 80/10/10, task-specific: 70/15/15)
- Verifikasi tidak ada data corruption atau missing values

**2. Validasi Tokenizer dengan Markdown**
- Test IndoT5 tokenizer (`Wikidepia/IndoT5-base`) dengan sample data yang mengandung markdown
- Verifikasi handling markdown formatting (`#`, `**`, `` ` ``, `\n`) tidak menyebabkan tokenization errors
- Analisis distribusi panjang token untuk memastikan mayoritas sample berada dalam batas `max_length=512`
- Konfirmasi tidak ada truncation issues pada code blocks yang panjang

**3. Setup Training Environment**
- GPU availability check: minimal 16GB VRAM (NVIDIA T4, V100, atau A100)
- Install dependencies: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`
- Konfigurasi hyperparameters awal berdasarkan best practices LoRA:
  * LoRA rank (`r`): 8
  * LoRA alpha (`α`): 16
  * Learning rate: 2e-4
  * Batch size: 8 (dengan gradient accumulation jika diperlukan)
  * Epochs: 6 untuk domain adaptation, 3 untuk task-specific
  * Warmup steps: 10% dari total steps

**4. Baseline Evaluation**
- Inference IndoT5 base model (pre-fine-tuning) pada 10 sample dari validation set
- Catat baseline metrics: BLEU-4, ROUGE-L, BERTScore
- Establish performance benchmark untuk perbandingan post-training
- Dokumentasikan contoh output untuk analisis kualitatif

### 3.3.1. Pemilihan Model Dasar

Menggunakan IndoT5 (`Wikidepia/IndoT5-base`) sebagai model *encoder-decoder* dasar yang telah dilatih pada korpus monolingual Indonesia. Model ini dipilih karena:
- Arsitektur T5 yang terbukti efektif untuk task text-to-text
- Pre-training pada bahasa Indonesia memastikan pemahaman struktur bahasa yang baik
- Ukuran model yang reasonable (~250M parameters) memungkinkan fine-tuning dengan resource terbatas

### 3.3.2. Implementasi LoRA

LoRA akan diimplementasikan pada lapisan-lapisan tertentu dari model IndoT5 untuk mengurangi jumlah parameter yang perlu dilatih, sehingga mempercepat proses *fine-tuning* dan mengurangi kebutuhan komputasi. Konfigurasi LoRA:
- Target modules: `q_proj`, `v_proj` (attention layers)
- LoRA rank (`r`): 8
- LoRA alpha (`α`): 16
- Dropout: 0.1
- Trainable parameters: ~0.5% dari total model parameters

### 3.3.3. Pelatihan Model

Model akan dilatih dalam dua tahap sesuai strategi hibrida dengan konfigurasi hyperparameter yang dioptimalkan untuk Google Colab T4 GPU (15GB VRAM):

**Tahap 1: Domain Adaptation (6 epochs)**

Dataset dan Objective:
- Dataset: 340 entri (118 span corruption + 222 QA generik)
- Split: 80/10/10 (train: 271, validation: 33, test: 36)
- Objective: Adaptasi terminologi teknis Python dan gaya instruksional Indonesia
- Expected outcome: Model memahami struktur kalimat teknis dan terminologi domain Python

Konfigurasi Training:
- Learning rate: 2e-4 (higher untuk domain adaptation)
- Batch size: 8 per device
- Gradient accumulation steps: 4 (effective batch size: 32)
- Warmup steps: 50 (~10% dari total training steps)
- Max sequence length: 512 tokens
- Optimizer: AdamW dengan weight decay 0.01
- Learning rate scheduler: Linear decay dengan warmup
- Mixed precision training: FP16 untuk efisiensi memory

Monitoring Metrics:
- Training loss: Expected decrease dari ~3.0 ke ~1.5
- Validation loss: Expected decrease dari ~2.8 ke ~1.8
- Perplexity: Expected decrease dari ~16.0 ke ~6.0
- Reconstruction accuracy: Dihitung implicitly dari validation loss
- GPU utilization dan memory usage
- Training speed (samples/second)

Checkpoint Management:
- Save strategy: Setiap epoch (total 6 checkpoints)
- Retention policy: Keep last 3 checkpoints untuk menghemat storage
- Checkpoint content: Model weights (LoRA adapters only ~5MB), optimizer state, training config, epoch number, metrics history
- Backup: Automatic copy ke Google Drive (`/content/drive/MyDrive/aqg_checkpoints/domain/`)
- Best model selection: Berdasarkan lowest validation loss
- Final output: `indot5-python-domain` (best checkpoint)

**Tahap 2: Task-Specific AQG (3 epochs)**

Dataset dan Objective:
- Dataset: 1,262 entri (674 MCQ + 588 Code Completion)
- Split: 70/15/15 (train: 876, validation: 175, test: 211)
- Stratifikasi: Berdasarkan module_name dan difficulty untuk representasi seimbang
- Objective: Pembelajaran pola generasi soal kuis dengan format terstruktur dan distraktor yang plausible
- Expected outcome: Model dapat generate soal MCQ dan Code Completion dengan kualitas tinggi

Konfigurasi Training:
- Learning rate: 1e-4 (lower untuk fine-tuning yang lebih halus)
- Batch size: 8 per device
- Gradient accumulation steps: 4 (effective batch size: 32)
- Warmup steps: 30 (~10% dari total training steps)
- Max sequence length: 512 tokens
- Optimizer: AdamW dengan weight decay 0.01
- Learning rate scheduler: Linear decay dengan warmup
- Mixed precision training: FP16
- Load from checkpoint: `indot5-python-domain` (output dari Stage 1)

Monitoring Metrics:
- BLEU-4: Expected increase dari ~0.10 (baseline) ke ~0.35-0.45
- ROUGE-L: Expected range ~0.40-0.50
- BERTScore F1: Expected range ~0.75-0.85
- Distinct-1 dan Distinct-2: Untuk mengukur diversity output
- Training/validation loss per epoch
- Generation quality: Sample outputs setiap epoch untuk qualitative analysis

Checkpoint Management:
- Save strategy: Setiap epoch (total 3 checkpoints)
- Retention policy: Keep best 2 checkpoints berdasarkan BLEU-4 score
- Evaluation strategy: Evaluate pada validation set setiap epoch
- Best model selection: Berdasarkan highest BLEU-4 score
- Final output: `indot5-python-aqg` (best checkpoint)
- Backup: Automatic copy ke Google Drive (`/content/drive/MyDrive/aqg_checkpoints/aqg/`)

**Error Handling dan Recovery**

Untuk memastikan robustness training process, implementasi mencakup:

1. **CUDA Out of Memory (OOM) Handling:**
   - Detection: `torch.cuda.OutOfMemoryError`
   - Recovery: Clear CUDA cache, reduce batch size by 50%, increase gradient accumulation proportionally
   - Fallback: Enable gradient checkpointing jika masih OOM

2. **Session Disconnect Recovery:**
   - Automatic checkpoint saving setiap epoch
   - Resume script: Load latest checkpoint dan continue dari epoch terakhir
   - State preservation: Optimizer state, learning rate scheduler, training metrics history

3. **Early Stopping:**
   - Trigger: Validation loss increases untuk 2 consecutive epochs
   - Action: Stop training, load best checkpoint berdasarkan validation metrics
   - Logging: Warning message dengan metrics history untuk analysis

4. **Data Loading Failures:**
   - Validation: Pre-check dataset files existence dan format validity
   - Error logging: Detailed error message dengan file path yang bermasalah
   - Graceful degradation: Skip corrupted entries dengan warning

**Training Environment**

Platform: Google Colab (Free Tier)
- GPU: NVIDIA T4 (15GB VRAM)
- RAM: ~12GB system RAM
- Storage: Google Drive untuk persistent checkpoint storage
- Session timeout: ~12 hours (checkpoints saved setiap epoch untuk recovery)

Dependencies:
```python
transformers>=4.35.0      # HuggingFace Transformers
peft>=0.7.0               # LoRA implementation
datasets>=2.15.0          # Dataset loading
accelerate>=0.25.0        # Training acceleration
torch>=2.1.0              # PyTorch
evaluate>=0.4.0           # Metrics calculation
rouge_score>=0.1.2        # ROUGE metrics
bert_score>=0.3.13        # BERTScore
```

**Expected Training Time**

Berdasarkan estimasi pada T4 GPU:
- Domain Adaptation (6 epochs): ~2-3 hours
  * ~20-30 minutes per epoch
  * Total training steps: ~500 steps
  * Training speed: ~1-2 samples/second
  
- Task-Specific AQG (3 epochs): ~1-2 hours
  * ~20-40 minutes per epoch
  * Total training steps: ~300 steps
  * Training speed: ~1-2 samples/second

Total training time: ~3-5 hours untuk complete two-stage pipeline

**Code Organization**

Implementasi menggunakan modular architecture untuk reusability:

```
src/finetuning/
├── data/
│   ├── dataset_loader.py      # Load dan validate JSONL datasets
│   └── tokenizer_tester.py    # Test tokenizer compatibility
├── model/
│   └── model_setup.py          # Setup IndoT5 + LoRA
├── training/
│   ├── domain_trainer.py       # Domain adaptation trainer
│   └── task_trainer.py         # Task-specific trainer
├── evaluation/
│   ├── metrics_calculator.py   # BLEU, ROUGE, BERTScore
│   └── model_evaluator.py      # Comprehensive evaluation
├── utils/
│   ├── checkpoint_manager.py   # Checkpoint save/load/cleanup
│   └── colab_helper.py         # Colab-specific utilities
└── config/
    └── training_config.yaml    # Centralized configuration
```

Setiap module dirancang sebagai reusable component dengan clear interfaces, memudahkan maintenance dan future iterations.


