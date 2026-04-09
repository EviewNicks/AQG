# Roadmap Fine-tuning IndoT5 untuk AQG

## Overview

Dokumen ini menjelaskan langkah-langkah detail yang perlu dilakukan **sebelum** membuat spec plan untuk tahap 2.3 Fine-tuning Model. Roadmap ini memastikan kita memiliki semua prerequisite dan pemahaman yang cukup sebelum eksekusi.

---

## Phase 0: Pre-Planning Assessment (SEKARANG)

### ✅ Status Saat Ini

**Completed:**
- ✅ Dataset preparation (Stage 2.1 & 2.2)
  - Domain adaptation: 340 entri
  - Task-specific AQG: 1,262 entri
- ✅ Dokumentasi metodologi updated
- ✅ Evaluasi gap dan keputusan desain
- ✅ Architecture document updated dengan detail tahap 2.3

**Ready to Start:**
- 🟡 Tahap 2.3.0: Persiapan Fine-tuning
- 🟡 Tahap 2.3.1-2.3.3: Training execution

---

## Phase 1: Technical Verification (1-2 hari)

**Objective:** Memastikan semua technical requirements terpenuhi sebelum training

### 1.1. Environment Check

**Tasks:**
- [ ] Verifikasi akses GPU (Colab / Local / Cloud)
  - Minimal: T4 15GB VRAM (Colab free tier)
  - Ideal: V100 16GB atau A100 40GB
- [ ] Test koneksi internet untuk download model
- [ ] Estimate storage requirements (~10GB untuk checkpoints)

**Deliverable:** Environment specification document

### 1.2. Dataset Verification

**Tasks:**
- [ ] Load dataset dengan HuggingFace `datasets` library
  ```python
  from datasets import load_dataset
  
  # Test domain adaptation dataset
  domain_ds = load_dataset('json', data_files={
      'train': 'dataset_aqg/output_domain/train.jsonl',
      'validation': 'dataset_aqg/output_domain/validation.jsonl',
      'test': 'dataset_aqg/output_domain/test.jsonl'
  })
  
  # Test task-specific dataset
  aqg_ds = load_dataset('json', data_files={
      'train': 'dataset_aqg/dataset-task-spesifc/train.jsonl',
      'validation': 'dataset_aqg/dataset-task-spesifc/validation.jsonl',
      'test': 'dataset_aqg/dataset-task-spesifc/test.jsonl'
  })
  ```
- [ ] Validate struktur data (input, target, metadata)
- [ ] Check for missing values atau corrupted entries
- [ ] Verify split ratios dan stratification

**Deliverable:** Dataset validation report

### 1.3. Tokenizer Testing

**Tasks:**
- [ ] Load IndoT5 tokenizer
  ```python
  from transformers import AutoTokenizer
  
  tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoT5-base")
  ```
- [ ] Test dengan sample yang mengandung markdown
  ```python
  sample = "# Heading\n\n**Bold text** dan `code`\n\n```python\nprint('hello')\n```"
  tokens = tokenizer(sample, return_tensors="pt")
  print(f"Token count: {len(tokens['input_ids'][0])}")
  ```
- [ ] Analisis distribusi panjang token
  ```python
  import matplotlib.pyplot as plt
  
  lengths = [len(tokenizer(ex['input'])['input_ids']) for ex in aqg_ds['train']]
  plt.hist(lengths, bins=50)
  plt.axvline(x=512, color='r', linestyle='--', label='Max length')
  plt.xlabel('Token count')
  plt.ylabel('Frequency')
  plt.legend()
  plt.savefig('token_length_distribution.png')
  ```
- [ ] Check truncation rate (berapa % sample > 512 tokens)

**Deliverable:** Tokenizer analysis report dengan visualisasi

### 1.4. Baseline Evaluation

**Tasks:**
- [ ] Load IndoT5 base model (pre-fine-tuning)
  ```python
  from transformers import AutoModelForSeq2SeqLM
  
  model = AutoModelForSeq2SeqLM.from_pretrained("Wikidepia/IndoT5-base")
  ```
- [ ] Run inference pada 10 sample validation set
  ```python
  from transformers import pipeline
  
  generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
  
  for i, sample in enumerate(aqg_ds['validation'].select(range(10))):
      output = generator(sample['input'], max_length=512)
      print(f"Sample {i+1}:")
      print(f"Input: {sample['input'][:100]}...")
      print(f"Expected: {sample['target'][:100]}...")
      print(f"Generated: {output[0]['generated_text'][:100]}...")
      print("-" * 80)
  ```
- [ ] Calculate baseline metrics (BLEU, ROUGE, BERTScore)
- [ ] Dokumentasikan contoh output untuk analisis kualitatif

**Deliverable:** Baseline evaluation report

---

## Phase 2: Code Development (2-3 hari)

**Objective:** Develop training code sebagai Python modules (reusable, testable)

### 2.1. Project Structure Setup

**Tasks:**
- [ ] Create project structure
  ```
  aqg-finetuning/
  ├── src/
  │   ├── __init__.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── loader.py          # Dataset loading
  │   │   └── preprocessor.py    # Tokenization
  │   ├── model/
  │   │   ├── __init__.py
  │   │   ├── setup.py           # Model + LoRA setup
  │   │   └── config.py          # Training configs
  │   ├── training/
  │   │   ├── __init__.py
  │   │   ├── trainer.py         # Training loop
  │   │   └── callbacks.py       # Custom callbacks
  │   └── evaluation/
  │       ├── __init__.py
  │       ├── metrics.py         # BLEU, ROUGE, BERTScore
  │       └── analyzer.py        # Qualitative analysis
  ├── notebooks/
  │   ├── 01_data_verification.ipynb
  │   ├── 02_baseline_evaluation.ipynb
  │   ├── 03_domain_adaptation.ipynb
  │   └── 04_task_specific_training.ipynb
  ├── configs/
  │   ├── domain_adaptation.yaml
  │   └── task_specific.yaml
  ├── tests/
  │   ├── test_data_loader.py
  │   ├── test_model_setup.py
  │   └── test_metrics.py
  ├── requirements.txt
  └── README.md
  ```

**Deliverable:** Project skeleton dengan folder structure

### 2.2. Data Loading Module

**Tasks:**
- [ ] Implement `src/data/loader.py`
  - Function: `load_aqg_dataset(data_dir, split)`
  - Function: `get_dataset_statistics(dataset)`
- [ ] Implement `src/data/preprocessor.py`
  - Function: `tokenize_dataset(dataset, tokenizer, max_length)`
  - Function: `create_data_collator(tokenizer)`
- [ ] Write unit tests: `tests/test_data_loader.py`

**Deliverable:** Data loading module dengan tests

### 2.3. Model Setup Module

**Tasks:**
- [ ] Implement `src/model/setup.py`
  - Function: `load_base_model(model_name)`
  - Function: `apply_lora(model, lora_config)`
  - Function: `print_trainable_parameters(model)`
- [ ] Implement `src/model/config.py`
  - Class: `LoRAConfig` (dataclass untuk LoRA parameters)
  - Class: `TrainingConfig` (dataclass untuk training parameters)
- [ ] Write unit tests: `tests/test_model_setup.py`

**Deliverable:** Model setup module dengan tests

### 2.4. Training Module

**Tasks:**
- [ ] Implement `src/training/trainer.py`
  - Function: `train_domain_adaptation(model, dataset, config)`
  - Function: `train_task_specific(model, dataset, config)`
  - Function: `save_checkpoint(model, output_dir)`
- [ ] Implement `src/training/callbacks.py`
  - Class: `MetricsCallback` (log metrics ke tensorboard)
  - Class: `EarlyStoppingCallback` (stop jika no improvement)
- [ ] Write integration tests

**Deliverable:** Training module dengan callbacks

### 2.5. Evaluation Module

**Tasks:**
- [ ] Implement `src/evaluation/metrics.py`
  - Function: `compute_bleu(predictions, references)`
  - Function: `compute_rouge(predictions, references)`
  - Function: `compute_bertscore(predictions, references)`
- [ ] Implement `src/evaluation/analyzer.py`
  - Function: `analyze_outputs(predictions, references, metadata)`
  - Function: `generate_evaluation_report(metrics, samples)`
- [ ] Write unit tests: `tests/test_metrics.py`

**Deliverable:** Evaluation module dengan tests

---

## Phase 3: Local Testing (1 hari)

**Objective:** Test training pipeline locally dengan subset kecil data

### 3.1. Smoke Test

**Tasks:**
- [ ] Create mini dataset (10 train, 5 val samples)
- [ ] Run training untuk 1 epoch
- [ ] Verify:
  - Training loop berjalan tanpa error
  - Loss decreases
  - Checkpoints tersimpan dengan benar
  - Metrics dihitung dengan benar

**Deliverable:** Smoke test report

### 3.2. Integration Test

**Tasks:**
- [ ] Test full pipeline dengan 50 samples
- [ ] Run 2 epochs domain adaptation
- [ ] Run 1 epoch task-specific
- [ ] Verify end-to-end workflow

**Deliverable:** Integration test report

---

## Phase 4: Colab Preparation (0.5 hari)

**Objective:** Prepare code untuk deployment di Google Colab

### 4.1. Colab Notebook Creation

**Tasks:**
- [ ] Create `notebooks/colab_full_training.ipynb`
  - Cell 1: Environment setup (GPU check, install dependencies)
  - Cell 2: Clone repository / upload code
  - Cell 3: Upload dataset (via Google Drive atau manual)
  - Cell 4: Data verification
  - Cell 5: Baseline evaluation
  - Cell 6: Stage 1 - Domain adaptation training
  - Cell 7: Stage 2 - Task-specific training
  - Cell 8: Final evaluation
  - Cell 9: Save model (to Drive / download)

**Deliverable:** Colab notebook ready to run

### 4.2. Documentation

**Tasks:**
- [ ] Update README.md dengan:
  - Installation instructions
  - Usage examples
  - Colab setup guide
  - Troubleshooting section
- [ ] Create `docs/training-guide.md` dengan step-by-step instructions

**Deliverable:** Complete documentation

---

## Phase 5: Spec Plan Creation (0.5 hari)

**Objective:** Create formal spec plan untuk eksekusi training

### 5.1. Requirements Document

**Tasks:**
- [ ] Define user stories:
  - As a researcher, I want to fine-tune IndoT5 for AQG task
  - As a researcher, I want to evaluate model performance
  - As a researcher, I want to save and load trained models
- [ ] Define acceptance criteria untuk setiap requirement
- [ ] Define correctness properties (jika applicable)

**Deliverable:** `requirements.md`

### 5.2. Design Document

**Tasks:**
- [ ] Document architecture decisions
- [ ] Document training strategy (two-stage)
- [ ] Document evaluation metrics
- [ ] Document error handling

**Deliverable:** `design.md`

### 5.3. Tasks Document

**Tasks:**
- [ ] Break down implementation into discrete tasks
- [ ] Add testing tasks
- [ ] Add checkpoint tasks
- [ ] Estimate time for each task

**Deliverable:** `tasks.md`

---

## Phase 6: Execution (2-3 hari)

**Objective:** Execute training plan

### 6.1. Stage 1: Domain Adaptation

**Tasks:**
- [ ] Run training (6 epochs, ~30-45 minutes)
- [ ] Monitor metrics
- [ ] Save best checkpoint
- [ ] Evaluate on validation set

**Deliverable:** `indot5-python-domain` model checkpoint

### 6.2. Stage 2: Task-Specific AQG

**Tasks:**
- [ ] Load Stage 1 checkpoint
- [ ] Run training (3 epochs, ~1.5-2 hours)
- [ ] Monitor metrics
- [ ] Save best checkpoint
- [ ] Evaluate on test set

**Deliverable:** `indot5-python-aqg` final model

### 6.3. Final Evaluation

**Tasks:**
- [ ] Run comprehensive evaluation
- [ ] Generate evaluation report
- [ ] Analyze sample outputs
- [ ] Document findings

**Deliverable:** Evaluation report dengan recommendations

---

## Decision Points

### Decision 1: Local vs Colab Training

**Options:**
- **A. Local Training** (jika punya GPU ≥16GB)
  - Pros: Faster iteration, no session timeout, full control
  - Cons: Requires local GPU
- **B. Google Colab** (recommended untuk free tier)
  - Pros: Free GPU (T4 15GB), no local setup
  - Cons: Session timeout (12 hours), need to save checkpoints frequently

**Recommendation:** Start dengan Colab untuk MVP, migrate ke local/cloud jika perlu scale up

### Decision 2: Training Strategy

**Options:**
- **A. Two-Stage (Domain → Task-Specific)** ✅ CURRENT
  - Pros: Better domain adaptation, follows literature best practices
  - Cons: Longer training time
- **B. Single-Stage (Task-Specific only)**
  - Pros: Faster, simpler
  - Cons: May not adapt well to Python domain

**Recommendation:** Stick dengan two-stage approach (sudah divalidasi di literature)

### Decision 3: Evaluation Metrics

**Primary Metrics:**
- BLEU-4 (n-gram overlap)
- ROUGE-L (longest common subsequence)
- BERTScore (semantic similarity)

**Secondary Metrics:**
- Diversity (Distinct-1, Distinct-2)
- Perplexity
- Human evaluation (post-training, 50-100 samples)

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Technical Verification | 1-2 hari | Dataset ready |
| Phase 2: Code Development | 2-3 hari | Phase 1 complete |
| Phase 3: Local Testing | 1 hari | Phase 2 complete |
| Phase 4: Colab Preparation | 0.5 hari | Phase 3 complete |
| Phase 5: Spec Plan Creation | 0.5 hari | Phase 4 complete |
| Phase 6: Execution | 2-3 hari | Phase 5 complete |
| **TOTAL** | **7-10 hari** | - |

---

## Next Immediate Actions

**Sekarang (hari ini):**
1. ✅ Review roadmap ini dengan user
2. ⏭️ Mulai Phase 1.1: Environment Check
3. ⏭️ Mulai Phase 1.2: Dataset Verification

**Besok:**
1. Complete Phase 1 (Technical Verification)
2. Start Phase 2 (Code Development)

**Minggu ini:**
1. Complete Phase 1-4
2. Ready untuk training execution

---

## Questions to Answer Before Starting

1. **GPU Access:** Apakah kamu punya akses GPU local, atau akan pakai Colab?
2. **Time Commitment:** Apakah kamu bisa dedicate 7-10 hari untuk ini, atau perlu adjust timeline?
3. **Evaluation Strategy:** Apakah kamu akan melakukan human evaluation post-training?
4. **Model Deployment:** Setelah training, apakah model akan di-deploy untuk inference, atau hanya untuk research?

**Jawab pertanyaan ini untuk finalize roadmap!**
