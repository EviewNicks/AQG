# Requirements Document: IndoT5 Fine-tuning untuk AQG

## Introduction

Spec ini mendefinisikan requirements untuk fine-tuning model IndoT5 dengan LoRA untuk task Automatic Question Generation (AQG) Python. Menggunakan pendekatan two-stage hybrid: (1) Domain Adaptation untuk pemahaman materi Python, (2) Task-Specific AQG untuk generasi soal kuis.

## Glossary

- **IndoT5**: Model T5 (Text-to-Text Transfer Transformer) yang di-pre-train pada korpus bahasa Indonesia
- **LoRA**: Low-Rank Adaptation, teknik PEFT (Parameter-Efficient Fine-Tuning) yang mengurangi parameter trainable
- **Domain Adaptation**: Fine-tuning tahap 1 untuk adaptasi terminologi dan gaya bahasa domain Python
- **Task-Specific AQG**: Fine-tuning tahap 2 untuk generasi soal kuis dengan format spesifik
- **Checkpoint**: Snapshot model yang disimpan selama training untuk recovery dan evaluation
- **BLEU/ROUGE/BERTScore**: Metrics evaluasi untuk mengukur kualitas text generation

## Requirements

### Requirement 1: Dataset Loading dan Validation

**User Story:** As a researcher, I want to load and validate training datasets, so that I can ensure data quality before training.

#### Acceptance Criteria

1. WHEN loading domain adaptation dataset, THE System SHALL load 340 entri dari `dataset_aqg/output_domain/` dengan split 80/10/10
2. WHEN loading task-specific dataset, THE System SHALL load 1,262 entri dari `dataset_aqg/dataset-task-spesifc/` dengan split 70/15/15
3. WHEN validating dataset structure, THE System SHALL verify bahwa setiap entri memiliki field `input`, `target`, dan `metadata`
4. WHEN checking data integrity, THE System SHALL detect missing values, duplicate entries, atau corrupted data
5. WHEN analyzing token distribution, THE System SHALL report berapa persen sample yang melebihi max_length 512 tokens

### Requirement 2: Tokenizer Setup dan Testing

**User Story:** As a researcher, I want to test tokenizer compatibility with markdown formatting, so that I can ensure proper tokenization.

#### Acceptance Criteria

1. WHEN loading IndoT5 tokenizer, THE System SHALL successfully load tokenizer dari `Wikidepia/IndoT5-base`
2. WHEN tokenizing markdown content, THE System SHALL handle special characters (`#`, `**`, `` ` ``, `\n`) tanpa error
3. WHEN tokenizing code blocks, THE System SHALL preserve code block integrity tanpa truncation di tengah block
4. WHEN analyzing token lengths, THE System SHALL generate histogram distribusi panjang token
5. WHEN detecting OOV tokens, THE System SHALL report token yang tidak ada di vocabulary

### Requirement 3: Model Setup dengan LoRA

**User Story:** As a researcher, I want to setup IndoT5 model dengan LoRA adapters, so that I can train efficiently dengan reduced parameters.

#### Acceptance Criteria

1. WHEN loading base model, THE System SHALL load `Wikidepia/IndoT5-base` dengan ~250M parameters
2. WHEN applying LoRA configuration, THE System SHALL set rank=8, alpha=16, dropout=0.1, target_modules=["q", "v"]
3. WHEN calculating trainable parameters, THE System SHALL report ~0.5% dari total parameters (~1.25M trainable)
4. WHEN printing model summary, THE System SHALL display LoRA configuration dan trainable parameter count
5. WHEN checking GPU memory, THE System SHALL verify bahwa model fit dalam 16GB VRAM

### Requirement 4: Domain Adaptation Training (Stage 1)

**User Story:** As a researcher, I want to train model pada domain Python, so that model memahami terminologi dan konteks Python.

#### Acceptance Criteria

1. WHEN starting domain adaptation, THE System SHALL load 340 entri domain dataset
2. WHEN configuring training, THE System SHALL set epochs=6, batch_size=8, learning_rate=2e-4, warmup_steps=50
3. WHEN training each epoch, THE System SHALL log training loss, validation loss, dan perplexity
4. WHEN saving checkpoints, THE System SHALL save checkpoint setiap epoch dengan naming `checkpoint-epoch-{N}`
5. WHEN training completes, THE System SHALL save best model sebagai `indot5-python-domain`
6. WHEN monitoring metrics, THE System SHALL verify training loss decreases dari ~3.0 ke ~1.5
7. WHEN evaluating validation, THE System SHALL verify validation loss decreases dari ~2.8 ke ~1.8

### Requirement 5: Task-Specific AQG Training (Stage 2)

**User Story:** As a researcher, I want to train model untuk AQG task, so that model dapat generate soal kuis dengan format yang benar.

#### Acceptance Criteria

1. WHEN starting task-specific training, THE System SHALL load checkpoint dari Stage 1 (`indot5-python-domain`)
2. WHEN loading AQG dataset, THE System SHALL load 1,262 entri task-specific dataset
3. WHEN configuring training, THE System SHALL set epochs=3, batch_size=8, learning_rate=1e-4, warmup_steps=30
4. WHEN training each epoch, THE System SHALL log BLEU-4, ROUGE-L, dan BERTScore pada validation set
5. WHEN saving checkpoints, THE System SHALL keep only best 2 checkpoints berdasarkan eval_bleu
6. WHEN training completes, THE System SHALL save final model sebagai `indot5-python-aqg`
7. WHEN evaluating metrics, THE System SHALL verify BLEU-4 increases dari ~0.10 ke ~0.35-0.45

### Requirement 6: Baseline Evaluation

**User Story:** As a researcher, I want to evaluate pre-trained model performance, so that I can establish baseline metrics.

#### Acceptance Criteria

1. WHEN loading pre-trained model, THE System SHALL load `Wikidepia/IndoT5-base` tanpa fine-tuning
2. WHEN running inference, THE System SHALL generate output untuk 10 sample dari validation set
3. WHEN calculating metrics, THE System SHALL compute BLEU-4, ROUGE-L, dan BERTScore
4. WHEN documenting baseline, THE System SHALL save sample outputs untuk qualitative analysis
5. WHEN comparing results, THE System SHALL verify baseline BLEU-4 < 0.15 (expected low performance)

### Requirement 7: Training Monitoring dan Logging

**User Story:** As a researcher, I want to monitor training progress, so that I can detect issues early dan optimize hyperparameters.

#### Acceptance Criteria

1. WHEN training starts, THE System SHALL log GPU utilization, memory usage, dan training speed
2. WHEN each batch completes, THE System SHALL log loss value setiap 50-100 steps
3. WHEN each epoch completes, THE System SHALL evaluate pada validation set dan log metrics
4. WHEN detecting anomalies, THE System SHALL warn jika loss tidak decreases selama 2 epochs
5. WHEN training completes, THE System SHALL generate training summary report dengan plots

### Requirement 8: Checkpoint Management

**User Story:** As a researcher, I want to save and load checkpoints, so that I can resume training jika session terputus.

#### Acceptance Criteria

1. WHEN saving checkpoint, THE System SHALL save model weights, optimizer state, dan training config
2. WHEN checkpoint directory full, THE System SHALL keep only last 3 checkpoints untuk domain adaptation
3. WHEN checkpoint directory full, THE System SHALL keep only last 2 checkpoints untuk task-specific
4. WHEN resuming training, THE System SHALL load checkpoint dan continue dari epoch terakhir
5. WHEN saving to Google Drive, THE System SHALL copy checkpoints ke `/content/drive/MyDrive/aqg_checkpoints/`

### Requirement 9: Model Evaluation dan Testing

**User Story:** As a researcher, I want to evaluate trained model, so that I can measure performance improvement.

#### Acceptance Criteria

1. WHEN evaluating on test set, THE System SHALL compute BLEU-4, ROUGE-L, BERTScore pada seluruh test set
2. WHEN generating samples, THE System SHALL generate output untuk 20 random samples dari test set
3. WHEN analyzing diversity, THE System SHALL compute Distinct-1 dan Distinct-2 metrics
4. WHEN comparing baseline, THE System SHALL show improvement percentage untuk setiap metric
5. WHEN documenting results, THE System SHALL save evaluation report dengan sample outputs

### Requirement 10: Error Handling dan Recovery

**User Story:** As a researcher, I want robust error handling, so that training dapat recover dari failures.

#### Acceptance Criteria

1. IF CUDA out of memory error occurs, THEN THE System SHALL reduce batch_size dan retry
2. IF session disconnects, THEN THE System SHALL have saved latest checkpoint untuk resume
3. IF dataset loading fails, THEN THE System SHALL log error message dengan file path yang bermasalah
4. IF model loading fails, THEN THE System SHALL verify internet connection dan retry download
5. IF validation metrics degrade, THEN THE System SHALL trigger early stopping dan load best checkpoint

### Requirement 11: Colab-Specific Requirements

**User Story:** As a researcher using Google Colab, I want optimized setup untuk Colab environment, so that I can train efficiently dengan free tier.

#### Acceptance Criteria

1. WHEN checking GPU, THE System SHALL verify T4 GPU available dengan minimal 15GB VRAM
2. WHEN installing dependencies, THE System SHALL install transformers, peft, datasets, accelerate, bitsandbytes
3. WHEN mounting Google Drive, THE System SHALL mount drive untuk persistent storage
4. WHEN session timeout approaching, THE System SHALL save checkpoint setiap epoch
5. WHEN training completes, THE System SHALL save final model ke Google Drive dan offer download

### Requirement 12: Code Organization

**User Story:** As a researcher, I want well-organized code, so that I can reuse dan maintain code easily.

#### Acceptance Criteria

1. WHEN organizing code, THE System SHALL structure code sebagai Python modules (data/, model/, training/, evaluation/)
2. WHEN loading data, THE System SHALL use reusable function `load_aqg_dataset(data_dir, split)`
3. WHEN setting up model, THE System SHALL use reusable function `setup_model_with_lora(model_name, lora_config)`
4. WHEN training, THE System SHALL use reusable function `train_model(model, dataset, training_args)`
5. WHEN evaluating, THE System SHALL use reusable function `evaluate_model(model, dataset, metrics)`
