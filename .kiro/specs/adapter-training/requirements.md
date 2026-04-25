# Requirements Document: Adapter-Based Fine-tuning untuk IndoNanoT5 AQG

## Introduction

Spec ini mendefinisikan requirements untuk fine-tuning IndoNanoT5 menggunakan **Adapter Layers** (bukan LoRA) untuk task Automatic Question Generation (AQG). Berdasarkan penelitian dari `docs/fine-tuned-setup.md`, adapter layers memberikan 99.6% performance dari full fine-tuning dengan hanya 3.6% trainable parameters.

## Glossary

- **Adapter Layers**: Parameter-efficient fine-tuning method yang menambahkan bottleneck layers ke model
- **Reduction Factor**: Rasio dimensi hidden ke adapter dimension (768/64 = 12)
- **Pfeiffer Configuration**: Standard adapter architecture dengan down-projection → activation → up-projection
- **IndoNanoT5**: Model T5 monolingual Indonesia (248M parameters)
- **Task-Specific Dataset**: Dataset 1500 samples untuk MCQ generation

## Requirements

### Requirement 1: Adapter Layer Setup

**User Story:** As a researcher, I want to setup adapter layers pada IndoNanoT5, so that I can train efficiently dengan memory constraints T4 GPU.

#### Acceptance Criteria

1. WHEN loading base model, THE System SHALL load `LazarusNLP/IndoNanoT5-base` tanpa LoRA
2. WHEN adding adapter, THE System SHALL use Pfeiffer configuration dengan reduction_factor=12 (d=64)
3. WHEN calculating trainable parameters, THE System SHALL report ~8.9M trainable (~3.6% dari 248M)
4. WHEN checking memory, THE System SHALL verify model fit dalam 14GB VRAM
5. WHEN activating adapter, THE System SHALL freeze base model dan train hanya adapter parameters

### Requirement 2: Dataset Loading dengan Backward Compatibility

**User Story:** As a researcher, I want to load dataset dengan support untuk format v2 dan v3, so that I can use existing datasets.

#### Acceptance Criteria

1. WHEN loading dataset, THE System SHALL support field `target` (v2) dan `output` (v3)
2. WHEN loading from `dataset-task-spesifc/`, THE System SHALL load 1500 samples total
3. WHEN validating dataset, THE System SHALL verify required fields exist
4. WHEN preprocessing, THE System SHALL handle both field names automatically
5. WHEN tokenizing, THE System SHALL use max_length=512 untuk input dan output

### Requirement 3: Training Configuration untuk Adapter

**User Story:** As a researcher, I want to configure training sesuai best practices untuk adapter tuning, so that I can achieve optimal performance.

#### Acceptance Criteria

1. WHEN setting learning rate, THE System SHALL use 1e-4 (standard untuk adapter tuning)
2. WHEN setting batch size, THE System SHALL use per_device_batch_size=4 dengan gradient_accumulation=2
3. WHEN setting epochs, THE System SHALL train untuk 8 epochs (sesuai user request)
4. WHEN setting warmup, THE System SHALL use warmup_steps=50
5. WHEN enabling optimizations, THE System SHALL use gradient_checkpointing=True dan fp16=True

### Requirement 4: Memory Optimization untuk T4 GPU

**User Story:** As a researcher using T4 GPU, I want memory-efficient training, so that I can avoid OOM errors.

#### Acceptance Criteria

1. WHEN enabling gradient checkpointing, THE System SHALL reduce memory usage ~30%
2. WHEN using FP16, THE System SHALL reduce memory usage ~50%
3. WHEN monitoring memory, THE System SHALL report GPU memory usage < 14GB
4. WHEN batch processing, THE System SHALL use effective batch size = 8 (4×2)
5. WHEN encountering OOM, THE System SHALL provide clear error message dengan suggestions

### Requirement 5: Training Monitoring dan Evaluation

**User Story:** As a researcher, I want to monitor training progress, so that I can verify convergence.

#### Acceptance Criteria

1. WHEN training starts, THE System SHALL log initial baseline metrics (BLEU, ROUGE)
2. WHEN each epoch completes, THE System SHALL evaluate pada validation set
3. WHEN logging metrics, THE System SHALL track BLEU-4, ROUGE-L, dan training loss
4. WHEN saving checkpoints, THE System SHALL save best model based on eval_bleu_4
5. WHEN training completes, THE System SHALL generate training curves plot

### Requirement 6: Model Saving dan Export

**User Story:** As a researcher, I want to save trained adapter, so that I can deploy atau continue training later.

#### Acceptance Criteria

1. WHEN saving adapter, THE System SHALL save only adapter weights (~5MB)
2. WHEN saving checkpoint, THE System SHALL include adapter config dan training state
3. WHEN exporting model, THE System SHALL save ke Google Drive untuk persistence
4. WHEN loading saved adapter, THE System SHALL load base model + adapter weights
5. WHEN documenting model, THE System SHALL save training config dan final metrics

### Requirement 7: Evaluation dan Sample Generation

**User Story:** As a researcher, I want to evaluate trained model, so that I can measure performance improvement.

#### Acceptance Criteria

1. WHEN evaluating on test set, THE System SHALL compute BLEU-4, ROUGE-L, BERTScore
2. WHEN generating samples, THE System SHALL generate 20 sample outputs dengan num_beams=4
3. WHEN comparing metrics, THE System SHALL show improvement dari baseline
4. WHEN saving results, THE System SHALL save evaluation report sebagai JSON
5. WHEN displaying results, THE System SHALL show metrics comparison table

### Requirement 8: Notebook Structure dan Documentation

**User Story:** As a researcher, I want well-documented notebook, so that I can understand dan reproduce training.

#### Acceptance Criteria

1. WHEN opening notebook, THE System SHALL display clear version info dan key differences dari v2
2. WHEN running cells, THE System SHALL provide progress indicators dan status messages
3. WHEN documenting code, THE System SHALL explain adapter-specific configurations
4. WHEN showing results, THE System SHALL include comparison dengan LoRA approach (v2)
5. WHEN completing training, THE System SHALL provide summary dengan actionable insights

### Requirement 9: Error Handling dan Recovery

**User Story:** As a researcher, I want robust error handling, so that I can recover dari failures.

#### Acceptance Criteria

1. IF adapter library not installed, THEN THE System SHALL install `adapter-transformers`
2. IF GPU memory insufficient, THEN THE System SHALL suggest reducing batch size
3. IF dataset loading fails, THEN THE System SHALL provide clear error message
4. IF training diverges, THEN THE System SHALL trigger early stopping
5. IF session disconnects, THEN THE System SHALL have saved latest checkpoint

### Requirement 10: Performance Expectations

**User Story:** As a researcher, I want to know expected performance, so that I can validate training success.

#### Acceptance Criteria

1. WHEN training completes, THE System SHALL achieve BLEU-4 > 0.20
2. WHEN training completes, THE System SHALL achieve ROUGE-L > 0.25
3. WHEN comparing with LoRA, THE System SHALL show comparable atau better performance
4. WHEN measuring training time, THE System SHALL complete dalam 6-8 hours pada T4
5. WHEN measuring memory, THE System SHALL use < 14GB peak memory

