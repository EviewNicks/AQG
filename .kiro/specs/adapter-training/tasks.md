# Implementation Tasks: Adapter-Based Fine-tuning Notebook

## Overview

Create notebook `03_task_specific_training_v3.ipynb` yang menggunakan Adapter Layers (bukan LoRA) untuk fine-tuning IndoNanoT5 pada task AQG.

## Tasks

- [ ] 1. Create notebook structure dan setup environment
  - Create file `src/finetuned/notebooks/03_task_specific_training_v3.ipynb`
  - Add notebook header dengan version info dan key differences dari v2
  - Add cell untuk install dependencies (adapter-transformers, transformers, datasets, evaluate)
  - Add cell untuk mount Google Drive dan setup paths
  - Add cell untuk verify GPU availability
  - _Requirements: 1.1, 8.1_

- [ ] 2. Implement model loading dengan adapter layers
  - [ ] 2.1 Create model loading cell
    - Load base model using `AutoAdapterModel.from_pretrained()`
    - Add adapter dengan Pfeiffer configuration (reduction_factor=12)
    - Activate adapter untuk training
    - Freeze base model parameters
    - Print trainable parameters (~3.6%)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [ ] 2.2 Add model info display
    - Display model architecture
    - Show adapter configuration
    - Show memory usage
    - Compare dengan LoRA approach (v2)
    - _Requirements: 1.3, 1.4_

- [ ] 3. Implement dataset loading dengan backward compatibility
  - [ ] 3.1 Create dataset loading cell
    - Load datasets from `dataset-task-spesifc/`
    - Support both `target` (v2) and `output` (v3) fields
    - Validate dataset structure
    - Display sample entries
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [ ] 3.2 Create preprocessing function
    - Tokenize inputs dengan max_length=512
    - Handle both field names automatically
    - Apply dynamic padding via DataCollator
    - _Requirements: 2.4, 2.5_

- [ ] 4. Implement baseline evaluation
  - Create evaluation cell untuk pre-trained model
  - Compute BLEU-4 dan ROUGE-L pada 10 samples
  - Display baseline metrics
  - Save baseline untuk comparison
  - _Requirements: 5.1_

- [ ] 5. Configure training arguments
  - [ ] 5.1 Create training configuration cell
    - Set num_train_epochs=8 (user request)
    - Set per_device_train_batch_size=4
    - Set gradient_accumulation_steps=2
    - Set learning_rate=1e-4
    - Set warmup_steps=50
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 5.2 Enable memory optimizations
    - Enable gradient_checkpointing=True
    - Enable fp16=True
    - Set dataloader_num_workers=2
    - Set dataloader_pin_memory=True
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 6. Implement training loop
  - [ ] 6.1 Create trainer initialization
    - Initialize Seq2SeqTrainer dengan adapter model
    - Setup DataCollatorForSeq2Seq dengan label_pad_token_id=-100
    - Setup compute_metrics function
    - Add early stopping callback
    - _Requirements: 5.2, 5.3_
  
  - [ ] 6.2 Create training execution cell
    - Start training dengan progress monitoring
    - Log metrics setiap epoch
    - Save checkpoints ke Google Drive
    - Display training time dan final loss
    - _Requirements: 5.1, 5.4, 5.5_

- [ ] 7. Implement model saving
  - Save adapter weights (~5MB)
  - Save training configuration
  - Save final metrics
  - Copy to Google Drive untuk persistence
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 8. Implement comprehensive evaluation
  - [ ] 8.1 Create evaluation cell
    - Evaluate on full test set
    - Compute BLEU-4, ROUGE-L, BERTScore
    - Compare dengan baseline metrics
    - Display metrics comparison table
    - _Requirements: 7.1, 7.3_
  
  - [ ] 8.2 Generate sample outputs
    - Generate 20 sample outputs dengan num_beams=4
    - Save samples ke JSON file
    - Display first 3 samples
    - _Requirements: 7.2_

- [ ] 9. Create visualization dan reporting
  - [ ] 9.1 Plot training curves
    - Plot training loss over epochs
    - Plot validation BLEU-4 over epochs
    - Save plot ke Google Drive
    - _Requirements: 5.5_
  
  - [ ] 9.2 Generate final report
    - Create evaluation report JSON
    - Include baseline vs final metrics
    - Include training configuration
    - Include performance comparison dengan v2 (LoRA)
    - _Requirements: 7.4, 8.4_

- [ ] 10. Add error handling dan documentation
  - [ ] 10.1 Add error handling cells
    - Handle OOM errors dengan suggestions
    - Handle missing dependencies
    - Handle dataset loading failures
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 10.2 Add documentation cells
    - Add markdown cells explaining adapter layers
    - Add comparison dengan LoRA approach
    - Add expected results dan performance metrics
    - Add troubleshooting guide
    - _Requirements: 8.2, 8.3, 8.5_

- [ ] 11. Test notebook end-to-end
  - Run all cells sequentially
  - Verify no errors
  - Verify memory usage < 14GB
  - Verify training completes dalam 6-8 hours
  - Verify final BLEU-4 > 0.20
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

## Notes

- Notebook harus self-contained dan dapat dijalankan di Google Colab
- Semua dependencies harus di-install otomatis
- Checkpoint harus disimpan ke Google Drive untuk persistence
- Memory usage harus dimonitor untuk avoid OOM
- Training time expected: 6-8 hours pada T4 GPU

## Success Criteria

- ✓ Notebook runs end-to-end tanpa errors
- ✓ Adapter layers successfully added dan trained
- ✓ Trainable parameters ~3.6% (8.9M dari 248M)
- ✓ Memory usage < 14GB peak
- ✓ Training completes dalam 8 hours
- ✓ Final BLEU-4 > 0.20
- ✓ Final ROUGE-L > 0.25
- ✓ Model saved successfully ke Google Drive

