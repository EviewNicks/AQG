# 4 configure training

============================================================
TRAINING CONFIGURATION
============================================================

✓ Hyperparameters (Spec Defaults):
  Epochs:              6
  Learning Rate:       0.0002
  Warmup Steps:        50
  Batch Size:          4
  Gradient Accum:      4
  Effective Batch:     16
  FP16:                True

✓ Training Estimates:
  Steps per epoch:     ~15
  Total steps:         ~90
  Warmup duration:     ~3.3 epochs
  Estimated time:      30-45 minutes on T4 GPU

✓ Checkpoint Directory:
  /content/drive/MyDrive/dataset_aqg/checkpoints/domain

============================================================
✅ Configuration ready for training
============================================================

# 5 Start Training

Starting domain adaptation training...
Checkpoints will be saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain
============================================================

============================================================
STARTING DOMAIN ADAPTATION TRAINING
============================================================

✓ Model moved to GPU: Tesla T4
  Model device: cuda:0
Preprocessing datasets...
Preprocessing 253 samples...
  Columns: ['input', 'target']
  Batch size: 32
  Removing columns: ['input', 'target']

✓ Preprocessed 253 samples
  Sample label check: 217 valid tokens, 0 masked (-100)
Preprocessing 33 samples...
  Columns: ['input', 'target']
  Batch size: 32
  Removing columns: ['input', 'target']

✓ Preprocessed 33 samples
  Sample label check: 24 valid tokens, 0 masked (-100)

=== Dataset Size After Preprocessing ===
Train samples (actual): 253
Eval samples (actual):  33

=== Training Configuration ===
Epochs: 6
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16
Learning rate: 0.0002
Warmup steps: 50
FP16: True
Train samples: 253
Eval samples: 33
Starting training...

=== Training Complete ===
Final training loss: 38.9387
Training time: 203.85 seconds
Training samples per second: 7.45
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/domain/training_results.json

Training completed in 0.06 hours
Final training loss: 38.9387

# 6 Save Best Model 

✓ Best model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain/indot5-python-domain
Model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain/indot5-python-domain
✓ Training curves saved to /content/drive/MyDrive/dataset_aqg/checkpoints/domain/training_curves.png
✓ Training curves saved

# 7  Evaluate on Validation Set

============================================================
INFERENCE TEST
============================================================

⚠️  IMPORTANT: Adding task prefix "question: " for inference
   (T5 models require task prefix for correct output)

Input:  Apa itu list dalam Python?
Output: - – - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
------------------------------------------------------------
Input:  Jelaskan fungsi dalam Python.
Output: - - – - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
------------------------------------------------------------
Input:  Apa itu variable dalam Python?
Output: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
------------------------------------------------------------

✅ Inference test complete
============================================================


Loading model from checkpoint: /content/drive/MyDrive/dataset_aqg/checkpoints/domain/indot5-python-domain
✓ Model loaded | device: cuda:0
✓ Evaluator re-initialized


Evaluating on validation set...

============================================================
EVALUATING ON TEST SET
============================================================

Evaluating 33 samples...
  Processed 10/33 samples...
  Processed 20/33 samples...
  Processed 30/33 samples...
✓ Generated 33 predictions
Computing metrics for 33 samples...
  Computing BLEU...

Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0000
  BLEU-1:   0.0700
  BLEU-2:   0.0140
  BLEU-3:   0.0003
  BLEU-4:   0.0000

ROUGE Scores:
  ROUGE-1:  0.1203
  ROUGE-2:  0.0018
  ROUGE-L:  0.0882

Diversity:
  Distinct-1: 0.1253
  Distinct-2: 0.6875

============================================================

=== Validation Metrics ===
  BLEU-4:  0.0000
  ROUGE-L: 0.0882
  ROUGE-1: 0.1203

