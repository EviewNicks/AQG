# Fix Training Issues - Update Guide

## Masalah yang Diperbaiki

### 1. ✅ Training Berhenti Prematur di Epoch 3/10
**File:** `src/finetuned/training/adapter_trainer.py`

**Root Cause:** 
- Resume logic tidak mendeteksi checkpoint dengan benar
- Training berhenti di step 1056 (epoch 3) padahal seharusnya 3520 (epoch 10)

**Solusi:**
- Tambah auto-detection checkpoint yang lebih robust
- Tambah logging untuk menunjukkan total epochs
- Fix resume logic agar training lanjut sampai selesai

### 2. ✅ FileNotFoundError di Generate Samples
**File:** `src/finetuned/evaluation/model_evaluator.py`

**Root Cause:**
- Folder `evaluation_results/09-indonanoot5-report/` belum dibuat
- Code langsung coba save file tanpa create folder dulu

**Solusi:**
- Auto-create folder jika belum ada menggunakan `Path.mkdir(parents=True, exist_ok=True)`

---

## Update Notebook Manual

### Cell 6: Start Training

**Location:** `src/finetuned/notebooks/04_task_specific_training.ipynb` - Cell #6

**PREVIOUS CODE:**
```python
import time
import os
from pathlib import Path

start_time = time.time()

# Ensure checkpoint directory exists
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# Check for existing checkpoints
checkpoints = []
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('checkpoint-')]

# Decide whether to resume
if checkpoints:
    print(f"📂 Found {len(checkpoints)} checkpoint(s): {sorted(checkpoints)}")
    print(f"🔄 Resuming from last checkpoint")
    resume = True
else:
    print("🆕 No checkpoints found - starting fresh training")
    resume = False

# Train
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint=resume  # ✅ Only True if checkpoints exist
)

elapsed = (time.time() - start_time) / 3600
print(f'\n✓ Training completed in {elapsed:.2f} hours')
print(f'  Final training loss: {results["training_loss"]:.4f}')
```

**UPDATED CODE:**
```python
import time
import os
from pathlib import Path

start_time = time.time()

# Ensure checkpoint directory exists
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# ✅ SIMPLIFIED: Let trainer handle checkpoint detection
# Just pass True to auto-resume, or False to start fresh
resume = True  # Set to False if you want fresh training

# Train - trainer will auto-detect and resume from last checkpoint
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint=resume
)

elapsed = (time.time() - start_time) / 3600
print(f'\n✓ Training completed in {elapsed:.2f} hours')
print(f'  Final training loss: {results["training_loss"]:.4f}')
```

**CHANGES:**
1. ❌ Removed manual checkpoint detection logic
2. ✅ Simplified to just `resume = True` or `False`
3. ✅ Trainer now handles checkpoint detection internally
4. ✅ Training akan lanjut sampai epoch terakhir (10 epochs)

---

### Cell 9: Generate Sample Outputs

**Location:** `src/finetuned/notebooks/04_task_specific_training.ipynb` - Cell #9

**PREVIOUS CODE:**
```python
EVAL_DIR = '/content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report'

samples = evaluator_final.generate_samples(
    test_dataset=test_dataset,
    num_samples=20,
    num_beams=4,
    save_path=f'{EVAL_DIR}/sample_outputs.json'
)

print(f'✓ {len(samples)} samples generated')

# Preview first 3 samples
if samples:
    print('\n=== Sample Outputs ===')
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Input: {sample['input']}...")
        print(f"Generated: {sample['generated']}...")
```

**UPDATED CODE:**
```python
EVAL_DIR = '/content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report'

# ✅ No need to create folder manually - generate_samples will auto-create it
samples = evaluator_final.generate_samples(
    test_dataset=test_dataset,
    num_samples=20,
    num_beams=4,
    save_path=f'{EVAL_DIR}/sample_outputs.json'
)

print(f'✓ {len(samples)} samples generated')

# Preview first 3 samples
if samples:
    print('\n=== Sample Outputs ===')
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Input: {sample['input']}...")
        print(f"Generated: {sample['prediction']}...")  # ✅ Changed from 'generated' to 'prediction'
```

**CHANGES:**
1. ✅ Folder akan auto-create jika belum ada
2. ✅ Fixed field name dari `'generated'` ke `'prediction'` (sesuai dengan output dari generate_samples)
3. ❌ No need manual `Path(EVAL_DIR).mkdir()` anymore

---

## Testing Checklist

Setelah update notebook, test dengan:

### Test 1: Fresh Training (No Checkpoints)
```python
# Delete checkpoints first
!rm -rf /content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report/checkpoint-*

# Run Cell 6 with resume=False
resume = False
results = trainer.train(...)

# Expected: Training runs from epoch 1 to 10 (3520 steps)
```

### Test 2: Resume Training (With Checkpoints)
```python
# Assume training stopped at epoch 3 (checkpoint-1056 exists)

# Run Cell 6 with resume=True
resume = True
results = trainer.train(...)

# Expected: 
# - Detects checkpoint-1056
# - Resumes from epoch 4
# - Continues to epoch 10 (3520 steps total)
```

### Test 3: Generate Samples (No Folder)
```python
# Delete eval folder first
!rm -rf /content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report

# Run Cell 9
samples = evaluator_final.generate_samples(...)

# Expected:
# - Folder auto-created
# - Samples saved successfully
# - No FileNotFoundError
```

---

## Expected Output After Fix

### Cell 6 Output (Fresh Training):
```
🆕 Starting fresh training (no resume)

============================================================
STARTING TRAINING
============================================================
Training with Adapter Layers (d=64, ~3.6% trainable params)
Expected time: 6-8 hours on T4 GPU
Total epochs: 10
============================================================

[1056/3520 53:12 < 2:04:23, 0.33 it/s, Epoch 3/10]
[2112/3520 1:46:24 < 1:11:11, 0.33 it/s, Epoch 6/10]
[3168/3520 2:39:36 < 0:18:00, 0.33 it/s, Epoch 9/10]
[3520/3520 3:32:48 < 0:00:00, 0.33 it/s, Epoch 10/10]

✓ Training completed in 3.55 hours
  Final training loss: 2.1234
```

### Cell 6 Output (Resume Training):
```
📂 Found 3 checkpoint(s): ['checkpoint-352', 'checkpoint-704', 'checkpoint-1056']
🔄 Resuming from: checkpoint-1056

============================================================
STARTING TRAINING
============================================================
Training with Adapter Layers (d=64, ~3.6% trainable params)
Expected time: 6-8 hours on T4 GPU
Total epochs: 10
============================================================

Loading model from checkpoint-1056...

[1408/3520 0:17:36 < 1:46:47, 0.33 it/s, Epoch 4/10]
[2464/3520 1:10:48 < 0:53:36, 0.33 it/s, Epoch 7/10]
[3520/3520 2:04:00 < 0:00:00, 0.33 it/s, Epoch 10/10]

✓ Training completed in 2.07 hours
  Final training loss: 2.1234
```

### Cell 9 Output:
```
Generating 20 sample outputs...

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Duck typing tidak berkaitan...
Reference: question: Dengan konsep apa duck typing...
Prediction: question: apa yang dimaksud dengan duck typing?...
BLEU: 0.1234

✓ Samples saved to /content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report/sample_outputs.json

✓ 20 samples generated
```

---

## Summary

**Files Modified:**
1. ✅ `src/finetuned/training/adapter_trainer.py` - Fix training resume logic
2. ✅ `src/finetuned/evaluation/model_evaluator.py` - Auto-create folder for samples

**Notebook Updates Required:**
1. ✅ Cell 6: Simplify resume logic
2. ✅ Cell 9: Fix field name `'generated'` → `'prediction'`

**Benefits:**
- ✅ Training akan lanjut sampai epoch 10 (tidak berhenti di epoch 3)
- ✅ No more FileNotFoundError saat generate samples
- ✅ Code lebih simple dan robust
- ✅ Auto-detection checkpoint yang lebih baik

---

**Last Updated:** April 30, 2026
**Status:** ✅ Ready to Deploy


---

### Cell 7: Save Adapter & Visualize

**Location:** `src/finetuned/notebooks/04_task_specific_training.ipynb` - Cell #7

**ISSUE:** `NameError: name 'elapsed' is not defined`

**Root Cause:**
- Variable `elapsed` defined in Cell 6
- Not available in Cell 7 if kernel restarted or Cell 6 skipped

**PREVIOUS CODE:**
```python
# Save adapter weights
adapter_save_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 6,
        "trainable_params": trainable,
        "total_params": total,
        "num_train_epochs": 10,
        "learning_rate": 5e-5,
        "training_time_hours": elapsed  # ❌ Not defined!
    }
)
```

**UPDATED CODE (Option 1 - Safe):**
```python
# ✅ Handle missing elapsed variable
try:
    elapsed
except NameError:
    elapsed = 0.0
    print("⚠️ Training time not available")

# Save adapter weights
adapter_save_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 6,
        "trainable_params": trainable,
        "total_params": total,
        "num_train_epochs": 10,
        "learning_rate": 5e-5,
        "training_time_hours": elapsed
    }
)

# Plot training curves
trainer.plot_training_curves(
    save_path=f'{CHECKPOINT_DIR}/training_curves.png'
)
```

**UPDATED CODE (Option 2 - Simplest):**
```python
# Save adapter weights (without training_time_hours)
adapter_save_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 6,
        "trainable_params": trainable,
        "total_params": total,
        "num_train_epochs": 10,
        "learning_rate": 5e-5
        # Removed training_time_hours
    }
)

# Plot training curves
trainer.plot_training_curves(
    save_path=f'{CHECKPOINT_DIR}/training_curves.png'
)
```

**CHANGES:**
1. ✅ Added try-except untuk handle missing `elapsed`
2. ✅ Or removed `training_time_hours` dari config
3. ✅ Cell 7 sekarang bisa dijalankan independent

---


---

## 🔍 Improvement: Full Output Display for Manual Testing

### Issue
Previous output was truncated at 150 characters, making manual testing impossible:
```python
# OLD - Truncated
print(f"Input: {input_text[:150]}...")
print(f"Reference: {reference[:150]}...")
print(f"Prediction: {prediction[:150]}...")
```

### Solution
**File Modified:** `src/finetuned/evaluation/model_evaluator.py`

**Method:** `generate_samples()`

**Changes:**
- ✅ Display full input, reference, and prediction
- ✅ Better formatting with separators
- ✅ Easier manual inspection

**New Output Format:**
```
================================================================================
Sample 1/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Duck typing tidak berkaitan langsung dengan dynamic typing atau loosely typed. Konsep duck typing lebih erat dengan pemrograman berorientasi objek (OOP) dan fokus pada kemampuan object melakukan operasi tertentu, bukan tipe datanya.

✅ REFERENCE:
question: Dengan konsep apa duck typing lebih erat kaitannya?
answer: Pemrograman berorientasi objek (OOP)
distractors: Dynamic typing | Loosely typed | Static typing | Type checking

🤖 PREDICTION:
question: apa yang dimaksud dengan duck typing? 
answer: dynamic typing lebih erat dengan python dan fokus pada kemampuan object melakukan operasi tertentu
distractors: tipe data | static typing | loosely typed | type checking

📊 BLEU Score: 0.0000
================================================================================
```

**Benefits:**
- ✅ Can see full output for manual quality assessment
- ✅ Easier to spot errors in question/answer/distractors
- ✅ Better readability with clear sections
- ✅ No more guessing what the full output looks like

---


---

### Cell 9: Generate Sample Outputs (UPDATED FIX)

**Location:** `src/finetuned/notebooks/04_task_specific_training.ipynb` - Cell #9

**ISSUE:**
- Duplicate output - `generate_samples()` already prints all 20 samples with full formatting, no need to print again

**CURRENT CODE:**
```python
EVAL_DIR = '/content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report'

samples = evaluator_final.generate_samples(
    test_dataset=test_dataset,
    num_samples=20,
    num_beams=4,
    save_path=f'{EVAL_DIR}/sample_outputs.json'
)

print(f'✓ {len(samples)} samples generated')

# Preview first 3 samples
if samples:
    print('\n=== Sample Outputs ===')
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Input: {sample['input']}...")
        print(f"Generated: {sample['prediction']}...")  # ✅ Field name is correct
```

**ISSUE:** The print loop creates duplicate output because `generate_samples()` already prints all 20 samples with full formatting (input, reference, prediction, BLEU score).

**UPDATED CODE:**
```python
EVAL_DIR = '/content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report'

# ✅ generate_samples() already prints full output with nice formatting
# No need to print again here
samples = evaluator_final.generate_samples(
    test_dataset=test_dataset,
    num_samples=20,
    num_beams=4,
    save_path=f'{EVAL_DIR}/sample_outputs.json'
)

print(f'\n✓ {len(samples)} samples generated and saved')
print(f'✓ Full output displayed above with BLEU scores')
```

**CHANGES:**
1. ✅ Removed duplicate print loop (already printed by `generate_samples()`)
2. ✅ Cleaner output - no more duplicate samples
3. ✅ Full output already shown by `generate_samples()` with better formatting (see example below)

**Example Output from `generate_samples()`:**
```
================================================================================
Sample 1/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Duck typing tidak berkaitan langsung dengan dynamic typing atau loosely typed. Konsep duck typing lebih erat dengan pemrograman berorientasi objek (OOP) dan fokus pada kemampuan object melakukan operasi tertentu, bukan tipe datanya.

✅ REFERENCE:
question: Dengan konsep apa duck typing lebih erat kaitannya?
answer: Pemrograman berorientasi objek (OOP)
distractors: Dynamic typing | Loosely typed | Static typing | Type checking

🤖 PREDICTION:
question: apa yang dimaksud dengan duck typing? 
answer: dynamic typing lebih erat dengan python dan fokus pada kemampuan object melakukan operasi tertentu
distractors: tipe data | static typing | loosely typed | type checking

📊 BLEU Score: 0.0000
================================================================================
```

**Why This Works:**
- `generate_samples()` method already prints all 20 samples with full formatting
- No need to print again in notebook
- Avoids duplicate output and keeps notebook clean

---

### Cell 10: Final Summary (UPDATED FIX)

**Location:** `src/finetuned/notebooks/04_task_specific_training.ipynb` - Cell #10

**ISSUE:** `NameError: name 'elapsed' is not defined`

**PREVIOUS CODE:**
```python
report = {
    'version': '3.0 (Adapter Layers)',
    'baseline_metrics': baseline_metrics,
    'final_metrics': final_metrics,
    'comparison': comparison,
    'training_time_hours': elapsed,  # ❌ Not defined!
    'adapter_path': adapter_save_path,
    'config': {...}
}

print(f'Training Time: {elapsed:.2f} hours')  # ❌ Not defined!
print(f'Trainable: {100*trainable/total:.2f}%')  # ❌ Not defined!
```

**UPDATED CODE:**
```python
import json
from pathlib import Path

# Compare with baseline
comparison = evaluator_final.compare_with_baseline(
    finetuned_metrics=final_metrics,
    baseline_metrics=baseline_metrics
)

# ✅ Handle missing variables
try:
    elapsed
except NameError:
    elapsed = 0.0
    print("⚠️ Training time not available")

try:
    trainable
    total
except NameError:
    trainable = 0
    total = 1
    print("⚠️ Model parameter info not available")

# Save evaluation report
Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
report = {
    'version': '3.0 (Adapter Layers)',
    'baseline_metrics': baseline_metrics,
    'final_metrics': final_metrics,
    'comparison': comparison,
    'training_time_hours': elapsed,
    'adapter_path': adapter_save_path,
    'config': {
        'adapter_config': 'pfeiffer',
        'reduction_factor': 6,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'epochs': 10
    }
}

with open(f'{EVAL_DIR}/evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Print summary
print('\n' + '='*60)
print('ADAPTER-BASED AQG TRAINING SUMMARY')
print('='*60)
print(f'Method: Adapter Layers (d=128)')
if elapsed > 0:
    print(f'Training Time: {elapsed:.2f} hours')
if trainable > 0 and total > 1:
    print(f'Trainable: {100*trainable/total:.2f}%')

print(f'\nMetrics Comparison:')
print(f"  BLEU-4:  {baseline_metrics.get('bleu_4',0):.4f} → {final_metrics.get('bleu_4',0):.4f}")
print(f"  ROUGE-L: {baseline_metrics.get('rouge_l',0):.4f} → {final_metrics.get('rouge_l',0):.4f}")

bleu_improvement = comparison.get('bleu_4_improvement_pct', 0)
print(f'\nBLEU-4 Improvement: {bleu_improvement:+.1f}%')

if final_metrics.get('bleu_4', 0) >= 0.20:
    print('\n✓ SUCCESS: BLEU-4 target achieved (>= 0.20)')
else:
    print(f"\n⚠ BLEU-4 = {final_metrics.get('bleu_4',0):.4f} (target: >= 0.20)")
    print('  Consider: more epochs or adjust hyperparameters')

print('\n✓ Fine-tuning pipeline complete!')
print(f'  Adapter: {adapter_save_path}')
print(f'  Report: {EVAL_DIR}/evaluation_report.json')
print(f'  Samples: {EVAL_DIR}/sample_outputs.json')

print('\n' + '='*60)
print('HOW TO LOAD TRAINED ADAPTER')
print('='*60)
print('from adapters import AutoAdapterModel')
print('from transformers import AutoTokenizer')
print('')
print('model = AutoAdapterModel.from_pretrained("LazarusNLP/IndoNanoT5-base")')
print('tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")')
print(f'model.load_adapter("{adapter_save_path}")')
print('model.set_active_adapters("mcq_generation")')
print('')
print('# Generate')
print('inputs = tokenizer("generate_mcq: [CONTEXT]", return_tensors="pt")')
print('outputs = model.generate(**inputs, max_length=512, num_beams=4)')
print('print(tokenizer.decode(outputs[0], skip_special_tokens=True))')
```

**CHANGES:**
1. ✅ Added try-except untuk handle missing `elapsed`
2. ✅ Added try-except untuk handle missing `trainable` and `total`
3. ✅ Conditional print - only show if variables available
4. ✅ Cell 10 sekarang bisa dijalankan independent

---
