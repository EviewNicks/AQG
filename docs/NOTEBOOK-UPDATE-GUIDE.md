# Notebook Update Guide - Manual Changes Required

**Date:** April 30, 2026  
**Notebook:** `src/finetuned/notebooks/04_task_specific_training.ipynb`  
**Status:** Code fixes complete, notebook updates needed

---

## ✅ What's Already Fixed (No Action Needed)

### 1. Training Resume Logic
**File:** `src/finetuned/training/adapter_trainer.py`
- ✅ Auto-detects checkpoints correctly
- ✅ Training continues to all 10 epochs (not stopping at epoch 3)
- ✅ Proper logging added

### 2. FileNotFoundError Fix
**File:** `src/finetuned/evaluation/model_evaluator.py`
- ✅ Auto-creates folders before saving samples
- ✅ No more FileNotFoundError

### 3. Full Output Display
**File:** `src/finetuned/evaluation/model_evaluator.py`
- ✅ Shows complete input, reference, and prediction (no truncation)
- ✅ Better formatting with emojis and separators
- ✅ Perfect for manual quality assessment

---

## 📝 Manual Notebook Updates Required

You need to update **3 cells** in the notebook:

### Cell 7: Save Adapter & Visualize

**Issue:** `NameError: name 'elapsed' is not defined`

**Current Code:**
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
        "training_time_hours": elapsed  # ❌ Not defined if Cell 6 not run!
    }
)
```

**Updated Code (Option 1 - Add try-except):**
```python
# ✅ Handle missing elapsed variable
try:
    elapsed
except NameError:
    elapsed = 0.0
    print("⚠️ Training time not available (Cell 6 not run)")

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

**Updated Code (Option 2 - Remove training_time_hours):**
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

---

### Cell 9: Generate Sample Outputs

**Issue:** Duplicate output (generate_samples already prints everything)

**Current Code:**
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
        print(f"Generated: {sample['prediction']}...")
```

**Updated Code:**
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

**Why:** The `generate_samples()` method already prints all 20 samples with full formatting:
```
================================================================================
Sample 1/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Duck typing tidak berkaitan langsung...

✅ REFERENCE:
question: Dengan konsep apa duck typing lebih erat kaitannya?
answer: Pemrograman berorientasi objek (OOP)
distractors: Dynamic typing | Loosely typed | Static typing | Type checking

🤖 PREDICTION:
question: apa yang dimaksud dengan duck typing? 
answer: dynamic typing lebih erat dengan python...
distractors: tipe data | static typing | loosely typed | type checking

📊 BLEU Score: 0.0000
================================================================================
```

---

### Cell 10: Final Summary

**Issue:** `NameError: name 'elapsed' is not defined` (and `trainable`, `total`)

**Current Code:**
```python
# ... (comparison code)

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

**Updated Code:**
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
    print("⚠️ Training time not available (Cell 6 not run)")

try:
    trainable
    total
except NameError:
    trainable = 0
    total = 1
    print("⚠️ Model parameter info not available (Cell 2 not run)")

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

---

## 🎯 Summary of Changes

| Cell | Issue | Fix |
|------|-------|-----|
| Cell 7 | `NameError: elapsed` | Add try-except or remove `training_time_hours` |
| Cell 9 | Duplicate output | Remove print loop (already printed by `generate_samples()`) |
| Cell 10 | `NameError: elapsed, trainable, total` | Add try-except for all missing variables |

---

## ✅ Benefits After Update

1. **Cell Independence:** All cells can run independently without NameError
2. **No Duplicate Output:** Clean, readable output in Cell 9
3. **Full Text Display:** Can see complete predictions for manual testing
4. **Training Completes:** All 10 epochs (not stopping at epoch 3)
5. **No FileNotFoundError:** Folders auto-created

---

## 📚 Related Documentation

- **Detailed Fixes:** `docs/fix-training-issues.md`
- **Summary:** `docs/SUMMARY-FIXES.md`
- **Error Logs:** `docs/error.md`

---

**Status:** Ready for manual notebook update  
**Priority:** Medium (cells work if run in order, but fail if run independently)
