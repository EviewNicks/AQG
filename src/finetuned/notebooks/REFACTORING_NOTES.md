# Refactoring Notes: v3 Notebook (Adapter-Based Training)

## 📋 Summary

Notebook v3 telah di-refactor dari **15 sections** menjadi **10 sections** yang clean dan modular, mengikuti struktur v2.

## 🎯 Changes Made

### 1. **Created New Modules**

#### `src/finetuned/utils/adapter_loader.py`
**Purpose:** Load model dengan adapter layers (modular approach)

**Functions:**
- `load_model_with_adapter()` - Load IndoNanoT5 + add adapter
- `print_adapter_info()` - Print parameter statistics

**Why:** Memisahkan logic adapter setup dari notebook, membuat code reusable

---

#### `src/finetuned/training/adapter_trainer.py`
**Purpose:** Handle training configuration dan execution untuk adapter-based fine-tuning

**Class:** `AdapterTrainer`

**Methods:**
- `preprocess_dataset()` - Tokenize dataset dengan backward compatibility
- `setup_training()` - Configure training arguments
- `compute_metrics()` - Compute BLEU & ROUGE during training
- `train()` - Execute training
- `save_adapter()` - Save adapter weights & config
- `plot_training_curves()` - Visualize training progress

**Why:** Menggabungkan sections 5-8 (preprocessing, config, metrics, trainer) menjadi 1 module

---

### 2. **Notebook Structure (10 Sections)**

| Section | Title | Description | Module Used |
|---------|-------|-------------|-------------|
| 1 | Setup Environment | Install deps, mount Drive, extract src | - |
| 2 | Load Model with Adapter | Load IndoNanoT5 + adapter layers | `adapter_loader` |
| 3 | Load Dataset | Load & validate dataset | `dataset_loader` |
| 4 | Baseline Evaluation | Evaluate pre-trained model | `model_evaluator` |
| 5 | Configure Training | Setup trainer & training args | `adapter_trainer` |
| 6 | Start Training | Train model (6-8 hours) | `adapter_trainer` |
| 7 | Save Adapter & Visualize | Save weights & plot curves | `adapter_trainer` |
| 8 | Final Evaluation | Comprehensive test set eval | `model_evaluator` |
| 9 | Generate Sample Outputs | Generate 20 samples | `model_evaluator` |
| 10 | Final Summary | Compare metrics & save report | - |

---

### 3. **Comparison: Before vs After**

#### **Before (15 sections):**
```
1. Setup Environment
2. Load Model with Adapter (inline code)
3. Load Dataset
4. Baseline Evaluation
5. Preprocess Dataset (inline)
6. Configure Training (inline)
7. Setup Metrics (inline)
8. Initialize Trainer (inline)
9. Start Training
10. Save Adapter (inline)
11. Plot Training Curves (inline)
12. Final Evaluation
13. Generate Samples
14. Final Summary
15. How to Load Adapter
```

#### **After (10 sections):**
```
1. Setup Environment
2. Load Model with Adapter (module: adapter_loader)
3. Load Dataset (module: dataset_loader)
4. Baseline Evaluation (module: model_evaluator)
5. Configure Training (module: adapter_trainer)
6. Start Training (module: adapter_trainer)
7. Save Adapter & Visualize (module: adapter_trainer)
8. Final Evaluation (module: model_evaluator)
9. Generate Sample Outputs (module: model_evaluator)
10. Final Summary
```

---

## ✅ Benefits of Refactoring

### 1. **Cleaner Notebook**
- Reduced from ~500 lines → ~200 lines
- Easier to read and understand
- Matches v2 structure (consistency)

### 2. **Modular Code**
- Reusable modules for adapter training
- Easier to test and maintain
- Can be used in other notebooks

### 3. **Better Organization**
- Clear separation: notebook = orchestration, modules = logic
- Follows v2 pattern (LoRA) but for adapters
- Easier to debug

### 4. **Backward Compatibility**
- Supports both `target` (v2) and `output` (v3) field names
- Works with existing datasets

---

## 🔧 How to Use

### **In Colab:**

1. Upload `src_finetuned.zip` to Google Drive
2. Open `03_task_specific_training_v3.ipynb`
3. Run cells sequentially
4. Training will take 6-8 hours on T4 GPU

### **Key Configuration:**

```python
# In Section 5: Configure Training
trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc,
    output_dir=CHECKPOINT_DIR,
    max_length=512
)

training_args = trainer.setup_training(
    num_train_epochs=8,              # User request
    per_device_train_batch_size=4,   # Safe for T4
    gradient_accumulation_steps=2,   # Effective BS = 8
    learning_rate=1e-4,              # Standard for adapters
    warmup_steps=50
)
```

---

## 📊 Expected Results

| Metric | Baseline | After Training | Target |
|--------|----------|----------------|--------|
| BLEU-4 | 0.005 | 0.20-0.28 | ≥ 0.20 |
| ROUGE-L | 0.0 | 0.25-0.35 | ≥ 0.25 |
| Training Loss | 39 | 2-5 | < 5 |
| Training Time | - | 6-8 hours | - |
| Memory Usage | - | 12-14GB | < 14GB |

---

## 🆚 Comparison with v2 (LoRA)

| Aspect | v2 (LoRA) | v3 (Adapter) |
|--------|-----------|--------------|
| **Method** | LoRA (PEFT) | Adapter Layers |
| **Trainable Params** | 0.36% (~0.9M) | 3.6% (~8.9M) |
| **Memory** | 8-10GB | 12-14GB |
| **Training Time** | 4-6 hours | 6-8 hours |
| **Performance** | Near full FT | 99.6% of full FT |
| **Inference** | +5-10ms latency | No overhead |
| **Stability** | Good | Excellent |
| **Best For** | Large models (>1B) | Small models (<1B) |

---

## 📝 Notes

1. **Why not use `task_trainer.py`?**
   - `task_trainer.py` is LoRA-specific (uses PEFT)
   - Adapter layers require different setup (adapter-transformers library)
   - Created `adapter_trainer.py` as adapter-specific alternative

2. **Why inline code in Section 2?**
   - Adapter setup is straightforward
   - Module `adapter_loader.py` keeps it clean
   - Similar to v2's `load_model_with_lora()`

3. **Backward Compatibility:**
   - Supports both `target` (v2) and `output` (v3) field names
   - Automatically detects which field to use
   - No need to modify existing datasets

---

## 🚀 Next Steps

1. Test notebook in Colab
2. Verify all modules work correctly
3. Run training and validate results
4. Compare with v2 (LoRA) performance
5. Document findings in training report

---

**Last Updated:** April 2026  
**Version:** 3.0 (Refactored)  
**Status:** ✅ Ready for testing
