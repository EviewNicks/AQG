# Design Document: Adapter-Based Fine-tuning untuk IndoNanoT5 AQG

## Overview

Sistem ini mengimplementasikan adapter-based fine-tuning untuk IndoNanoT5 model pada task Automatic Question Generation (AQG). Berdasarkan paper Houlsby et al. (2019), adapter layers mencapai 99.6% performance dari full fine-tuning dengan hanya 3.6% trainable parameters, making it ideal untuk T4 GPU constraints dan small dataset (1500 samples).

**Key Design Decisions:**
- Use Adapter Layers (bukan LoRA atau full fine-tuning)
- Adapter dimension d=64 (reduction_factor=12)
- Batch size 4 + gradient accumulation 2 (effective batch size 8)
- 8 epochs training (user request)
- Learning rate 1e-4 (standard untuk adapter tuning)

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                     │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              1. SETUP & DEPENDENCIES                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  - Install adapter-transformers                  │  │
│  │  - Install transformers, datasets, evaluate      │  │
│  │  - Mount Google Drive                            │  │
│  │  - Extract source code                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              2. MODEL SETUP (ADAPTER)                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Load Base Model: IndoNanoT5-base (248M)        │  │
│  │         ↓                                        │  │
│  │  Add Adapter: Pfeiffer config (d=64)            │  │
│  │         ↓                                        │  │
│  │  Freeze Base Model                               │  │
│  │         ↓                                        │  │
│  │  Train Only Adapter (~8.9M params)              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              3. DATASET LOADING                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Load from: dataset-task-spesifc/               │  │
│  │  - train.jsonl (1200 samples)                   │  │
│  │  - validation.jsonl (150 samples)               │  │
│  │  - test.jsonl (150 samples)                     │  │
│  │  Support: 'target' (v2) & 'output' (v3)        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              4. BASELINE EVALUATION                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Evaluate pre-trained model (10 samples)        │  │
│  │  Metrics: BLEU-4, ROUGE-L                       │  │
│  │  Expected: BLEU ~0.005, ROUGE ~0.0              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              5. TRAINING (8 EPOCHS)                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Config:                                         │  │
│  │  - Learning rate: 1e-4                          │  │
│  │  - Batch size: 4 (per device)                   │  │
│  │  - Gradient accumulation: 2                     │  │
│  │  - Effective batch size: 8                      │  │
│  │  - Epochs: 8                                    │  │
│  │  - Warmup steps: 50                             │  │
│  │  - FP16: True                                   │  │
│  │  - Gradient checkpointing: True                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              6. EVALUATION & ANALYSIS                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  - Comprehensive test set evaluation            │  │
│  │  - Generate 20 sample outputs                   │  │
│  │  - Compare with baseline                        │  │
│  │  - Save evaluation report                       │  │
│  │  - Plot training curves                         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Adapter Configuration

**Adapter Architecture (Pfeiffer):**
```
Input (768-dim)
    ↓
[Layer Norm]
    ↓
[Down-Projection: 768 → 64]
    ↓
[ReLU Activation]
    ↓
[Up-Projection: 64 → 768]
    ↓
[Residual Connection]
    ↓
Output (768-dim)
```

**Configuration:**
```python
from adapters import AdapterConfig

adapter_config = AdapterConfig.load(
    "pfeiffer",
    reduction_factor=12,  # 768 / 64 = 12
    non_linearity="relu"
)
```

**Parameter Breakdown:**
- Base model: 248M parameters (frozen)
- Adapter layers: ~2.4M parameters (trainable) for d=64
- Trainable percentage: 0.95% (Pfeiffer adapter)
- Memory footprint: ~12-14GB (vs ~32GB full fine-tuning)

**VERIFIED:** Actual trainable params = 2.38M (0.95%) ✅ CORRECT

**Note:** The 8.9M (3.6%) often mentioned refers to:
- Houlsby adapter (double_seq_bn) with d=64 (~4.8M), OR
- Pfeiffer adapter with d=256 (~9.6M)
- For Pfeiffer d=64: 2.4M (0.95%) is CORRECT and EXPECTED
- This is OPTIMAL for dataset size 5,560 samples

### 2. Model Loading Module

**Interface:**
```python
def load_model_with_adapter(
    model_name: str = 'LazarusNLP/IndoNanoT5-base',
    adapter_name: str = 'mcq_generation',
    reduction_factor: int = 12,
    device: str = 'cuda'
) -> Tuple[AdapterModel, Tokenizer]:
    """
    Load IndoNanoT5 dengan adapter layers.
    Uses NEW 'adapters' library (not deprecated 'adapter-transformers').
    
    Returns:
        (model_with_adapter, tokenizer)
    """
```

**Implementation Steps:**
1. Load base model using `AutoAdapterModel` or `transformers + adapters.init()`
2. Add adapter dengan Pfeiffer config (or 'seq_bn' in new naming)
3. Activate adapter untuk training
4. Freeze base model parameters
5. Move to GPU dan enable gradient checkpointing

**Loading Strategy (NEW Library):**
```python
import adapters
from adapters import AutoAdapterModel
from transformers import AutoModelForSeq2SeqLM

# Method 1: Direct loading (recommended)
try:
    model = AutoAdapterModel.from_pretrained(model_name)
except:
    # Method 2: Load with transformers, then init adapters
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    adapters.init(model)  # Initialize adapter support
```

**Key Changes from Old Library:**
- No need for `ignore_mismatched_sizes`, `trust_remote_code`, `_fast_init` flags
- Cleaner API, no state dict errors
- Config names: 'pfeiffer' → 'seq_bn' (but old names still work)
- Must call `adapters.init()` if using transformers model classes

### 3. Dataset Preprocessing

**Backward Compatibility:**
```python
def preprocess_dataset(examples):
    # Support both 'target' (v2) and 'output' (v3)
    target_field = "target" if "target" in examples else "output"
    
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True
    )
    
    labels = tokenizer(
        text_target=examples[target_field],
        max_length=512,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

### 4. Training Configuration (OPTIMIZED for T4 GPU)

**Seq2SeqTrainingArguments:**
```python
# Using AdapterTrainer for optimized configuration
from src.finetuned.training.adapter_trainer import AdapterTrainer

trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc,
    output_dir=CHECKPOINT_DIR,
    max_length=512
)

# Setup training with OPTIMIZED defaults
training_args = trainer.setup_training(
    num_train_epochs=10,
    per_device_train_batch_size=8,   # ✅ Optimized (was 4) - 2x increase
    per_device_eval_batch_size=16,   # ✅ Optimized (was 8) - 2x increase
    gradient_accumulation_steps=1,   # ✅ Optimized (was 2) - faster updates
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01
)

# Optimizations applied automatically:
# - gradient_checkpointing=False (disabled - not needed with adapters)
# - dataloader_num_workers=4 (increased from 2)
# - dataloader_prefetch_factor=2 (added for better pipeline)
# - fp16=True (mixed precision)
```

**Expected Training Metrics (OPTIMIZED):**
- Training time: 3-4 hours (T4 GPU) - **2x faster than before!** ⚡
- Memory usage: 13-15GB peak (better utilization)
- Training loss: 39 → 2-5
- BLEU-4: 0.005 → 0.20-0.28
- ROUGE-L: 0.0 → 0.25-0.35

**Performance Improvements:**
- **2x faster training** (6-8h → 3-4h for 8 epochs)
- **Better GPU utilization** (12-14GB → 13-15GB)
- **Same learning stability** (effective batch size = 8)
- **Faster evaluation** (2x faster with batch size 16)

**Why These Optimizations Work:**
- Adapter layers only train 0.95-3.6% of parameters
- Base model frozen → no gradients needed
- Memory abundant → no need for gradient checkpointing
- Larger batch size → better GPU utilization
- More workers + prefetching → less GPU idle time

### 5. Evaluation Module

**Metrics Calculation:**
```python
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    
    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    bleu = metrics_calculator.compute_bleu(decoded_preds, decoded_labels)
    rouge = metrics_calculator.compute_rouge(decoded_preds, decoded_labels)
    
    return {
        "bleu_4": bleu["bleu"],
        "rouge_l": rouge["rougeL"]
    }
```

## Data Models

### Dataset Entry Format

**Input Format:**
```json
{
  "input": "buat_soal_pilihan_ganda: [CONTEXT]",
  "target": "question: [Q]\nanswer: [A]\ndistractors: [D1] | [D2] | [D3]",
  "metadata": {
    "format": "MCQ",
    "difficulty": "medium"
  }
}
```

**Note:** System supports both `target` (v2) and `output` (v3) field names.

### Training State

```python
{
    "epoch": int,
    "global_step": int,
    "training_loss": float,
    "eval_bleu_4": float,
    "eval_rouge_l": float,
    "learning_rate": float,
    "gpu_memory_allocated": float
}
```

## Error Handling

### 1. Out of Memory (OOM)

**Detection:** `torch.cuda.OutOfMemoryError`

**Recovery:**
```python
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("⚠️ GPU OOM detected!")
    print("Suggestions:")
    print("  1. Reduce batch size to 2")
    print("  2. Increase gradient accumulation to 4")
    print("  3. Disable gradient checkpointing if enabled")
    torch.cuda.empty_cache()
```

### 2. Adapter Library Not Found

**Detection:** `ImportError: No module named 'adapters'`

**Recovery:**
```python
try:
    import adapters
    from adapters import AutoAdapterModel
except ImportError:
    print("Installing NEW adapters library...")
    !pip install -q adapters  # NOT adapter-transformers!
    import adapters
    from adapters import AutoAdapterModel
```

**IMPORTANT:** 
- Install `adapters` (NEW library)
- NOT `adapter-transformers` (deprecated, causes errors)

### 3. Model Loading State Dict Error (RESOLVED - Library Migration)

**Detection:** `ValueError: The state dictionary of the model you are trying to load is corrupted`

**Root Cause:** Using DEPRECATED `adapter-transformers` library which has compatibility issues with newer transformers versions.

**Solution:** Migrate to NEW `adapters` library

```python
# WRONG (old, deprecated library)
!pip install adapter-transformers
from adapters import AutoAdapterModel  # This causes state dict errors

# CORRECT (new library)
!pip install adapters
import adapters
from adapters import AutoAdapterModel

# Method 1: Direct loading
model = AutoAdapterModel.from_pretrained(model_name)

# Method 2: If Method 1 fails
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
adapters.init(model)  # Initialize adapter support
```

**Why This Works:**
- `adapters` is the official successor to `adapter-transformers`
- No state dict compatibility issues
- Cleaner API, no workaround flags needed
- 100% backward compatible with adapter weights
- Actively maintained and updated

**Status:** ✅ Fixed by library migration

### 4. Dataset Loading Failure

**Detection:** `FileNotFoundError`

**Recovery:**
```python
if not os.path.exists(TASK_DIR + 'train.jsonl'):
    print("Dataset not found. Copying from Drive...")
    # Copy from Drive
    shutil.copy(f'{DRIVE_ROOT}/dataset-task-spesifc/train.jsonl', TASK_DIR)
```

## Performance Optimization

### Memory Optimization Techniques

1. **Gradient Checkpointing** (~30% memory savings)
   - Trade computation for memory
   - Recompute activations during backward pass

2. **Mixed Precision (FP16)** (~50% memory savings)
   - Use float16 for forward/backward
   - Use float32 for optimizer updates

3. **Gradient Accumulation** (enables larger effective batch size)
   - Accumulate gradients over multiple steps
   - Update weights less frequently

4. **Adapter Layers** (~10x memory savings vs full fine-tuning)
   - Train only 3.6% of parameters
   - Freeze base model weights

### Training Speed Optimization

1. **DataLoader Configuration:**
   ```python
   dataloader_num_workers=2
   dataloader_pin_memory=True
   ```

2. **Batch Size Tuning:**
   - Start with batch_size=4
   - Monitor GPU utilization
   - Increase if < 80% utilization

3. **Gradient Accumulation:**
   - Effective batch size = batch_size × accumulation_steps
   - Target effective batch size: 8-16

## Testing Strategy

### Unit Tests

1. **Adapter Setup Test:**
   - Verify adapter added successfully
   - Check trainable parameters ~3.6%
   - Verify base model frozen

2. **Model Loading Robustness Test:**
   - Test primary loading path with compatibility flags
   - Test fallback loading path with explicit config
   - Verify both paths produce valid model
   - Test error handling for corrupted checkpoints

3. **Dataset Loading Test:**
   - Test with v2 format (`target` field)
   - Test with v3 format (`output` field)
   - Verify backward compatibility

4. **Memory Test:**
   - Load model + adapter
   - Run forward pass
   - Verify memory < 14GB

### Integration Tests

1. **End-to-End Training (1 epoch):**
   - Load model with adapter
   - Load dataset
   - Train for 1 epoch
   - Verify checkpoint saved
   - Verify metrics logged

2. **Evaluation Pipeline:**
   - Load trained adapter
   - Evaluate on test set
   - Generate sample outputs
   - Verify metrics computed

## Comparison: Adapter vs LoRA

| Aspect | LoRA (v2) | Adapter (v3) |
|--------|-----------|--------------|
| **Trainable Params** | 0.36% (~0.9M) | 3.6% (~8.9M) |
| **Memory Usage** | 8-10GB | 12-14GB |
| **Training Time** | 4-6 hours | 6-8 hours |
| **Performance** | Near full FT | 99.6% of full FT |
| **Inference Latency** | +5-10ms | No overhead |
| **Stability** | Good | Excellent |
| **Best For** | Large models (>1B) | Small models (<1B) |

**Conclusion:** Adapter layers lebih cocok untuk IndoNanoT5 (248M) dengan small dataset (1500 samples).

## Dependencies

```python
# Core libraries
adapters>=1.0.0  # NEW library (replaces adapter-transformers)
transformers>=4.35.0
datasets>=2.15.0
torch>=2.1.0

# Evaluation
evaluate>=0.4.0
rouge_score>=0.1.2
bert_score>=0.3.13

# Utilities
accelerate>=0.25.0
matplotlib>=3.8.0
```

**IMPORTANT - Library Migration:**
- ❌ DO NOT use `adapter-transformers` (deprecated, causes state dict errors)
- ✅ USE `adapters` (official successor, fully compatible)
- Migration guide: https://docs.adapterhub.ml/transitioning.html
- All adapter weights from old library work with new library

## Configuration Files

### adapter_config.json
```json
{
  "adapter_name": "mcq_generation",
  "config": "pfeiffer",
  "reduction_factor": 12,
  "non_linearity": "relu",
  "adapter_residual_before_ln": true,
  "ln_before": false,
  "ln_after": false
}
```

### training_config.json
```json
{
  "model_name": "LazarusNLP/IndoNanoT5-base",
  "adapter_config": "pfeiffer",
  "reduction_factor": 12,
  "num_train_epochs": 8,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "learning_rate": 1e-4,
  "warmup_steps": 50,
  "weight_decay": 0.01,
  "fp16": true,
  "gradient_checkpointing": true
}
```

## Deployment Considerations

### Model Export

**Adapter-only export** (~5MB):
```python
model.save_adapter("./adapter_weights", "mcq_generation")
```

**Full model export** (~1GB):
```python
model.save_pretrained("./full_model_with_adapter")
```

### Inference Pipeline

```python
from adapters import AutoAdapterModel

# Load base model
model = AutoAdapterModel.from_pretrained("LazarusNLP/IndoNanoT5-base")

# Load adapter
model.load_adapter("./adapter_weights")
model.set_active_adapters("mcq_generation")

# Generate
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, num_beams=4)
```

## Expected Results

### Performance Metrics

**Target Metrics (after 8 epochs):**
- BLEU-4: 0.20-0.28
- ROUGE-L: 0.25-0.35
- BERTScore F1: 0.75-0.85

**Training Metrics:**
- Training loss: 39 → 2-5
- Validation loss: ~2-3
- Training time: 6-8 hours (T4 GPU)
- Memory usage: 12-14GB peak

### Comparison with v2 (LoRA)

**Expected improvements:**
- Performance: +5-10% BLEU-4
- Stability: More stable training
- Inference: No additional latency

**Trade-offs:**
- Memory: +2-4GB
- Training time: +2 hours
- Model size: +3MB adapter weights

## Warning Analysis & Resolution (April 2026)

### Overview

Setelah implementasi dan testing, kami mengidentifikasi 5 warnings yang muncul saat training. Berikut adalah analisis lengkap dan resolusi untuk setiap warning.

### Warning 1: num_items_in_batch TypeError ✅ FIXED

**Symptom:**
```
TypeError: T5ForConditionalGeneration.forward() got an unexpected keyword argument 'num_items_in_batch'
```

**Root Cause:**
- Transformers 4.46+ introduced `num_items_in_batch` parameter in Trainer
- Adapters library's model wrapper doesn't accept this parameter
- Known compatibility issue between `adapters` 1.3.0 and `transformers` 4.57.6

**Resolution:**
Created `CompatibleSeq2SeqTrainer` class in `adapter_trainer.py`:

```python
class CompatibleSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that handles num_items_in_batch parameter compatibility.
    
    Fixes compatibility issue between transformers 4.46+ and adapters library.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Call parent's compute_loss WITHOUT num_items_in_batch parameter
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
```

**Status:** ✅ RESOLVED
**File:** `src/finetuned/training/adapter_trainer.py`
**Reference:** https://discuss.huggingface.co/t/typeerror-sentencetransformertrainer-compute-loss-got-an-unexpected-keyword-argument-num-items-in-batch/114298/3

### Warning 2: use_cache Incompatible with Gradient Checkpointing ✅ NORMAL

**Warning:**
```
WARNING:adapters.models.t5.modeling_t5:`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
```

**Analysis:**
- This is NOT an error, it's an **INFORMATIONAL WARNING**
- Gradient checkpointing and use_cache have conflicting memory strategies
- Adapters library **AUTOMATICALLY** detects and resolves the conflict
- Training proceeds normally after this warning

**Technical Explanation:**
- **Gradient Checkpointing:** Discards activations to save memory, recomputes during backward pass
- **use_cache:** Stores key-value pairs from attention layers for faster generation
- **Conflict:** One wants to discard, the other wants to store
- **Resolution:** Library automatically disables use_cache during training

**Status:** ✅ NORMAL - NO ACTION REQUIRED
**Reference:** https://discuss.huggingface.co/t/why-is-use-cache-incompatible-with-gradient-checkpointing/18811

### Warning 3: past_key_value Deprecated ℹ️ IGNORE

**Warning:**
```
FutureWarning: `past_key_value` is deprecated and will be removed in version 4.58 for `T5Block.forward`. Use `past_key_values` instead.
```

**Analysis:**
- Deprecation warning from **INSIDE** transformers library (not our code)
- Does NOT affect training functionality
- Will be fixed by HuggingFace team in transformers 4.58+

**Status:** ℹ️ INFORMATIONAL - SAFE TO IGNORE

### Warning 4: top_p Generation Flag ✅ FIXED

**Warning:**
```
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
```

**Root Cause:**
- `top_p` (nucleus sampling) only valid when `do_sample=True`
- We use `num_beams=4` (beam search)
- Beam search doesn't use sampling parameters

**Resolution:**
Updated `model_evaluator.py` to conditionally set generation parameters:

```python
def generate_prediction(self, input_text, num_beams=4, do_sample=False, ...):
    gen_kwargs = {
        'max_length': max_length,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
    }
    
    if do_sample:
        # Sampling mode: use temperature, top_k, top_p
        gen_kwargs.update({
            'do_sample': True,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
        })
    else:
        # Beam search mode: don't use sampling parameters
        gen_kwargs.update({
            'num_beams': num_beams,
            'do_sample': False,
        })
    
    outputs = self.model.generate(**inputs, **gen_kwargs)
```

**Status:** ✅ RESOLVED
**File:** `src/finetuned/evaluation/model_evaluator.py`

### Warning 5: Parameter Function Hashing ℹ️ IGNORE

**Warning:**
```
Parameter 'function'=<function AdapterTrainer.preprocess_dataset.<locals>.preprocess_function at 0x...> couldn't be hashed properly
```

**Analysis:**
- HuggingFace datasets tries to cache preprocessing results
- Nested functions can't be hashed with pickle
- Dataset will be reprocessed each time (no caching)
- **Impact:** ~1-2 seconds preprocessing (0.01% of 6-8 hour training)

**Status:** ℹ️ INFORMATIONAL - MINIMAL IMPACT, SAFE TO IGNORE

### Summary Table

| Warning | Severity | Status | Action Required |
|---------|----------|--------|-----------------|
| num_items_in_batch TypeError | ❌ CRITICAL | ✅ FIXED | Re-extract code |
| use_cache incompatible | ⚠️ INFO | ✅ NORMAL | None - auto-handled |
| past_key_value deprecated | ⚠️ INFO | ℹ️ IGNORE | None - library issue |
| top_p generation flag | ⚠️ INFO | ✅ FIXED | Re-extract code |
| Parameter function hashing | ⚠️ INFO | ℹ️ IGNORE | None - minimal impact |

**Overall Status:** ✅ READY TO TRAIN

**Documentation:**
- Detailed analysis: `docs/warning-analysis.md`
- Quick reference: `docs/error.md`
- Summary: `docs/SUMMARY-FIXES.md`

---

## Known Issues and Resolutions

### Issue 1: Model Loading State Dict Error (RESOLVED - Library Migration)

**Symptom:** `ValueError: The state dictionary of the model you are trying to load is corrupted`

**Affected Component:** `src/finetuned/utils/adapter_loader.py`

**Root Cause:** 
- Using DEPRECATED `adapter-transformers` library (v1.3.0)
- This library has been REPLACED by new `adapters` library
- Old library has compatibility issues with transformers 4.57.6
- Internal validation in `_get_key_renaming_mapping()` fails

**Resolution (Library Migration):**
1. **Migrate to NEW Library:**
   - Uninstall: `pip uninstall adapter-transformers`
   - Install: `pip install adapters`
   
2. **Update Imports:**
   ```python
   # Add this import
   import adapters
   
   # Keep these (same namespace)
   from adapters import AutoAdapterModel, AdapterConfig
   ```

3. **Simplified Loading:**
   - Method 1: `AutoAdapterModel.from_pretrained(model_name)`
   - Method 2: Load with transformers + `adapters.init(model)`
   - No workaround flags needed

**Code Location:** `src/finetuned/utils/adapter_loader.py` (rewritten)

**Status:** ✅ Fixed by migrating to official `adapters` library

**Documentation:** 
- New library docs: https://docs.adapterhub.ml/
- Migration guide: https://docs.adapterhub.ml/transitioning.html
- Error details: `docs/error.md`

**Key Benefits:**
- No state dict errors
- Cleaner API
- Actively maintained
- 100% backward compatible with adapter weights
- Better transformers version compatibility

### Issue 2: Notebook Cell Execution Order

**Symptom:** Cells fail if executed out of order

**Resolution:**
- All cells numbered sequentially
- Dependencies clearly documented
- Each cell checks prerequisites
- Clear error messages if prerequisites missing

**Status:** ✅ Documented in notebook

### Issue 3: Google Drive Disconnection

**Symptom:** Training interrupted if Drive disconnects

**Resolution:**
- Checkpoints saved every epoch
- Can resume from last checkpoint
- Training state preserved in checkpoint

**Status:** ✅ Handled by checkpoint system

## Troubleshooting Guide

### Model Loading Fails

**Symptoms:**
- ValueError about corrupted state dict
- ImportError for adapters module

**Solutions:**
1. **CRITICAL:** Ensure you're using NEW `adapters` library:
   ```bash
   pip uninstall adapter-transformers  # Remove old library
   pip install adapters  # Install new library
   ```

2. Update imports in your code:
   ```python
   import adapters  # Add this
   from adapters import AutoAdapterModel, AdapterConfig
   ```

3. Verify internet connection for model download

4. If persistent, try clearing cache: `!rm -rf ~/.cache/huggingface/`

**Common Mistake:** Installing `adapter-transformers` instead of `adapters`
- `adapter-transformers` is DEPRECATED and causes state dict errors
- `adapters` is the official successor

### Out of Memory (OOM)

**Symptoms:**
- `torch.cuda.OutOfMemoryError`
- Training crashes during forward/backward pass

**Solutions:**
1. Reduce batch size to 2: `per_device_train_batch_size=2`
2. Increase gradient accumulation to 4: `gradient_accumulation_steps=4`
3. Disable gradient checkpointing (trades memory for speed)
4. Restart runtime to clear GPU memory

### Training Diverges

**Symptoms:**
- Loss increases instead of decreases
- NaN values in loss

**Solutions:**
1. Reduce learning rate to 5e-5
2. Increase warmup steps to 100
3. Check for data quality issues
4. Verify tokenization is correct

### Slow Training

**Symptoms:**
- Training takes > 10 hours
- Low GPU utilization

**Solutions:**
1. Verify T4 GPU selected (not CPU)
2. Check if FP16 enabled
3. Increase batch size if memory allows
4. Verify dataloader workers set to 2

## Version History

### v3.0 (Current)
- ✅ Adapter Layers implementation
- ✅ **MIGRATED to NEW `adapters` library** (from deprecated `adapter-transformers`)
- ✅ Fixed state dict loading errors via library migration
- ✅ Simplified model loading (no workaround flags needed)
- ✅ 8 epochs training
- ✅ Comprehensive error handling
- ✅ Memory optimization for T4 GPU

### v2.0 (Previous)
- LoRA implementation
- 3 epochs training
- Basic error handling

### v1.0 (Initial)
- Full fine-tuning
- High memory requirements
- Not suitable for T4 GPU

