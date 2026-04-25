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
- Adapter layers: ~8.9M parameters (trainable)
- Trainable percentage: 3.6%
- Memory footprint: ~12-14GB (vs ~32GB full fine-tuning)

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
    
    Returns:
        (model_with_adapter, tokenizer)
    """
```

**Implementation Steps:**
1. Load base model using `AutoAdapterModel`
2. Add adapter dengan Pfeiffer config
3. Activate adapter untuk training
4. Freeze base model parameters
5. Move to GPU dan enable gradient checkpointing

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

### 4. Training Configuration

**Seq2SeqTrainingArguments:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/adapter",
    num_train_epochs=8,  # User request
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01,
    
    # Memory optimization
    gradient_checkpointing=True,
    fp16=True,
    
    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu_4",
    greater_is_better=True,
    
    # Logging
    logging_steps=10,
    report_to=["none"],
    
    # Generation
    predict_with_generate=True,
    generation_max_length=512,
    
    # Checkpointing
    save_total_limit=2,
)
```

**Expected Training Metrics:**
- Training time: 6-8 hours (T4 GPU)
- Memory usage: 12-14GB peak
- Training loss: 39 → 2-5
- BLEU-4: 0.005 → 0.20-0.28
- ROUGE-L: 0.0 → 0.25-0.35

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
    from adapters import AutoAdapterModel
except ImportError:
    print("Installing adapter-transformers...")
    !pip install -q adapter-transformers
    from adapters import AutoAdapterModel
```

### 3. Dataset Loading Failure

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

2. **Dataset Loading Test:**
   - Test with v2 format (`target` field)
   - Test with v3 format (`output` field)
   - Verify backward compatibility

3. **Memory Test:**
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
adapter-transformers>=3.2.0  # Adapter layers support
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

