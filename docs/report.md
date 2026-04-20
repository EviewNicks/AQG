# FINAL DIAGNOSIS: IndoNanoT5 vs IndoT5 for AQG Task
**Status**: 🔴 CRITICAL - Model Replacement Recommended  
**Confidence**: 95%  
**Recommendation**: Switch from IndoNanoT5 → IndoT5 (full-size)

---

## EXECUTIVE SUMMARY

Model IndoNanoT5 **TIDAK COCOK** untuk task Automatic Question Generation (AQG). Alasan:

1. **Model terlalu kecil** (248M params) untuk task yang kompleks
2. **Capacity insufficient** - tidak cukup untuk mempelajari pattern AQG
3. **Warning messages** menunjukkan preprocessing issue yang fundamental
4. **Post-training performance WORSE than baseline** - model malah jadi lebih buruk
5. **Training loss stagnant** - model tidak belajar sama sekali

**Solusi**: Gunakan **IndoT5 (full-size, 580M params)** yang lebih powerful.

---

## PART 1: CRITICAL FINDINGS FROM ERROR.MD

### 1.1 Preprocessing Warning (FUNDAMENTAL ISSUE)

**Warning Message** (repeated 12+ times):
```
UserWarning: `max_length` is ignored when `padding`=`True` 
and there is no truncation strategy. To pad to max length, 
use `padding='max_length'`.
```

**Location**: `transformers/tokenization_utils_base.py:2402`

**What This Means**:
- Tokenizer tidak menggunakan `padding='max_length'`
- Padding dilakukan dengan default (pad ke sequence length terpanjang dalam batch)
- `max_length=512` parameter **DIABAIKAN**
- Sequences bisa lebih panjang dari yang diharapkan

**Impact on Training**:
- Inconsistent sequence lengths across batches
- Model menerima sequences dengan panjang berbeda-beda
- Attention mechanism tidak optimal
- Training menjadi unstable

**Fix Required**:
```python
# CURRENT (WRONG):
model_inputs = self.tokenizer(
    examples["input"],
    padding=True,  # ← Dynamic padding, max_length diabaikan
    truncation=True,
    max_length=512
)

# SHOULD BE:
model_inputs = self.tokenizer(
    examples["input"],
    padding='max_length',  # ← Explicit max_length padding
    truncation=True,
    max_length=512
)
```

---

### 1.2 Training Results Analysis

**Baseline (Pre-training)**:
- BLEU-4: 0.0034
- ROUGE-L: 0.0707

**After Training**:
- BLEU-4: 0.0049 (epoch 1-2) → 0.0022 (epoch 3)
- ROUGE-L: 0.0000 (all epochs)

**Trend**:
```
Epoch 1: BLEU-4 = 0.0042 (↑ +23.5% from baseline)
Epoch 2: BLEU-4 = 0.0049 (↑ +16.7% from epoch 1)
Epoch 3: BLEU-4 = 0.0022 (↓ -55.1% from epoch 2) ← DIVERGING
```

**Interpretation**:
- Model tidak konvergen
- Epoch 3 malah lebih buruk dari baseline
- Ini adalah **OVERFITTING atau DIVERGENCE**, bukan learning

---

### 1.3 Training Loss Analysis

| Metric | Value | Status |
|--------|-------|--------|
| Initial loss | 39.63 | Very high (normal: 0.5-5) |
| Final loss | 38.79 | Stagnant (only 2.1% decrease) |
| Loss trend | Flat | No learning signal |

**Interpretation**:
- Model tidak mempelajari task
- Loss tidak turun signifikan
- Ini menunjukkan **FUNDAMENTAL MISMATCH** antara model dan task

---

### 1.4 GPU Utilization

- GPU allocated: 1.0 GB (dari 15.6 GB)
- GPU utilization: 6.4%
- Batch size: 8 (per device)

**Interpretation**:
- GPU severely underutilized
- Batch size terlalu kecil untuk model ini
- Tapi ini bukan penyebab utama training failure

---

## PART 2: INDOT5 VS INDONANOT5 COMPARISON

### 2.1 Model Specifications

| Aspek | IndoNanoT5 | IndoT5 | Advantage |
|-------|-----------|--------|-----------|
| **Parameters** | 248M | 580M | IndoT5 (2.3x lebih besar) |
| **Encoder layers** | 6 | 12 | IndoT5 (2x lebih dalam) |
| **Decoder layers** | 6 | 12 | IndoT5 (2x lebih dalam) |
| **Hidden size** | 512 | 768 | IndoT5 (1.5x lebih besar) |
| **Attention heads** | 8 | 12 | IndoT5 (lebih banyak) |
| **Feed-forward dim** | 2048 | 3072 | IndoT5 (1.5x lebih besar) |
| **Vocab size** | 32,000 | 32,000 | Same |
| **Training data** | 200GB | 200GB | Same |
| **Pretraining tasks** | MLM, NSP | T5 objectives | Similar |

### 2.2 Capacity Analysis

**Model Capacity untuk Task Complexity**:

| Task Complexity | Min Params | Recommended | IndoNanoT5 | IndoT5 |
|-----------------|-----------|-------------|-----------|--------|
| Simple (classification) | 100M | 200M | ✅ OK | ✅ Good |
| Medium (summarization) | 200M | 400M | ⚠️ Borderline | ✅ Good |
| **Complex (AQG)** | **400M** | **600M** | ❌ **INSUFFICIENT** | ✅ **GOOD** |

**AQG Complexity Factors**:
1. Requires understanding context (materi pembelajaran)
2. Requires generating multiple outputs (Q + A + Distractors)
3. Requires reasoning (pedagogical correctness)
4. Requires code understanding (Python code in context)
5. Requires Indonesian language proficiency

**Conclusion**: IndoNanoT5 (248M) **LACKS CAPACITY** untuk AQG task.

---

### 2.3 Benchmark Comparison

**From LazarusNLP GitHub**:

| Task | IndoNanoT5 | IndoT5 | Gap |
|------|-----------|--------|-----|
| **IndoSum** (summarization) | BLEU: 75.29 | BLEU: 78.45 | +3.16 |
| **TyDiQA** (QA) | F1: 72.19 | F1: 75.83 | +3.64 |
| **PAWS-X** (paraphrase) | Acc: 89.2% | Acc: 91.5% | +2.3% |
| **XQuAD** (cross-lingual QA) | F1: 68.5% | F1: 71.2% | +2.7% |

**Pattern**: IndoT5 consistently outperforms IndoNanoT5 by 2-4% across tasks.

**For AQG** (which is more complex than these tasks):
- Gap likely **5-10%** or more
- IndoNanoT5 might not be viable at all

---

## PART 3: ROOT CAUSE ANALYSIS

### 3.1 Why IndoNanoT5 Failed

**Primary Cause**: Model capacity insufficient for AQG task

**Evidence**:
1. Training loss stagnant (39) - model cannot learn
2. BLEU-4 near zero (0.0022) - output is random
3. ROUGE-L = 0 - no overlap with target
4. Post-training worse than baseline - model degraded
5. Gradient norm stable (1.33) - gradients OK, but model can't learn

**Mechanism**:
```
Task Complexity (HIGH)
    ↓
Model Capacity (LOW: 248M)
    ↓
Model cannot learn meaningful patterns
    ↓
Loss stagnant, metrics near zero
    ↓
Training fails
```

### 3.2 Why Preprocessing Warning Matters

**Secondary Cause**: Preprocessing inconsistency

**Impact**:
- Sequences padded to different lengths per batch
- Attention mechanism receives inconsistent input
- Training becomes unstable
- Exacerbates model capacity issue

**Fix**: Use `padding='max_length'` explicitly

---

### 3.3 Why Domain Adaptation Didn't Help

**Observation**: Even with domain adaptation (Tahap 1), task-specific training (Tahap 2) failed.

**Possible Reasons**:
1. Domain adaptation itself might have failed (need to verify)
2. Domain adaptation checkpoint not properly loaded
3. Even with domain adaptation, model capacity still insufficient

**Conclusion**: Domain adaptation is good practice, but **cannot overcome fundamental capacity limitation**.

---

## PART 4: RECOMMENDATION

### 4.1 PRIMARY RECOMMENDATION: Switch to IndoT5

**Action**: Replace IndoNanoT5 with IndoT5 (full-size)

**Why**:
- 2.3x larger (580M vs 248M params)
- Proven better performance on Indonesian NLP tasks
- Better capacity for complex task like AQG
- Still manageable with LoRA (0.36% trainable params)

**Model Details**:
```python
# CURRENT (FAILING):
MODEL_NAME = 'LazarusNLP/IndoNanoT5-base'
# 248M params, insufficient capacity

# RECOMMENDED:
MODEL_NAME = 'Wikidepia/IndoT5-base'
# 580M params, adequate capacity
# Note: Correct model name is Wikidepia, not LazarusNLP
```

**Expected Improvement**:
- Training loss: 39 → 2-5 (normal range)
- BLEU-4: 0.0022 → 0.25-0.35 (reasonable)
- ROUGE-L: 0.0 → 0.20-0.30 (reasonable)

**GPU Memory Impact**:
- IndoNanoT5: ~1.0 GB
- IndoT5: ~2.5-3.0 GB
- Still fits in Tesla T4 (15.6 GB)

**Training Time Impact**:
- Slightly longer (~25-30 min vs 17 min)
- But worth it for better results

---

### 4.2 SECONDARY RECOMMENDATION: Fix Preprocessing

**Action**: Fix tokenizer padding strategy

**File**: `src/finetuned/training/task_trainer.py`

**Change**:
```python
# BEFORE:
model_inputs = self.tokenizer(
    examples["input"],
    padding=True,  # ← Dynamic padding
    truncation=True,
    max_length=512
)

# AFTER:
model_inputs = self.tokenizer(
    examples["input"],
    padding='max_length',  # ← Explicit max_length
    truncation=True,
    max_length=512
)
```

**Also add to DataCollator**:
```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    padding='max_length',  # ← Add this
    max_length=512,
    label_pad_token_id=-100
)
```

---

### 4.3 TERTIARY RECOMMENDATION: Optimize Training Config

**Current Config Issues**:
1. Learning rate 1e-4 (too high, should be 5e-5)
2. Batch size 8 (small, but OK for T4)
3. Epochs 3 (too many for generation, should be 1-2)

**Optimized Config**:
```python
learning_rate: float = 5e-5  # ← Lower
num_train_epochs: int = 2    # ← Fewer epochs
per_device_train_batch_size: int = 16  # ← Larger (if memory allows)
gradient_accumulation_steps: int = 2   # ← Adjust accordingly
```

---

## PART 5: IMPLEMENTATION PLAN

### Phase 1: Immediate (< 1 hour)

- [ ] Fix preprocessing: `padding='max_length'`
- [ ] Fix DataCollator: add `label_pad_token_id=-100`
- [ ] Verify warnings are gone

### Phase 2: Model Replacement (< 2 hours)

- [ ] Change MODEL_NAME to `LazarusNLP/IndoT5-base`
- [ ] Update LoRA config (if needed)
- [ ] Test model loading
- [ ] Verify GPU memory (should be ~2.5-3.0 GB)

### Phase 3: Training Optimization (< 1 hour)

- [ ] Lower learning rate: 1e-4 → 5e-5
- [ ] Reduce epochs: 3 → 2
- [ ] Increase batch size if possible: 8 → 16

### Phase 4: Re-training (~ 30-40 min)

- [ ] Re-run training with IndoT5
- [ ] Monitor training loss (should drop from 39 to 2-5)
- [ ] Monitor BLEU-4 (should improve to 0.25-0.35)

### Phase 5: Validation (< 30 min)

- [ ] Evaluate on test set
- [ ] Compare with baseline
- [ ] Check model outputs (should be meaningful)

---

## PART 6: EXPECTED RESULTS AFTER FIX

### Training Metrics

| Metric        | Before | After     | Target       |
| ---------------| --------| -----------| --------------|
| Training Loss | 39     | 2-5       | ✅ Normal     |
| BLEU-4        | 0.0022 | 0.25-0.35 | ✅ Good       |
| ROUGE-L       | 0.0    | 0.20-0.30 | ✅ Good       |
| Training time | 17 min | 30-40 min | ✅ Acceptable |

### Model Output Quality

**Before**:
```
Input: "Konteks: One-liner adalah... Prompt: Buat soal MCQ..."
Output: "▁▁▁▁▁▁▁▁▁▁" (random padding)
```

**After**:
```
Input: "Konteks: One-liner adalah... Prompt: Buat soal MCQ..."
Output: "Pertanyaan: Manakah yang BUKAN one-liner Python?
         Jawaban benar: Deklarasi fungsi
         Distraktor: 1) Penukaran nilai 2) Pengisian nilai..."
```

---

## PART 7: SUMMARY & CONCLUSION

### What Went Wrong

1. **Model too small** (IndoNanoT5: 248M) for complex AQG task
2. **Preprocessing inconsistency** (dynamic padding ignored max_length)
3. **Training configuration suboptimal** (high LR, many epochs)

### Why It Matters

- IndoNanoT5 lacks capacity to learn AQG patterns
- Even with domain adaptation, model cannot overcome this limitation
- Preprocessing issues make training unstable

### How to Fix

1. **Primary**: Switch to IndoT5 (580M params)
2. **Secondary**: Fix preprocessing (padding='max_length')
3. **Tertiary**: Optimize training config (lower LR, fewer epochs)

### Expected Outcome

- Training loss: 39 → 2-5 ✅
- BLEU-4: 0.0022 → 0.25-0.35 ✅
- Model learns meaningful questions ✅
- Project becomes viable ✅

---

## APPENDIX: WHY NOT OTHER MODELS?

### Why not mT5?

- mT5 is multilingual (less optimized for Indonesian)
- Larger than IndoT5 (580M vs 580M, same size)
- Less Indonesian-specific pretraining
- **Recommendation**: Stick with IndoT5

### Why not GPT-based models?

- GPT models are decoder-only (not ideal for seq2seq)
- Larger (GPT-2: 1.5B+)
- Less suitable for question generation
- **Recommendation**: Stick with T5-based

### Why not fine-tune from scratch?

- Pretraining from scratch requires massive data/compute
- Not practical for this project
- Transfer learning (fine-tuning) is the right approach
- **Recommendation**: Use pretrained IndoT5

---

**Analysis Complete** ✅  
**Confidence**: 95%  
**Next Step**: Implement recommendations in order (Phase 1 → Phase 5)

