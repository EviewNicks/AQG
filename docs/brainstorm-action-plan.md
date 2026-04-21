# BRAINSTORM: Action Plan untuk Proyek AQG IndoT5

**Tanggal:** 20 April 2026  
**Status:** Critical Decision Point  
**Tujuan:** Menentukan langkah terbaik berdasarkan analisis komprehensif

---

## 🎯 SITUASI SAAT INI

### Yang Sudah Benar ✅
1. **Preprocessing pipeline**: Tokenization 100% valid
2. **DataCollator fix**: Sudah diidentifikasi dan diperbaiki
3. **Model selection**: IndoT5-base (297M) cocok untuk task
4. **Dataset size**: 1,262 samples (cukup untuk initial training)
5. **Infrastructure**: GPU ready, training pipeline ready

### Masalah Kritis ❌
1. **Training loss = 0.0** → Model tidak belajar
2. **Eval loss = NaN** → Numerical instability
3. **BLEU-4 turun** setelah training (-7.6%)
4. **Format dataset** tidak sesuai HuggingFace standard

### Root Cause Analysis
```
HYPOTHESIS 1: DataCollator Bug
Status: ✅ FIXED (sudah diidentifikasi)
Evidence: Forward pass menghasilkan loss 9.9250 (valid)

HYPOTHESIS 2: Dataset Format Issue
Status: ⚠️ HIGHLY LIKELY
Evidence: 
- Input mengandung prompt instruction
- Target mengandung struktur kompleks
- Tidak sesuai dengan 7 literatur
- Tidak sesuai HuggingFace standard
```

---

## 🤔 CRITICAL QUESTIONS

### Q1: Apakah DataCollator Fix Sudah Di-Apply di Training?

**Dari training_results.json:**
```json
{
  "train_loss": 0.0,
  "eval_loss": NaN
}
```

**Analisis:**
- ❌ Loss masih 0.0 → Fix belum di-apply ATAU ada masalah lain
- ❌ Eval loss masih NaN → Masalah masih ada

**KESIMPULAN:** Fix DataCollator mungkin belum di-apply di training yang tercatat di report, ATAU ada masalah format dataset yang lebih fundamental.

---

### Q2: Apakah Format Dataset Benar-Benar Masalah?

**Evidence PRO (Format adalah masalah):**
1. ✅ 7 literatur SEMUA menggunakan format sederhana (input → target)
2. ✅ HuggingFace standard TIDAK menggunakan prefix/instruction dalam input
3. ✅ Model output "Perbandingan Penggunaan Memori..." (copy input) bukan "Pertanyaan..."
4. ✅ Training loss 0.0 menunjukkan model tidak belajar mapping yang benar

**Evidence CONTRA (Format bukan masalah):**
1. ⚠️ Forward pass menghasilkan loss valid (9.9250)
2. ⚠️ Tokenization 100% benar
3. ⚠️ DataCollator 96.6% valid

**KESIMPULAN:** Format dataset SANGAT MUNGKIN adalah masalah utama, TAPI perlu verify apakah DataCollator fix sudah di-apply.

---

### Q3: Apa yang Harus Dilakukan Pertama?

**OPSI A: Re-run Training dengan DataCollator Fix**
- Waktu: 10-15 menit
- Risiko: Low
- Benefit: Verify apakah fix sudah cukup
- Rekomendasi: ✅ **DO THIS FIRST**

**OPSI B: Transform Dataset Format**
- Waktu: 2-3 jam
- Risiko: Low-Medium
- Benefit: Align dengan standard
- Rekomendasi: ⚠️ **DO THIS IF OPSI A FAILS**

**OPSI C: Hyperparameter Tuning**
- Waktu: Variable
- Risiko: Medium
- Benefit: Marginal improvement
- Rekomendasi: ❌ **NOT YET** (fix fundamental issues first)

---

## 📊 DECISION TREE

```
START
  │
  ├─ Apakah DataCollator fix sudah di-apply?
  │   │
  │   ├─ TIDAK → Re-run training dengan fix
  │   │           │
  │   │           ├─ Loss > 0.0? → ✅ SOLVED
  │   │           │
  │   │           └─ Loss = 0.0? → Lanjut ke format transformation
  │   │
  │   └─ YA → Lanjut ke format transformation
  │
  ├─ Transform dataset format (Approach 2)
  │   │
  │   ├─ Re-run training
  │   │   │
  │   │   ├─ Loss > 0.0 & Metrics improve? → ✅ SOLVED
  │   │   │
  │   │   └─ Still issues? → Try Approach 3 (Structured Output)
  │   │
  │   └─ If still fails → Deep dive into model/data compatibility
  │
  └─ Hyperparameter tuning & data augmentation
```

---

## 🎬 RECOMMENDED ACTION PLAN

### PHASE 1: Immediate Verification (30 menit)

#### Step 1.1: Verify DataCollator Fix
```python
# Check if fix is applied in current training code
# File: src/finetuned/training/task_trainer.py

# CORRECT (should be):
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    label_pad_token_id=-100,
    padding=True,
    pad_to_multiple_of=8
    # NO max_length parameter!
)

# WRONG (should NOT be):
data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.tokenizer,
    model=self.model,
    max_length=self.max_length  # ❌ This causes bug!
)
```

**Action:** Buka `src/finetuned/training/task_trainer.py` dan verify line 245-252

#### Step 1.2: Quick Re-run Test
```bash
# Re-run training dengan 1 epoch untuk quick test
python train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --logging_steps 10

# Check loss progression
# Expected: loss > 0.0 dan decreasing
```

**Decision Point:**
- ✅ Loss > 0.0 → DataCollator fix berhasil, lanjut optimize
- ❌ Loss = 0.0 → Lanjut ke Phase 2 (Format Transformation)

---

### PHASE 2: Format Transformation (2-3 jam)

#### Step 2.1: Backup Original Dataset
```bash
cd dataset_aqg/dataset-task-spesifc/
cp train.jsonl train_original.jsonl
cp validation.jsonl validation_original.jsonl
cp test.jsonl test_original.jsonl
```

#### Step 2.2: Create Transformation Script
```python
# File: scripts/transform_dataset.py

import json
import re
from pathlib import Path

def clean_input(text):
    """Remove 'Konteks: ' prefix and prompt instruction"""
    if text.startswith('Konteks: '):
        text = text[len('Konteks: '):]
    
    if '\n\nPrompt:' in text:
        text = text.split('\n\nPrompt:')[0]
    
    return text.strip()

def clean_target(text):
    """Extract only question, remove prefix and answer/distractors"""
    if text.startswith('Pertanyaan: '):
        text = text[len('Pertanyaan: '):]
    
    if '? Jawaban benar:' in text:
        text = text.split('? Jawaban benar:')[0] + '?'
    
    return text.strip()

def transform_file(input_path, output_path):
    """Transform one JSONL file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    transformed = []
    for item in data:
        transformed.append({
            'input': clean_input(item['input']),
            'target': clean_target(item['target'])
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in transformed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(transformed)

# Transform all files
base_dir = Path('dataset_aqg/dataset-task-spesifc')
for split in ['train', 'validation', 'test']:
    input_file = base_dir / f'{split}.jsonl'
    output_file = base_dir / f'{split}_formatted.jsonl'
    
    n = transform_file(input_file, output_file)
    print(f'✅ Transformed {split}: {n} samples')
```

#### Step 2.3: Verify Transformation
```python
# Quick verification script
import json

with open('dataset_aqg/dataset-task-spesifc/train_formatted.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print("Sample transformed item:")
print(f"Input: {sample['input'][:200]}...")
print(f"Target: {sample['target'][:200]}...")

# Check for problematic patterns
with open('dataset_aqg/dataset-task-spesifc/train_formatted.jsonl', 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        if 'Konteks:' in item['input']:
            print(f"❌ Line {i}: Still has 'Konteks:' prefix")
        if 'Pertanyaan:' in item['target']:
            print(f"❌ Line {i}: Still has 'Pertanyaan:' prefix")
        if 'Jawaban benar:' in item['target']:
            print(f"❌ Line {i}: Still has answer in target")

print("✅ Verification complete")
```

#### Step 2.4: Re-run Training with Formatted Dataset
```python
# Update training script to use formatted files
train_dataset = loader.load_dataset(
    'dataset_aqg/dataset-task-spesifc',
    split='train',
    filename='train_formatted.jsonl'
)

# Run full training
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    early_stopping=True
)
```

**Expected Outcome:**
- ✅ Training loss > 0.0 (e.g., 2.5-3.5)
- ✅ Eval loss valid (e.g., 2.8-3.2)
- ✅ BLEU-4: 0.06-0.09 (50-70% improvement)
- ✅ ROUGE-L: 0.18-0.25 (50-70% improvement)

---

### PHASE 3: Optimization (jika Phase 2 berhasil)

#### Step 3.1: Hyperparameter Tuning
```python
# Try different learning rates
learning_rates = [5e-5, 1e-4, 2e-4]

# Try different batch sizes
batch_sizes = [8, 16]

# Try more epochs
num_epochs = [3, 5, 7]
```

#### Step 3.2: Data Augmentation
```python
# Back-translation
# Paraphrasing
# Synthetic data generation
```

#### Step 3.3: Advanced Techniques
```python
# Curriculum learning
# Multi-task learning
# Ensemble methods
```

---

## 🚨 CRITICAL DECISION POINTS

### Decision Point 1: After Phase 1
```
IF loss > 0.0 after DataCollator fix:
    → Continue with current format
    → Focus on hyperparameter tuning
    → Consider data augmentation

ELSE:
    → Proceed to Phase 2 (Format Transformation)
```

### Decision Point 2: After Phase 2
```
IF metrics improve significantly (>50%):
    → Format transformation successful
    → Proceed to Phase 3 (Optimization)

ELSE IF metrics improve moderately (20-50%):
    → Partial success
    → Try Approach 3 (Structured Output)
    → Consider hybrid approach

ELSE:
    → Deep dive required
    → Check model compatibility
    → Review data quality
    → Consider alternative architectures
```

---

## 💡 ALTERNATIVE SCENARIOS

### Scenario A: "Quick Win" (Best Case)
```
1. DataCollator fix already applied
2. Re-run training → Loss > 0.0
3. Metrics improve moderately
4. Apply format transformation → Metrics improve significantly
5. Hyperparameter tuning → Final optimization

Timeline: 1-2 days
Probability: 60%
```

### Scenario B: "Format is Key" (Likely Case)
```
1. DataCollator fix not enough
2. Format transformation required
3. Re-run training → Loss > 0.0, metrics improve
4. Hyperparameter tuning → Further improvement
5. Data augmentation → Final boost

Timeline: 2-3 days
Probability: 30%
```

### Scenario C: "Deep Issues" (Worst Case)
```
1. Both fixes don't work
2. Need to investigate model-data compatibility
3. Consider alternative approaches:
   - Different model architecture
   - Different tokenization strategy
   - Different task formulation

Timeline: 1-2 weeks
Probability: 10%
```

---

## 📋 IMMEDIATE TODO LIST

### Priority 1: CRITICAL (Do Today)
- [ ] **Verify DataCollator fix** in `task_trainer.py`
- [ ] **Re-run training** dengan 1 epoch untuk quick test
- [ ] **Check loss progression** (should be > 0.0)
- [ ] **Decision:** Proceed to format transformation or not?

### Priority 2: HIGH (Do Tomorrow if P1 fails)
- [ ] **Create transformation script** (`scripts/transform_dataset.py`)
- [ ] **Transform all datasets** (train, val, test)
- [ ] **Verify transformation** (no prefixes, correct format)
- [ ] **Re-run training** with formatted dataset
- [ ] **Compare metrics** (before vs after)

### Priority 3: MEDIUM (Do This Week)
- [ ] **Hyperparameter tuning** (if Phase 2 successful)
- [ ] **Data augmentation** (if needed)
- [ ] **Documentation** (update README, record changes)

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Success
- ✅ Training loss > 0.0
- ✅ Eval loss valid (not NaN)
- ✅ BLEU-4 > 0.03 (2x baseline)
- ✅ ROUGE-L > 0.15 (1.2x baseline)

### Target Success
- ✅ Training loss decreasing consistently
- ✅ BLEU-4 > 0.06 (4x baseline)
- ✅ ROUGE-L > 0.20 (1.5x baseline)
- ✅ Output format consistent

### Stretch Goal
- ✅ BLEU-4 > 0.10 (7x baseline)
- ✅ ROUGE-L > 0.25 (1.8x baseline)
- ✅ BERTScore F1 > 0.70
- ✅ Human evaluation positive

---

## 🤝 MY RECOMMENDATION

Berdasarkan analisis mendalam, saya merekomendasikan:

### **IMMEDIATE ACTION: Two-Phase Approach**

**Phase 1 (30 menit):**
1. Verify DataCollator fix di kode
2. Re-run training 1 epoch
3. Check loss > 0.0?

**Phase 2 (2-3 jam, jika Phase 1 gagal):**
1. Transform dataset format (Approach 2)
2. Re-run training full
3. Compare metrics

### **WHY THIS APPROACH?**

1. **Low Risk**: Minimal changes, easy to rollback
2. **High Reward**: Expected 50-70% improvement
3. **Evidence-Based**: Supported by 7 literatur + HuggingFace standard
4. **Quick Validation**: Can verify in < 1 day
5. **Clear Decision Points**: Know when to proceed or pivot

### **CONFIDENCE LEVEL**

- **DataCollator fix alone**: 40% chance of solving issue
- **Format transformation**: 80% chance of significant improvement
- **Combined approach**: 90% chance of success

---

## 📞 NEXT STEPS

**Immediate (Right Now):**
1. Buka `src/finetuned/training/task_trainer.py`
2. Check DataCollator configuration (line 245-252)
3. Confirm fix is applied

**Short-term (Today):**
1. Re-run training dengan 1 epoch
2. Monitor loss progression
3. Make decision: continue or transform format

**Medium-term (This Week):**
1. Implement format transformation if needed
2. Full training run
3. Metrics comparison and analysis

---

**Prepared by:** Kiro AI Assistant  
**Confidence:** HIGH (90%)  
**Recommendation:** Execute Two-Phase Approach  
**Timeline:** 1-2 days for complete resolution
