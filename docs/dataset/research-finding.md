# Research Findings: IndoT5 Dataset Format Compatibility for AQG

## 1. Error Analysis from error.md

### Current Status
- ✅ Tokenization: 201/201 labels valid (100%)
- ✅ DataCollator: 201/208 non-masked (96.6% valid)
- ✅ Forward Pass: Loss = 9.9250 (VALID)
- ⚠️ Generation: Output format mismatch

### Key Issue Identified
**Problem**: Model generates "Perbandingan Penggunaan Memori Konteks..." instead of "Pertanyaan: Sesuai catatan modul..."

**Root Cause**: NOT a dataset/preprocessing/tokenizer issue - it's a **model training issue**. The base model hasn't been fine-tuned on AQG task yet.

### Data Pipeline Status
```
Input: 973 chars → Tokenized: 319 tokens ✓
Target: 422 chars → Tokenized: 201 tokens ✓
Padding: Correct (0 tokens) ✓
Masking: 96.6% valid ✓
```

---

## 2. IndoT5 Official Documentation Findings

### Repository: LazarusNLP/IndoT5

#### Expected Dataset Format for Fine-tuning

**For Summarization Task** (most similar to AQG):
```python
# JSONLINES Format (Recommended)
{"input": "source text", "target": "summary text"}
{"input": "another source", "target": "another summary"}

# CSV Format (Alternative)
text,summary
"source text","summary text"
"another source","another summary"
```

#### Key Configuration Parameters

From `run_summarization.py`:
```bash
--model_name_or_path LazarusNLP/IndoNanoT5-base
--dataset_name LazarusNLP/indonlg
--dataset_config indosum
--input_column_name input
--target_column_name target
--input_max_length 512
--target_max_length 512
--num_beams 5
--per_device_train_batch_size 8
--learning_rate 1e-3
--weight_decay 0.01
--num_train_epochs 5
```

#### Tokenizer Details
- **Type**: SentencePiece (trained on Indonesian CulturaX dataset)
- **Vocab Size**: 32,000 tokens
- **Encoding**: Byte-Pair Encoding (BPE)
- **Special Tokens**:
  - `<pad>`: 0
  - `</s>`: 1
  - `<unk>`: 2
  - `<extra_id_0>` to `<extra_id_99>`: 3-102

#### Task Prefix Convention
For T5 models, input format follows:
```
"<task_prefix>: <input_text>"
```

For AQG, this should be:
```
"generate_question: <context_text>"
```

---

## 3. HuggingFace Transformers Summarization Example

### Supported T5 Architectures
- `T5ForConditionalGeneration`
- `MT5ForConditionalGeneration`
- `BartForConditionalGeneration`
- `MBartForConditionalGeneration`

### Dataset Format Standards

#### JSONLINES Format (Recommended)
```json
{
  "input": "Context or source text",
  "target": "Expected output or summary"
}
```

#### CSV Format
```csv
input,target
"Context text","Expected output"
"Another context","Another output"
```

**Key Requirement**: Column names MUST match `--input_column_name` and `--target_column_name` parameters.

### Preprocessing Pipeline

1. **Load Dataset**:
   ```python
   dataset = load_dataset("json", data_files="train.jsonl")
   ```

2. **Tokenization**:
   ```python
   def preprocess_function(examples):
       inputs = [ex for ex in examples["input"]]
       targets = [ex for ex in examples["target"]]
       
       model_inputs = tokenizer(
           inputs,
           max_length=512,
           truncation=True,
           padding="max_length"
       )
       
       labels = tokenizer(
           targets,
           max_length=512,
           truncation=True,
           padding="max_length"
       )
       
       model_inputs["labels"] = labels["input_ids"]
       return model_inputs
   ```

3. **DataCollator**:
   ```python
   from transformers import DataCollatorForSeq2Seq
   
   data_collator = DataCollatorForSeq2Seq(
       tokenizer,
       model=model,
       padding=True,
       pad_to_multiple_of=8
   )
   ```

### Training Configuration

**Recommended for T5 Fine-tuning**:
- **Learning Rate**: 1e-3 to 1e-4
- **Batch Size**: 8-16
- **Epochs**: 3-5
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup
- **Warmup Steps**: 6% of total steps

---

## 4. T5 Model for Question Generation (Research Paper)

### Key Findings from "AI and NLP-Based Question Generation Using T5 Model"

#### Dataset Format Requirements
- **Input Column**: Context/passage text
- **Target Column**: Generated question
- **Format**: JSONLINES or CSV

#### Preprocessing Steps
1. Text normalization (lowercase, remove special chars)
2. Tokenization using T5 tokenizer
3. Truncation to max_length (512 tokens typical)
4. Padding to uniform length
5. Label creation from target text

#### Model Configuration
```python
T5Config(
    vocab_size=32000,
    d_model=512,  # or 768 for larger models
    d_kv=64,
    d_ff=2048,
    num_layers=12,
    num_decoder_layers=12,
    num_heads=8,
    relative_attention_num_buckets=32,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="relu",
    is_encoder_decoder=True,
    decoder_start_token_id=0,
    pad_token_id=0,
    eos_token_id=1,
    max_length=512
)
```

#### Task Prefix Convention
```
"generate_question: <context>"
```

This prefix tells the model which task to perform.

---

## 5. Analysis: Why Model Doesn't Understand Current Format

### Current Dataset Format (Your Project)
```json
{
  "input": "Konteks: ### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Pertanyaan: Sesuai catatan modul..."
}
```

### Issues Identified

1. **Missing Task Prefix**
   - ❌ Current: No prefix in input
   - ✅ Expected: `"generate_question: Konteks: ..."`

2. **Input Format Inconsistency**
   - ❌ Current: "Konteks: " prefix in input
   - ✅ Expected: Standardized format with task prefix

3. **Target Format Inconsistency**
   - ❌ Current: "Pertanyaan: " prefix in target
   - ✅ Expected: Just the question text without prefix

### Why This Matters

T5 models are trained with specific conventions:
- **Input**: `<task>: <content>`
- **Target**: `<output_text>` (no task prefix)

The model learns to recognize the task prefix and generate appropriate output. Without it, the model defaults to copying input tokens.

---

## 6. Recommended Dataset Format Alignment

### Option A: Minimal Changes (Recommended)

**Current Format**:
```json
{
  "input": "Konteks: ### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Pertanyaan: Sesuai catatan modul..."
}
```

**Recommended Format**:
```json
{
  "input": "generate_question: Konteks: ### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul..."
}
```

**Changes**:
1. Add `generate_question: ` prefix to input
2. Remove `Pertanyaan: ` prefix from target

### Option B: Full Standardization

```json
{
  "input": "generate_question: ### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul..."
}
```

**Changes**:
1. Add `generate_question: ` prefix to input
2. Remove `Konteks: ` from input (implicit from task)
3. Remove `Pertanyaan: ` from target

### Option C: Multi-Task Format (Advanced)

```json
{
  "input": "generate_question_code: ### Perbandingan Penggunaan Memori\n\n```python\n...",
  "target": "Sesuai catatan modul..."
}
```

**Benefit**: Allows training on multiple task types (generate_question_code, generate_distractor, etc.)

---

## 7. Preprocessing Recommendations

### Current Preprocessing (Good)
```python
# From error.md analysis
Input IDs length: 319 ✓
Label IDs length: 201 ✓
Padding tokens: 0 ✓
```

### Recommended Enhancements

1. **Add Task Prefix During Preprocessing**:
```python
def preprocess_aqg(examples):
    inputs = [f"generate_question: {ex}" for ex in examples["input"]]
    targets = examples["target"]  # Remove prefixes
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

2. **Handle Code Blocks Properly**:
```python
def preprocess_with_code_handling(examples):
    # Preserve code block structure
    inputs = []
    for ex in examples["input"]:
        # Keep triple backticks and code formatting
        inputs.append(f"generate_question: {ex}")
    
    # ... rest of preprocessing
```

3. **Validation**:
```python
# Ensure no token overflow
assert max(len(tokenizer.encode(ex)) for ex in inputs) <= 512
assert max(len(tokenizer.encode(ex)) for ex in targets) <= 256
```

---

## 8. Tokenizer Configuration

### Current Tokenizer (Good)
- Model: IndoT5 SentencePiece tokenizer
- Vocab Size: 32,000
- Encoding: BPE
- Pad Token ID: 0 ✓
- EOS Token ID: 1 ✓

### Recommended Tokenizer Settings

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")

# Verify special tokens
print(f"Pad token ID: {tokenizer.pad_token_id}")  # Should be 0
print(f"EOS token ID: {tokenizer.eos_token_id}")  # Should be 1
print(f"Vocab size: {len(tokenizer)}")            # Should be 32000

# Test tokenization
test_input = "generate_question: Konteks: test"
tokens = tokenizer.encode(test_input)
print(f"Tokenized length: {len(tokens)}")
```

### Code-Specific Tokenization

For Python code in mixed content:
```python
# The standard T5 tokenizer handles code reasonably well
# But you can verify with:

code_snippet = '''python
import numpy
var_list = [[1, 2, 3]]
'''

tokens = tokenizer.encode(code_snippet)
decoded = tokenizer.decode(tokens)
print(f"Original length: {len(code_snippet)}")
print(f"Tokens: {len(tokens)}")
print(f"Decoded matches: {decoded.strip() == code_snippet.strip()}")
```

---

## 9. Summary of Required Changes

### Dataset Format
| Aspect | Current | Recommended | Impact |
|--------|---------|-------------|--------|
| Input Prefix | None | `generate_question: ` | HIGH |
| Input Content | With "Konteks:" | Keep as-is | LOW |
| Target Prefix | "Pertanyaan: " | Remove | HIGH |
| Target Content | Full question | Same | NONE |
| File Format | JSONLINES | JSONLINES | NONE |
| Column Names | "input", "target" | "input", "target" | NONE |

### Preprocessing
| Step         | Current                   | Recommended          | Impact |
| --------------| ---------------------------| ----------------------| --------|
| Task Prefix  | Not added                 | Add in preprocessing | HIGH   |
| Tokenization | Standard T5               | Standard T5          | NONE   |
| Max Length   | 512 (input), 256 (target) | Same                 | NONE   |
| Padding      | Correct                   | Correct              | NONE   |
| DataCollator | Correct                   | Correct              | NONE   |

### Expected Improvements
- ✅ Model will recognize task intent
- ✅ Generation will start with appropriate tokens
- ✅ Output format will match expectations
- ✅ Training convergence will be faster
- ✅ Evaluation metrics will improve

---

## 10. Validation Checklist

Before retraining, verify:

- [ ] Dataset format updated with task prefix
- [ ] Target text cleaned (no "Pertanyaan: " prefix)
- [ ] JSONLINES file is valid (test with `json.loads()`)
- [ ] Tokenization test passes
- [ ] DataCollator works correctly
- [ ] Forward pass produces valid loss (not 0.0 or NaN)
- [ ] Sample predictions show reasonable output format
- [ ] Training loss decreases over epochs
- [ ] Evaluation metrics improve

