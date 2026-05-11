# Dataset Design Guide: IndoNanoT5 MCQ Generation

**Status:** FINAL DESIGN  
**Task Type:** Multiple Choice Question Generation (MCQ-G)  
**Model:** LazarusNLP/IndoNanoT5-base  
**Format:** JSONL (JSON Lines)  
**Date:** 22 April 2026

---

## 1. TASK NLP CLASSIFICATION

### Task Type: **Multiple Choice Question Generation (MCQ-G)**

**Definition:**
```
Input:  Context/passage (plain text, preserve code blocks)
Output: Structured MCQ (question + correct answer + distractors)
```

**Why MCQ-G (Not Just QG)?**

| Aspect | Question Generation | MCQ Generation |
|--------|-------------------|-----------------|
| **Output** | Only question | Q + Answer + Distractors |
| **Complexity** | Simple seq2seq | Structured generation |
| **Use Case** | General QA | Educational quizzes |
| **Your Project** | ❌ | ✅ Perfect fit |

**Architecture:** Encoder-Decoder (T5 style)
```
Encoder: Processes context
Decoder: Generates structured MCQ output
```

---

## 2. FORMAT DATASET FINAL

### 2.1 File Format: **JSONL** (JSON Lines)

**Why JSONL, not JSON?**

| Criteria           | JSON                   | JSONL             |
| --------------------| ------------------------| -------------------|
| **Structure**      | Single array           | One JSON per line |
| **Streaming**      | Load all at once       | Load line-by-line |
| **Memory**         | High (all data in RAM) | Low (streaming)   |
| **HuggingFace**    | ❌ Not preferred        | ✅ Preferred       |
| **Large datasets** | Problematic            | Efficient         |

**Your Choice:** ✅ **JSONL** (better for large datasets)

---

### 2.2 Input Format (Source Text)

```
Plain text ONLY - NO markdown formatting
- ✅ Keep code blocks with triple backticks (```)
- ❌ Remove ## headers
- ❌ Remove ** bold markers
- ❌ Remove * italics markers
- ✅ Keep newlines for readability
```

**Why?**
- IndoNanoT5 trained on plain text
- Markdown tokens confuse the model
- Code blocks are content, not formatting

---

### 2.3 Output Format (Target Text)

**Format: Structured Plain Text**

```
question: [QUESTION TEXT]
answer: [CORRECT ANSWER]
distractors: [DISTRACTOR1] | [DISTRACTOR2] | [DISTRACTOR3]
```

**Example:**
```
question: Indeks array dalam Python dimulai dari berapa?
answer: 0
distractors: 1 | -1 | n
```

**Why This Format?**
- Structured but parseable
- No special tokens needed
- Easy to split during inference
- Compatible with T5 text-to-text framework

---

### 2.4 Task Prefix (T5 Convention)

**Prefix to prepend to input:**

```
generate_mcq:
```

**Full Input Format:**
```
generate_mcq: [PLAIN TEXT CONTEXT]
```

**Why Task Prefix?**
- T5 uses prefixes to specify tasks
- Helps model understand what to do
- Standard practice in T5 fine-tuning
- Reference: Raffel et al., T5 paper (2019)

---

## 3. CONTOH DATASET (3 SAMPLES)

### Sample 1: Array Indexing

```json
{
  "input": "generate_mcq: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1. Contoh: jika Anda memiliki list = [10, 20, 30], maka list[0] adalah 10, list[1] adalah 20, dan list[2] adalah 30.",
  "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1"
}
```

### Sample 2: Code Execution

```json
{
  "input": "generate_mcq: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(var_list[1][2])\n```\nKode ini mengakses elemen pada baris kedua (indeks 1) dan kolom ketiga (indeks 2) dari nested list.",
  "output": "question: Apa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9"
}
```

### Sample 3: String Operations

```json
{
  "input": "generate_mcq: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat. Jika Anda mencoba mengubah karakter dalam string dengan indexing seperti s[0] = 'A', Python akan menampilkan error TypeError. Untuk mengubah string, Anda harus membuat string baru.",
  "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string dengan s[0] = 'A'?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError"
}
```

---

## 4. BEST PRACTICES

### 4.1 Input Preprocessing

```python
def preprocess_input(context):
    """
    Preprocess context untuk IndoNanoT5
    """
    # 1. Remove markdown formatting (## ** __ etc)
    context = re.sub(r'^#+\s+', '', context, flags=re.MULTILINE)  # Remove headers
    context = re.sub(r'\*\*', '', context)  # Remove bold
    context = re.sub(r'__', '', context)   # Remove underscores
    context = re.sub(r'\*', '', context)   # Remove italics
    
    # 2. Preserve code blocks (keep ``` markers)
    # DO NOT remove triple backticks
    
    # 3. Normalize whitespace (but keep code indentation)
    lines = context.split('\n')
    processed_lines = []
    in_code_block = False
    
    for line in lines:
        if '```' in line:
            in_code_block = not in_code_block
        
        if in_code_block:
            # Keep indentation in code blocks
            processed_lines.append(line)
        else:
            # Normalize whitespace outside code blocks
            processed_lines.append(line.strip())
    
    context = '\n'.join(processed_lines)
    
    # 4. Remove extra blank lines
    context = re.sub(r'\n\n+', '\n', context)
    
    return context.strip()
```

### 4.2 Output Formatting

```python
def format_output(question, answer, distractors):
    """
    Format output sesuai dengan T5 MCQ-G format
    """
    # Ensure distractors is list
    if isinstance(distractors, str):
        distractors = [d.strip() for d in distractors.split('|')]
    
    # Format output
    output = f"question: {question}\n"
    output += f"answer: {answer}\n"
    output += f"distractors: {' | '.join(distractors)}"
    
    return output
```

### 4.3 Tokenization Considerations

**For IndoNanoT5:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")

# Tokenization settings
max_input_length = 512
max_output_length = 256

# Example
input_text = "generate_mcq: [context]"
output_text = "question: ...\nanswer: ...\ndistractors: ..."

# Tokenize
input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)
output_ids = tokenizer.encode(output_text, max_length=max_output_length, truncation=True)
```

**Key Settings:**
- Max input length: 512 tokens (standard for T5)
- Max output length: 256 tokens (MCQ is shorter)
- Truncation: True (if context too long)
- Padding: True (during training)

### 4.4 Data Validation Checklist

```python
def validate_dataset(jsonl_file):
    """
    Validate dataset format sebelum training
    """
    issues = []
    
    with open(jsonl_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"Line {idx}: Invalid JSON")
                continue
            
            # Check required fields
            if 'input' not in item:
                issues.append(f"Line {idx}: Missing 'input' field")
            if 'output' not in item:
                issues.append(f"Line {idx}: Missing 'output' field")
            
            # Check input format
            if not item['input'].startswith('generate_mcq:'):
                issues.append(f"Line {idx}: Input missing 'generate_mcq:' prefix")
            
            # Check output format
            output = item['output']
            if 'question:' not in output:
                issues.append(f"Line {idx}: Output missing 'question:' field")
            if 'answer:' not in output:
                issues.append(f"Line {idx}: Output missing 'answer:' field")
            if 'distractors:' not in output:
                issues.append(f"Line {idx}: Output missing 'distractors:' field")
            
            # Check input length
            if len(item['input']) > 2048:
                issues.append(f"Line {idx}: Input too long ({len(item['input'])} chars)")
            
            # Check for markdown formatting in input
            if '##' in item['input'] or '**' in item['input']:
                issues.append(f"Line {idx}: Input contains markdown formatting")
    
    return issues
```

---

## 5. REFERENSI

### 5.1 T5 Text-to-Text Framework

**Reference:** Raffel, C., Shazeer, N., Roberts, A., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." arXiv:1910.10683

**Key Points:**
- T5 converts all NLP tasks to text-to-text format
- Task prefixes guide model behavior
- No task-specific architectures needed
- Format: `[prefix]: [input] → [output]`

**Citation:**
```bibtex
@article{raffel2019exploring,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and others},
  journal={arXiv preprint arXiv:1910.10683},
  year={2019}
}
```

---

### 5.2 IndoNanoT5 Documentation

**Reference:** LazarusNLP/IndoNanoT5-base (HuggingFace Hub)

**Model Details:**
- Architecture: T5 (Encoder-Decoder)
- Language: Indonesian
- Pretraining: 4B tokens from CulturaX corpus
- Input length: 512 tokens
- License: Apache 2.0

**URL:** https://huggingface.co/LazarusNLP/IndoNanoT5-base

---

### 5.3 Question Generation with T5

**Reference:** Patil, S. (2021). "Question Generation using Transformers." GitHub Repository

**Key Insights:**
- T5 can handle multiple QG variants (answer-aware, end-to-end, MCQ)
- Task prefixes: `generate_question:`, `generate_mcq:`, etc.
- Output format: Structured text with separators
- JSONL format recommended for large datasets

**URL:** https://github.com/patil-suraj/question_generation

---

### 5.4 MCQ Generation Literature

**Reference:** Automatic Generation of Multiple-Choice Questions (arXiv:2303.14576)

**Key Findings:**
- Two-stage approach: QG + Distractor generation
- T5 can be fine-tuned for end-to-end MCQ generation
- Output format: Structured text (question + answer + distractors)
- Evaluation: BLEU, ROUGE, human evaluation

---

### 5.5 HuggingFace Question Answering Documentation

**Reference:** HuggingFace Transformers - Seq2Seq QA

**Best Practices:**
- JSONL format preferred for streaming
- Separate input/output fields
- Tokenization: max_length=512 for input, 256 for output
- Batch size: 16-32 for fine-tuning

**URL:** https://huggingface.co/docs/transformers/tasks/question_answering

---

## 6. IMPLEMENTATION CHECKLIST

### Before Training:

- [ ] Dataset in JSONL format
- [ ] All inputs start with `generate_mcq:` prefix
- [ ] All inputs have plain text (no markdown)
- [ ] All inputs preserve code blocks (``` markers)
- [ ] All outputs have `question:`, `answer:`, `distractors:` fields
- [ ] No markdown formatting in inputs (##, **, __, etc.)
- [ ] Dataset validated with validation script
- [ ] Train/val/test split created (80/10/10)
- [ ] Sample data reviewed manually

### During Training:

- [ ] Monitor training loss (should > 0, not 0)
- [ ] Monitor eval loss (should be valid number, not NaN)
- [ ] Check generated outputs periodically
- [ ] Verify no gradient conflicts

### After Training:

- [ ] Evaluate on test set
- [ ] Calculate BLEU, ROUGE metrics
- [ ] Manual evaluation of generated MCQs
- [ ] Compare with baseline

---

## 7. EXAMPLE JSONL FILE

**File: train.jsonl**

```jsonl
{"input": "generate_mcq: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1.", "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1"}
{"input": "generate_mcq: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```", "output": "question: Apa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9"}
{"input": "generate_mcq: String dalam Python bersifat immutable, artinya tidak dapat diubah setelah dibuat.", "output": "question: Apa yang terjadi jika Anda mencoba mengubah karakter string?\nanswer: TypeError\ndistractors: ValueError | IndexError | AttributeError"}
```

---

## 8. COMPARISON: JSON vs JSONL vs CSV

| Criteria | JSON | JSONL | CSV |
|----------|------|-------|-----|
| **Format** | Single array | Line-delimited | Comma-separated |
| **Streaming** | ❌ No | ✅ Yes | ✅ Yes |
| **Memory** | High | Low | Low |
| **HuggingFace** | ⚠️ Limited | ✅ Preferred | ⚠️ Limited |
| **Nested data** | ✅ Yes | ✅ Yes | ❌ No |
| **Large datasets** | ❌ Problematic | ✅ Efficient | ✅ Efficient |
| **Your choice** | ❌ | ✅ | ⚠️ |

**Recommendation:** Use **JSONL** for IndoNanoT5

---

## 9. TROUBLESHOOTING

### Problem: Model generates repetitive output

**Cause:** Markdown formatting in input confuses model
**Solution:** Remove all markdown (##, **, __, etc.)

### Problem: Training loss = 0

**Cause:** Duplicate samples or incorrect label masking
**Solution:** Deduplicate dataset, verify DataCollator

### Problem: Eval loss = NaN

**Cause:** Numerical instability or incorrect output format
**Solution:** Check output format, verify tokenization

### Problem: Model doesn't understand task

**Cause:** Missing or incorrect task prefix
**Solution:** Ensure all inputs start with `generate_mcq:`

---

**Version:** 1.0  
**Last Updated:** 22 April 2026  
**Status:** READY FOR IMPLEMENTATION

