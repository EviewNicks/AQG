# Dataset Design Guide: IndoNanoT5 MCQ Generation

**Status:** FINAL DESIGN  
**Task Type:** Multiple Choice Question Generation (MCQ-G)  
**Model:** LazarusNLP/IndoNanoT5-base  
**Format:** JSONL (JSON Lines)  
**Date:** 1 May 2026  
**Version:** 3.1

---

## 1. OVERVIEW

**Task:** Generate Multiple Choice Questions (MCQ) from Indonesian text context

**Input:** `buat_soal_pilihan_ganda: [context]`  
**Output:** `question: [text]\nanswer: [text]\ndistractors: [opt1] | [opt2] | [opt3]`

**Architecture:** T5 Encoder-Decoder (text-to-text)

---

## 2. FORMAT REQUIREMENTS

### 2.1 File Format

**Use JSONL** (one JSON object per line)
- Efficient for large datasets
- Streaming-friendly
- HuggingFace preferred format

### 2.2 Input Format

```
buat_soal_pilihan_ganda: [PLAIN TEXT CONTEXT]
```

**Rules:**
- Plain text only (no markdown: ##, **, __)
- Keep code blocks with triple backticks (```)
- Minimal 1-2 sentences explanation
- For code: explain what the code does

### 2.3 Output Format

```
question: [QUESTION TEXT]
answer: [CORRECT ANSWER]
distractors: [DISTRACTOR1] | [DISTRACTOR2] | [DISTRACTOR3]
```

### 2.4 Input Length Limits

**Token Limit:** IndoNanoT5 has **512 tokens maximum** for input

**Guidelines:**
- Input context: 50-200 words recommended
- Maximum: ~400 words (to stay under 512 tokens)
- If context too long, split into multiple samples
- Check token count before finalizing dataset

**Why?**
- Model architecture constraint (T5 base)
- Longer contexts may be truncated
- Shorter contexts = better focus and quality

---

## 3. METADATA STRUCTURE

Every sample MUST include metadata:

```json
{
  "input": "...",
  "output": "...",
  "metadata": {
    "difficulty": "Mudah|Sedang|Sulit",
    "type": "knowledge|code"
  }
}
```

### Metadata Fields

**difficulty** (Required)
- `"Mudah"`: Direct recall, simple concepts
- `"Sedang"`: Application, multi-step reasoning
- `"Sulit"`: Synthesis, complex scenarios

**type** (Required)
- `"knowledge"`: Conceptual questions, definitions, theory
- `"code"`: Questions with code blocks, output prediction

---

## 4. TYPE DISTRIBUTION (CRITICAL)

**Target Ratio:**
- `type: "knowledge"` → **≥ 60%**
- `type: "code"` → **≤ 40%**

**Tolerance Range (±10%):**
- ✅ **VALID:** Knowledge 50-100%, Code 0-50%
- ⚠️ **ACCEPTABLE:** Knowledge 50-60%, Code 40-50% (within tolerance)
- ❌ **INVALID:** Knowledge <50%, Code >50%

**Examples:**
- 65% knowledge, 35% code → ✅ Valid
- 55% knowledge, 45% code → ⚠️ Acceptable (within ±10% tolerance)
- 70% knowledge, 30% code → ✅ Valid
- 48% knowledge, 52% code → ❌ Invalid (exceeds tolerance)

**Why?**
- Prevents overfitting on code patterns
- Improves conceptual understanding
- Better generalization
- Tolerance allows natural variation per topic

**Validation:**
```
Knowledge % = (knowledge_count / total) × 100
Code % = (code_count / total) × 100

Valid if: Knowledge ≥ 50% AND Code ≤ 50%
Ideal if: Knowledge ≥ 60% AND Code ≤ 40%
```

---

## 5. CRITICAL RULES

### 5.1 Self-Contained Output

Question MUST be complete and independent from input!

**❌ WRONG:**
```json
{
  "input": "buat_soal_pilihan_ganda: Kode: ```python\nx = 5\n```",
  "output": "question: Apa output kode di atas?\nanswer: 5"
}
```

**✅ CORRECT:**
```json
{
  "input": "buat_soal_pilihan_ganda: Kode: ```python\nx = 5\n```",
  "output": "question: Perhatikan kode berikut:\n```python\nx = 5\n```\nApa nilai dari x?\nanswer: 5"
}
```

### 5.2 Complete Input Context

Input must have sufficient explanation (1-2 sentences minimum)

**❌ WRONG:**
```json
{"input": "buat_soal_pilihan_ganda: Aksi sekuensial"}
```

**✅ CORRECT:**
```json
{"input": "buat_soal_pilihan_ganda: Aksi sekuensial adalah sederetan instruksi yang dijalankan komputer berdasarkan urutan penulisannya. Dalam Python, program dijalankan dari atas ke bawah."}
```

### 5.3 Code Block Rules

**Include code block in question if:**
- Question refers to "kode di atas/berikut"
- Question asks for output/execution result
- Question asks for code analysis

**Don't include if:**
- Question only mentions inline syntax: `print()`
- Question is purely conceptual

### 5.4 Distractor Quality Rules

Distractors MUST be plausible and educationally valuable!

**Requirements:**
- Plausible (not random/nonsense)
- Common misconceptions or similar concepts
- Avoid obviously wrong answers
- All 3 distractors distinct from each other and from answer

**❌ BAD Distractors:**
```json
{"distractors": "999 | abc | qwerty"}  // Random/nonsense
{"distractors": "Error | Error | Salah"}  // Not distinct
```

**✅ GOOD Distractors:**
```json
{"distractors": "1 | -1 | n"}  // Related to indexing
{"distractors": "ValueError | IndexError | AttributeError"}  // Similar error types
```

### 5.5 Answer Length Guidelines

Keep answers concise and focused!

**Guidelines:**
- Short answers preferred (1-5 words)
- For code output: exact output only
- For definitions: key phrase, not full sentence
- Avoid explanations in answer field

**Examples:**
- ✅ `"answer": "20"`
- ✅ `"answer": "TypeError"`
- ✅ `"answer": "Fungsi yang menerima argumen bervariasi"`
- ❌ `"answer": "Jawaban yang benar adalah 20 karena list[1] mengakses elemen kedua"`

### 5.6 Question Clarity Rules

Questions must be clear, specific, and unambiguous!

**Requirements:**
- Use specific wording: "Apa output...", "Apa yang dimaksud...", "Berapa nilai..."
- Avoid vague questions: "Bagaimana dengan...", "Apa tentang..."
- One question per sample (no compound questions)
- Question must be answerable from given context

**❌ VAGUE:**
```json
{"question": "Bagaimana dengan variabel x?"}
{"question": "Apa tentang list di Python?"}
```

**✅ CLEAR:**
```json
{"question": "Apa nilai dari variabel x?"}
{"question": "Apa yang dimaksud dengan list di Python?"}
```

### 5.7 Indonesian Language Quality

Use proper Indonesian grammar and formal tone!

**Requirements:**
- Follow EYD (Ejaan Yang Disempurnakan)
- Use formal/educational tone
- Technical terms in English are acceptable (e.g., "list", "array", "function")
- Avoid colloquial language

**Examples:**
- ✅ `"Apa yang dimaksud dengan..."`
- ❌ `"Apa maksudnya..."`
- ✅ `"Perhatikan kode berikut"`
- ❌ `"Liat kode ini"`
- ✅ `"Fungsi print() digunakan untuk..."`
- ❌ `"Fungsi print() dipake buat..."`

### 5.8 Context Generation from Material

When generating context from source material, follow these rules:

**Context Source:**
- All contexts MUST be derived from or aligned with the source material
- Example: For `01-perkenalan-pythn.md`, contexts must relate to topics covered:
  - Python Introduction (history, creator, purpose)
  - Python Features (readability, exception handling, syntax)
  - Python Versions (2.x, 3.x, 3.11)
  - Python Overview (PSF, PEP, Zen of Python)
  - Why Python (use cases, paradigms, sectors)

**Context Variation:**
- ✅ Paraphrase material content in different ways
- ✅ Rephrase explanations while maintaining accuracy
- ✅ Create new examples aligned with material topics
- ✅ Combine multiple concepts from material
- ❌ Do NOT introduce concepts outside material scope
- ❌ Do NOT contradict material content

**Context Flexibility:**
- Contexts can be presented as stories, scenarios, or direct explanations
- Contexts can use different wording/phrasing as long as meaning is preserved
- Contexts can combine related topics from material
- Contexts should feel natural and varied, not repetitive

**Coverage Requirement:**
- Ensure all major topics from source material are represented
- Distribute samples across different topics/sections
- Avoid over-focusing on single topic

---

## 6. BATCH GENERATION GUIDELINES

When generating large datasets (100+ samples), use incremental batch approach:

**Batch Size:**
- Recommended batch size: **30-40 samples per batch**
- Rationale: Prevents file write errors and allows validation between batches
- Example: For 205 samples → 6 batches of ~34 samples each

**Batch Workflow:**
1. **Batch 1 (Samples 1-30/40):** Create new file `[filename]_generated.jsonl`
2. **Batch 2-N:** Append to existing file
3. **Final:** Merge generated file with original file

**Quality Assurance per Batch:**
- Validate each batch before appending
- Check for duplicates within batch
- Verify metadata distribution
- Confirm all rules compliance

**Batch Documentation:**
- Track batch number and sample count
- Record any issues or adjustments
- Note topic coverage per batch

### Example 1: Knowledge Type

```json
{
  "input": "buat_soal_pilihan_ganda: Dalam Python, array atau list menggunakan indexing untuk mengakses elemen. Indexing dimulai dari 0, bukan 1.",
  "output": "question: Jika list = [10, 20, 30], apa nilai dari list[1]?\nanswer: 20\ndistractors: 10 | 30 | 1",
  "metadata": {"difficulty": "Mudah", "type": "knowledge"}
}
```

### Example 2: Code Type

```json
{
  "input": "buat_soal_pilihan_ganda: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```\nKode ini mengakses elemen baris kedua, kolom ketiga.",
  "output": "question: Perhatikan kode berikut:\n```python\nvar_list = [[1, 2, 3], [4, 5, 6]]\nprint(var_list[1][2])\n```\nApa output dari kode di atas?\nanswer: 6\ndistractors: 5 | 8 | 9",
  "metadata": {"difficulty": "Sedang", "type": "code"}
}
```

---

## 7. VALIDATION CHECKLIST

Before finalizing dataset:

- [ ] All inputs start with `buat_soal_pilihan_ganda:`
- [ ] Input has 1-2 sentences explanation minimum
- [ ] Input length under 400 words (~512 tokens)
- [ ] Questions are self-contained
- [ ] Code blocks copied to question if referenced
- [ ] All samples have metadata (difficulty, type)
- [ ] Type distribution: knowledge ≥ 50%, code ≤ 50% (ideal: 60/40)
- [ ] Plain text format (no markdown except code blocks)
- [ ] Output format: `question:`, `answer:`, `distractors:`
- [ ] Answers are concise (1-5 words preferred)
- [ ] Questions are clear and specific
- [ ] Distractors are plausible and distinct
- [ ] Indonesian language follows EYD
- [ ] No duplicate samples

---

## 8. TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Model generates repetitive output | Remove markdown formatting from input |
| Training loss = 0 | Check for duplicates, verify DataCollator |
| Model doesn't understand task | Ensure all inputs have task prefix |
| Type distribution imbalanced | Follow 60/40 ratio (knowledge/code) |
| Input too long (>512 tokens) | Split context into multiple samples or shorten |
| Poor distractor quality | Use plausible misconceptions, not random values |
| Vague questions | Use specific wording: "Apa output...", "Berapa nilai..." |

---

**Version:** 3.1  
**Last Updated:** 1 May 2026  
**Status:** READY FOR IMPLEMENTATION
