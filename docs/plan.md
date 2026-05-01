# Plan: Dataset Generation & Integration - dataset-task-v4

**Date:** 1 May 2026  
**Goal:** Generate 220 MCQ samples per materi file compliant with 03-Dataset-Design-Guide-v3.md  
**Target:** Each materi file has minimum 220 valid samples

---

## Current State

**File:** `dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python.jsonl`
- Total samples: 110
- Valid samples: 15
- Invalid samples: 95 (duplicates to remove)
- Gap: 205 samples needed to reach 220 target

---

## Pipeline Overview

```
Phase 1: Cleanup (Remove Invalid Samples)
  └── Remove 95 duplicate/invalid samples
      └── Result: 15 valid samples remain

Phase 2: Generate New Samples
  └── Create 205 new MCQ samples
      ├── Follow 03-Dataset-Design-Guide-v3.md rules
      ├── Maintain 60/40 knowledge/code ratio
      ├── Distribute difficulty: Mudah/Sedang/Sulit
      └── Save to temporary file: `01-perkenalan-python_generated.jsonl`

Phase 3: Merge & Validate
  └── Append generated samples to main file
      ├── Result: 220 total samples
      ├── Validate no duplicates
      └── Confirm all rules compliance
```

---

## Phase 1: Cleanup

**Action:** Remove 95 invalid samples (lines 16-110)
- Reason: Duplicate input text
- Keep: 15 valid samples (lines 1-15)
- Result: Clean base file with 15 samples

---

## Phase 2: Generate New Samples (205 samples)

**Format Reference:** `docs/dataset/03-Dataset-Design-Guide-v3.md`

### Sample Structure (JSONL format)
```json
{
  "input": "buat_soal_pilihan_ganda: [CONTEXT 1-2 SENTENCES]",
  "output": "question: [CLEAR QUESTION]\nanswer: [CONCISE ANSWER]\ndistractors: [OPT1] | [OPT2] | [OPT3]",
  "metadata": {
    "difficulty": "Mudah|Sedang|Sulit",
    "type": "knowledge|code"
  }
}
```

### Generation Rules (from 03-Dataset-Design-Guide-v3.md)

**Input Requirements:**
- ✅ Start with `buat_soal_pilihan_ganda:`
- ✅ Plain text only (no markdown except code blocks)
- ✅ 1-2 sentences minimum explanation
- ✅ 50-200 words recommended (max ~400 words / 512 tokens)
- ✅ For code: explain what code does

**Output Requirements:**
- ✅ Questions self-contained (complete without input)
- ✅ Include code block in question if referenced
- ✅ Answers concise (1-5 words preferred)
- ✅ Distractors plausible & distinct (3 options separated by `|`)
- ✅ Format: `question:`, `answer:`, `distractors:`

**Metadata Requirements:**
- ✅ `difficulty`: Mudah (direct recall) | Sedang (application) | Sulit (synthesis)
- ✅ `type`: knowledge (conceptual) | code (with code blocks)

**Type Distribution (CRITICAL):**
- Knowledge: ≥ 60% (132 samples minimum)
- Code: ≤ 40% (88 samples maximum)

**Difficulty Distribution (Target):**
- Mudah: ~40% (88 samples)
- Sedang: ~45% (99 samples)
- Sulit: ~15% (33 samples)

**Language Quality:**
- ✅ Follow EYD (Ejaan Yang Disempurnakan)
- ✅ Formal/educational tone
- ✅ Technical terms in English acceptable
- ✅ Avoid colloquial language

**Quality Checks:**
- ✅ No duplicate inputs
- ✅ No duplicate questions
- ✅ No duplicate text within questions
- ✅ All samples have required fields
- ✅ All samples have valid metadata

### Output File
- **Temporary file:** `dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python_generated.jsonl`
- **Content:** 205 new samples (one JSON object per line)

---

## Phase 3: Merge & Validate

**Action:** Merge generated samples directly into main file
```
1. Generate samples in batches to temporary _generated.jsonl file
2. Append all generated samples directly to main file (01-perkenalan-python.jsonl)
3. Result: 220 total samples in main file
4. Delete temporary _generated.jsonl file after merge
```

**Merge Process:**
- Read from: `01-perkenalan-python_generated.jsonl` (temporary)
- Write to: `01-perkenalan-python.jsonl` (main file - DIRECT MERGE)
- Keep only main file after merge complete

**Validation:**
- ✅ Total count: 220 samples in main file
- ✅ Type ratio: knowledge ≥ 60%, code ≤ 40%
- ✅ No duplicates (input, question, or text)
- ✅ All samples have required fields
- ✅ All samples follow format rules
- ✅ All samples follow language rules

---

## Execution Order

1. **Phase 1:** Remove 95 invalid samples → 15 valid remain in main file
2. **Phase 2:** Generate 224 new samples → save to `_generated.jsonl` (temporary)
3. **Phase 3:** Merge directly → append all from `_generated.jsonl` to main file (01-perkenalan-python.jsonl)
4. **Cleanup:** Delete temporary `_generated.jsonl` file
5. **Validate:** Confirm 220 total samples in main file with all rules compliance

---

## Key References

- **Design Guide:** `docs/dataset/03-Dataset-Design-Guide-v3.md`
- **Main File (Final):** `dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python.jsonl`
- **Temporary File (Delete after merge):** `dataset_aqg/dataset-task-v4/01-perkenalan-python/01-perkenalan-python_generated.jsonl`
- **Report:** `scripts/03-dataset-design/report-dataset.md`

---

## Notes

- Generate samples to temporary `_generated.jsonl` file for batching
- **MERGE DIRECTLY to main file** (01-perkenalan-python.jsonl) - not to _generated.jsonl
- Delete temporary `_generated.jsonl` after successful merge
- Follow all rules from 03-Dataset-Design-Guide-v3.md strictly
- Validate each batch before appending to main file
- Maintain data integrity throughout process
- Final result: 220 samples in main file only
