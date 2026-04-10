# Dataset Validation Guide

## Overview

Notebook `src/pipeline/validate_dataset.ipynb` melakukan validasi 2-level pada dataset AQG untuk memastikan kualitas sebelum fine-tuning.

## Struktur Validasi

### Level 1: Per-Module Validation (`output_modul/`)

**Tujuan:** Deteksi masalah per modul sebelum merge

**Metrics yang Dicek:**
1. **Total Data Points** - Jumlah soal per modul
2. **Difficulty Distribution** - Distribusi easy/medium/hard
3. **Imbalance Ratio** - max(difficulty) / min(difficulty)
   - ✅ OK: ratio ≤ 2.0
   - ⚠️ WARNING: ratio > 2.0
4. **Validation Rate** - % data yang lolos validasi
   - ✅ OK: ≥ 90%
   - ⚠️ WARNING: < 90%
5. **Misconception Tags Coverage** - % data yang punya tags
   - ✅ OK: ≥ 90%
   - ⚠️ WARNING: < 90%
6. **Concept Diversity** - Jumlah konsep unik per modul
   - ⚠️ WARNING: < 3 konsep

**Output:**
- Summary table (console)
- Per-module report (JSON)
- Issues list (untuk debugging)

---

### Level 2: Final Dataset Validation (`dataset-task-spesifc/`)

**Tujuan:** Validasi dataset gabungan siap fine-tuning

**Checks:**
1. **File Structure**
   - train.jsonl, validation.jsonl, test.jsonl ada
   - dataset_info.json lengkap
   
2. **Split Ratio**
   - Train: 70% ± 2%
   - Validation: 15% ± 2%
   - Test: 15% ± 2%

3. **Stratification**
   - Setiap split punya semua difficulty levels
   - Setiap split punya semua question types
   - Distribusi concept seimbang

4. **HuggingFace Compatibility**
   - Load dengan `datasets.load_dataset()` berhasil
   - Schema konsisten (input, target, metadata)
   - Tipe data benar (string, dict)

5. **Cross-Module Analysis**
   - Distribusi modul di train/val/test
   - Concept overlap detection
   - Global difficulty balance

**Output:**
- Final validation report (JSON)
- HuggingFace dataset preview
- Pass/Fail decision
- Recommendations

---

## Usage

### 1. Run Validation

```bash
# Open notebook
jupyter notebook src/pipeline/validate_dataset.ipynb

# Run all cells
```

### 2. Interpret Results

**Level 1 Output:**
```
Modul                                          Total   Easy    Med   Hard  Ratio   Val%  Tags%  Status
====================================================================================================
01-berkenalan-dengan-python                      120     45     40     35   1.29  98.0%  95.0%  ✓ OK
02-berinteraksi-dengan-data                      150     30     60     60   2.00  96.0%  94.0%  ✓ OK
03-ekspresi                                       80     10     40     30   4.00  92.0%  90.0%  ⚠️ WARNING
```

**Interpretation:**
- Modul 03 has imbalance (ratio=4.00) → Need re-generate with more `easy` questions

**Level 2 Output:**
```
✅ PASS: Dataset siap untuk fine-tuning!

Summary:
  Total: 1500 data points
  Train: 1050 (70.0%)
  Validation: 225 (15.0%)
  Test: 225 (15.0%)
  
  HuggingFace load: ✓ Success
  Stratification: ✓ OK
  Schema validation: ✓ OK
```

---

## Troubleshooting

### Issue: Imbalanced Difficulty

**Symptom:**
```
⚠️ WARNING: Imbalanced difficulty: ratio=4.00x
```

**Solution:**
```python
# Re-run pipeline untuk modul tersebut dengan fokus difficulty yang kurang
run_pipeline(
    section_filter='03-ekspresi',
    difficulties=['easy'],  # Fokus ke easy saja
    max_chunks_per_section=50  # Tambah jumlah
)
```

### Issue: Low Validation Rate

**Symptom:**
```
⚠️ WARNING: Low validation rate: 65.0%
```

**Solution:**
1. Cek `validation_failures.jsonl` di modul tersebut
2. Identifikasi pola error (target format salah, tags kosong, dll)
3. Fix prompt LLM atau re-generate

### Issue: HuggingFace Load Failed

**Symptom:**
```
❌ FAIL: Cannot load dataset with HuggingFace
```

**Solution:**
1. Cek JSONL format (setiap baris valid JSON?)
2. Cek encoding (UTF-8?)
3. Cek schema konsisten (semua entry punya input, target, metadata?)

---

## Best Practices

1. **Run Level 1 after each module generation**
   - Deteksi masalah early
   - Fix sebelum merge

2. **Run Level 2 before fine-tuning**
   - Final check
   - Ensure compatibility

3. **Save validation reports**
   - For documentation
   - For reproducibility

4. **Re-validate after any changes**
   - After re-generation
   - After manual edits
   - After augmentation

---

## Files Generated

```
dataset_aqg/
├── output_modul/
│   ├── 01-berkenalan-dengan-python/
│   │   ├── accumulated.jsonl
│   │   ├── validation_failures.jsonl
│   │   └── module_validation_report.json  ← NEW
│   └── .../
├── dataset-task-spesifc/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── dataset_info.json
│   └── final_validation_report.json  ← NEW
└── validation_summary.png  ← NEW (grafik)
```

---

## Next Steps

After validation passes:
1. ✅ Dataset ready for fine-tuning
2. Proceed to `indot5-finetuning` spec
3. Run domain adaptation first (optional)
4. Run task-specific fine-tuning

If validation fails:
1. ⚠️ Fix issues identified
2. Re-run pipeline for problematic modules
3. Re-validate
4. Repeat until pass
