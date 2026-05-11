# Dataset Analysis Report

**File:** `dataset_aqg\dataset-task-v4\01-perkenalan-python\01-perkenalan-python.jsonl`  
**Date:** 2026-05-01 10:02  
**Target:** 220 samples per materi

---

## Summary

| Metric | Value |
|--------|-------|
| Total samples | 110 |
| Valid | 15 |
| Invalid (to remove) | 95 |
| Status | ❌ Need 205 more samples |

## Type Distribution (auto-detected)

| Type | Count | % | Status |
|------|-------|---|--------|
| knowledge | 15 | 100% | ✅ |
| code | 0 | 0% | ✅ |

## Difficulty Distribution

| Difficulty | Count | % |
|------------|-------|---|
| Mudah | 9 | 60% |
| Sedang | 6 | 40% |
| Sulit | 0 | 0% |

## Issues (95 samples to remove)

| Line | Reason |
|------|--------|
| 16 | Duplicate input |
| 17 | Duplicate input |
| 18 | Duplicate input |
| 19 | Duplicate input |
| 20 | Duplicate input |
| 21 | Duplicate input |
| 22 | Duplicate input |
| 23 | Duplicate input |
| 24 | Duplicate input |
| 25 | Duplicate input |
| 26 | Duplicate input |
| 27 | Duplicate input |
| 28 | Duplicate input |
| 29 | Duplicate input |
| 30 | Duplicate input |
| 31 | Duplicate input |
| 32 | Duplicate input |
| 33 | Duplicate input |
| 34 | Duplicate input |
| 35 | Duplicate input |
| 36 | Duplicate input |
| 37 | Duplicate input |
| 38 | Duplicate input |
| 39 | Duplicate input |
| 40 | Duplicate input |
| 41 | Duplicate input |
| 42 | Duplicate input |
| 43 | Duplicate input |
| 44 | Duplicate input |
| 45 | Duplicate input |
| 46 | Duplicate input |
| 47 | Duplicate input |
| 48 | Duplicate input |
| 49 | Duplicate input |
| 50 | Duplicate input |
| 51 | Duplicate input |
| 52 | Duplicate input |
| 53 | Duplicate input |
| 54 | Duplicate input |
| 55 | Duplicate input |
| 56 | Duplicate input |
| 57 | Duplicate input |
| 58 | Duplicate input |
| 59 | Duplicate input |
| 60 | Duplicate input |
| 61 | Duplicate input |
| 62 | Duplicate input |
| 63 | Duplicate input |
| 64 | Duplicate input |
| 65 | Duplicate input |
| 66 | Duplicate input |
| 67 | Duplicate input |
| 68 | Duplicate input |
| 69 | Duplicate input |
| 70 | Duplicate input |
| 71 | Duplicate input |
| 72 | Duplicate input |
| 73 | Duplicate input |
| 74 | Duplicate input |
| 75 | Duplicate input |
| 76 | Duplicate input |
| 77 | Duplicate input |
| 78 | Duplicate input |
| 79 | Duplicate input |
| 80 | Duplicate input |
| 81 | Duplicate input |
| 82 | Duplicate input |
| 83 | Duplicate input |
| 84 | Duplicate input |
| 85 | Duplicate input |
| 86 | Duplicate input |
| 87 | Duplicate input |
| 88 | Duplicate input |
| 89 | Duplicate input |
| 90 | Duplicate input |
| 91 | Duplicate input |
| 92 | Duplicate input |
| 93 | Duplicate input |
| 94 | Duplicate input |
| 95 | Duplicate input |
| 96 | Duplicate input |
| 97 | Duplicate input |
| 98 | Duplicate input |
| 99 | Duplicate input |
| 100 | Duplicate input |
| 101 | Duplicate input |
| 102 | Duplicate input |
| 103 | Duplicate input |
| 104 | Duplicate input |
| 105 | Duplicate input |
| 106 | Duplicate input |
| 107 | Duplicate input |
| 108 | Duplicate input |
| 109 | Duplicate input |
| 110 | Duplicate input |

## Next Steps

1. **Cleanup** — remove 95 invalid samples:
   ```
   python scripts/03-dataset-design/02_clean_dataset.py dataset_aqg\dataset-task-v4\01-perkenalan-python\01-perkenalan-python.jsonl
   ```
2. **Generate** — create 205 new samples to reach target of 220
