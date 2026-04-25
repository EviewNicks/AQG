
# 1 Setup Environtment 

Python:  3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
OS:      Linux
Torch:   2.10.0+cu128
CUDA:    True

 
=== Library Versions ===
  transformers         5.0.0
  peft                 0.18.1
  datasets             4.0.0
  accelerate           1.13.0
  evaluate             0.4.6
  torch                2.10.0+cu128
  tokenizers           0.22.2
  rouge_score          unknown
  bert_score           0.3.12

  python               3.12.13
  cuda available       True
  cuda version         12.8
  gpu name             Tesla T4


# 2 Load Model LoRA 

/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:103: UserWarning: 
Error while fetching `HF_TOKEN` secret value from your vault: 'Requesting secret HF_TOKEN timed out. Secrets can only be fetched when running from the Colab UI.'.
You are not authenticated with the Hugging Face Hub in this notebook.
If the error persists, please let us know by opening an issue on GitHub (https://github.com/huggingface/huggingface_hub/issues/new).
  warnings.warn(
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

✓ Base model loaded
✓ LoRA applied: r=8, alpha=16, target=['q', 'v']
  Trainable: 884,736 (0.36%)
  Total:     248,462,592
✓ Model device: cuda:0
  GPU allocated: 1.00 GB

=== Model Information ===
Model type: PeftModelForSeq2SeqLM
Tokenizer: T5Tokenizer
Vocab size: 32000
Pad token: <pad> (ID: 0)
EOS token: </s> (ID: 1)

Parameters:
  Total: 248,462,592
  Trainable: 884,736 (0.36%)
  Frozen: 247,577,856

# 3 Load Dataset
✓ Dataset copied from Drive
✓ Loaded 876 entries from /content/dataset_aqg/dataset-task-spesifc/train.jsonl
✓ Loaded 175 entries from /content/dataset_aqg/dataset-task-spesifc/validation.jsonl

✓ Loaded 211 entries from /content/dataset_aqg/dataset-task-spesifc/test.jsonl

Dataset loaded:
  Train: 876 samples
  Val:   175 samples
  Test:  211 samples

=== Dataset Validation Summary ===
Total Entries: 876
Duplicate Count: 1
Avg Input Length: 821.26 chars
Avg Target Length: 343.94 chars
Has Metadata: True
⚠ Warning: Found 1 duplicate entries

=== Sample Entry ===
Input: Konteks: ### Perbandingan Penggunaan Memori

```python
import numpy
import sys

var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Ukuran k...
Target: Pertanyaan: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran...
```

✓ Metadata dropped
  Columns: ['input', 'target']
  Train: 876 | Val: 175 | Test: 211

# 4 Baselie Evaluation

Computing baseline metrics (10 samples)...

============================================================
EVALUATING ON TEST SET
============================================================

Evaluating 10 samples...
  Processed 10/10 samples...
✓ Generated 10 predictions
Computing metrics for 10 samples...
  Computing BLEU...

Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0141
  BLEU-1:   0.1928
  BLEU-2:   0.0299
  BLEU-3:   0.0084
  BLEU-4:   0.0034

ROUGE Scores:
  ROUGE-1:  0.0996
  ROUGE-2:  0.0270
  ROUGE-L:  0.0707

Diversity:
  Distinct-1: 0.4048
  Distinct-2: 0.6855

============================================================

Baseline Metrics:
  BLEU-4:  0.0034
  ROUGE-L: 0.0707

# 6  Start Training

Starting task-specific AQG training...
============================================================

============================================================
STARTING TASK-SPECIFIC AQG TRAINING
============================================================

Preprocessing datasets...
Preprocessing 876 samples...

✓ Preprocessed 175 samples

=== Training Configuration ===
Epochs: 3
Batch size: 8
Gradient accumulation: 4
Effective batch size: 32
Learning rate: 0.0001
Warmup steps: 50
FP16: True
Train samples: 876
Eval samples: 175
Metrics: BLEU-4, ROUGE-L

Starting training...
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(

/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(

=== Training Complete ===
Final training loss: 38.9012
Training time: 1066.43 seconds
Training samples per second: 2.46
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/transformers/tokenization_utils_base.py:2402: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(

=== Final Evaluation Metrics ===
eval_loss: 9.7079
eval_bleu_1: 0.0049
eval_bleu_4: 0.0049
eval_rouge_l: 0.0000
eval_runtime: 121.3881
eval_samples_per_second: 1.4420
eval_steps_per_second: 0.1810
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_results.json

✓ Training completed in 0.33 hours
  Final training loss: 38.9012