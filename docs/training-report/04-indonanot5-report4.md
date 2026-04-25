# IndoNanoT5 Fine-tued + LoRA with Dataset V3 

## 1 Setup Environtment 

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

## 2 Load Model With LoRA 

```

from src.finetuned.utils.model_loader import load_model_with_lora, print_model_info

# Load model with LoRA - UPDATED: Using IndoT5 (580M params) instead of IndoNanoT5 (248M)
# IndoNanoT5 was insufficient for complex AQG task
peft_model, tokenizer = load_model_with_lora(
    model_name='LazarusNLP/IndoNanoT5-base',  
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q', 'v']
)

# Print detailed info
print_model_info(peft_model, tokenizer)

```

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

## 3 Load Dataset 

```

from src.finetuned.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
TASK_DIR = '/content/dataset_aqg/dataset-task-spesifc/'

# Copy dataset from Drive if needed
if not os.path.exists(TASK_DIR + 'train.jsonl'):
    drive_task = f'{DRIVE_ROOT}/dataset-task-spesifc'
    os.makedirs(TASK_DIR, exist_ok=True)
    for f in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
        shutil.copy(f'{drive_task}/{f}', f'{TASK_DIR}{f}')
    print('✓ Dataset copied from Drive')

# Load datasets
train_dataset = loader.load_dataset(TASK_DIR, split='train')
val_dataset = loader.load_dataset(TASK_DIR, split='validation')
test_dataset = loader.load_dataset(TASK_DIR, split='test')

print(f'\nDataset loaded:')
print(f'  Train: {len(train_dataset)} samples')
print(f'  Val:   {len(val_dataset)} samples')
print(f'  Test:  {len(test_dataset)} samples')

```


✓ Loaded 319 entries from /content/dataset_aqg/dataset-task-spesifc/test.jsonl

Dataset loaded:
  Train: 2544 samples
  Val:   318 samples
  Test:  319 samples

✓ Using output field: 'output'

=== Dataset Validation Summary ===
Total Entries: 2544
Duplicate Count: 0
Avg Input Length: 167.59 chars
Avg Target Length: 215.58 chars
Has Metadata: True
✓ No duplicates found

=== Sample Entry ===
Input: buat_soal_pilihan_ganda: One-liner menggunakan simultaneous assignment yang memungkinkan Python mengevaluasi semua nilai di sisi kanan sebelum melakukan assignment. Ini membuat pertukaran nilai menjadi sangat efisien....
Output: question: Apa yang dimaksud dengan simultaneous assignment?
answer: Mengevaluasi semua nilai di sisi kanan sebelum assignment
distractors: Assignment satu per satu | Assignment acak | Assignment berurutan...

✓ Metadata dropped
  Columns: ['input', 'output', 'metadata']
  Train: 2544 | Val: 318 | Test: 319

## 4 baseline Evaluation ( Pre-Training )

```

from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
from src.finetuned.evaluation.model_evaluator import ModelEvaluator

metrics_calc = MetricsCalculator()
evaluator = ModelEvaluator(
    model=peft_model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc
)

print('Computing baseline metrics (10 samples)...')
baseline_metrics = evaluator.evaluate_on_test_set(
    test_dataset=val_dataset,
    num_beams=4,
    include_bertscore=False,
    max_samples=10
)

print(f"\nBaseline Metrics:")
print(f"  BLEU-4:  {baseline_metrics.get('bleu_4', 0):.4f}")
print(f"  ROUGE-L: {baseline_metrics.get('rouge_l', 0):.4f}")

```

Computing baseline metrics (10 samples)...

============================================================
EVALUATING ON TEST SET
============================================================

The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
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
  BLEU:     0.0459
  BLEU-1:   0.1410
  BLEU-2:   0.0515
  BLEU-3:   0.0319
  BLEU-4:   0.0191

ROUGE Scores:
  ROUGE-1:  0.1504
  ROUGE-2:  0.0636
  ROUGE-L:  0.1293

Diversity:
  Distinct-1: 0.3591
  Distinct-2: 0.6799

============================================================

Baseline Metrics:
  BLEU-4:  0.0191
  ROUGE-L: 0.1293

## 5 Start Training 


Starting task-specific AQG training...
============================================================

============================================================
STARTING TASK-SPECIFIC AQG TRAINING
============================================================

Preprocessing datasets...
Preprocessing 2544 samples...

✓ Preprocessed 318 samples
  Note: Padding and label masking will be handled by DataCollatorForSeq2Seq

=== Training Configuration ===
Epochs: 3
Batch size: 8
Gradient accumulation: 4
Effective batch size: 32
Learning rate: 0.0001
Warmup steps: 50
FP16: True
Train samples: 2544
Eval samples: 318
Metrics: BLEU-4, ROUGE-L

Starting training...

![alt text](image/image.png)

=== Training Complete ===
Final training loss: 37.1673
Training time: 2187.78 seconds
Training samples per second: 3.49

=== Final Evaluation Metrics ===
eval_loss: 9.3815
eval_bleu_1: 0.0077
eval_bleu_4: 0.0077
eval_rouge_l: 0.0000
eval_runtime: 583.3445
eval_samples_per_second: 0.5450
eval_steps_per_second: 0.0690
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_results.json

✓ Training completed in 0.77 hours
  Final training loss: 37.1673

##  Save Model 

✓ Final model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/indot5-python-aqg
✓ Model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/indot5-python-aqg
✓ Training curves saved to /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_curves.png

## 8 Final Evaluation

BertModel LOAD REPORT from: bert-base-multilingual-cased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0405
  BLEU-1:   0.2147
  BLEU-2:   0.0641
  BLEU-3:   0.0265
  BLEU-4:   0.0074

ROUGE Scores:
  ROUGE-1:  0.0760
  ROUGE-2:  0.0211
  ROUGE-L:  0.0647

BERTScore:
  Precision: 0.6135
  Recall:    0.5685
  F1:        0.5897

Diversity:
  Distinct-1: 0.0735
  Distinct-2: 0.4171

============================================================

=== Evaluation Results ===
bleu: 0.0405
bleu_1: 0.2147
bleu_2: 0.0641
bleu_3: 0.0265
bleu_4: 0.0074
brevity_penalty: 1.0000
length_ratio: 1.0326
rouge_1: 0.0760
rouge_2: 0.0211
rouge_l: 0.0647
rouge_1_fmeasure: 0.0760
rouge_2_fmeasure: 0.0211
rouge_l_fmeasure: 0.0647
bertscore_precision: 0.6135
bertscore_recall: 0.5685
bertscore_f1: 0.5897
distinct_1: 0.0735
distinct_2: 0.4171

## 9 Generate Smaple Outputs 

Generating 5 sample outputs...

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]
for a, b, c in zip(list1, list2, lis...
Reference: question: Perhatikan kode berikut:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]
for a, b, c in zip(list1, list2, list3):
    print(...
Prediction: `'`` ` ` ` `, 2, 3, 4, 5, 6, 7, 9, 10, 8, 0 1, (2,), - 9, # 3 = [3, b, b + b, " 2. 4. 5 1] 1 1. 3. 2, c =, '...
BLEU: 0.1550

--- Sample 2 ---
Input: buat_soal_pilihan_ganda: Dalam pemrograman, abstraksi data adalah jembatan menuju pemahaman yang lebih tinggi tentang arsitektur perangkat lunak....
Reference: question: Abstraksi data dianggap sebagai jembatan menuju pemahaman...
answer: Arsitektur perangkat lunak
distractors: Arsitektur bangunan fisik | Ars...
Prediction: ) :   -  =    ,   :)   =>   ::  =: 1 == 2  1: 3 - 3: 4 - 5  : 1,   : 3,  5, : 5:  :   ...
BLEU: 0.0000


--- Sample 4 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
for i in range(3):
    try:
        if i == 1:
            raise ValueError("Skip")
      ...
Reference: question: Perhatikan kode berikut:
```python
for i in range(3):
    try:
        if i == 1:
            raise ValueError("Skip")
        print(i)
    ...
Prediction: `''' ` `` ` ` ` ` ` `' " `" "): '` (") ` `'` ` ') '' = '' - '' –''''''''  ...
BLEU: 0.0000
'''

## 10 Final Summary 


============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0459       0.0405      -11.87%
bleu_1                          0.1410       0.2147       52.21%
bleu_2                          0.0515       0.0641       24.45%
bleu_3                          0.0319       0.0265      -17.16%
bleu_4                          0.0191       0.0074      -61.57%
brevity_penalty                 1.0000       1.0000        0.00%
length_ratio                    1.3635       1.0326      -24.27%
rouge_1                         0.1504       0.0760      -49.45%
rouge_2                         0.0636       0.0211      -66.84%
rouge_l                         0.1293       0.0647      -49.99%
rouge_1_fmeasure                0.1504       0.0760      -49.45%
rouge_2_fmeasure                0.0636       0.0211      -66.84%
rouge_l_fmeasure                0.1293       0.0647      -49.99%
distinct_1                      0.3591       0.0735      -79.53%
distinct_2                      0.6799       0.4171      -38.66%

============================================================
TASK-SPECIFIC AQG TRAINING SUMMARY
============================================================
Training Time: 0.77 hours
Model saved: /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/indot5-python-aqg

Metrics Comparison:
  BLEU-4:       0.0191 → 0.0074
  ROUGE-L:      0.1293 → 0.0647
  BERTScore F1: 0.0000 → 0.5897

BLEU-4 Improvement: -61.6%

⚠ BLEU-4 = 0.0074 (target: >= 0.35)
  Consider: more epochs, lower lr, or larger dataset

✓ Fine-tuning pipeline complete!
  Evaluation report: /content/drive/MyDrive/dataset_aqg/evaluation_results/evaluation_report.json
  Sample outputs: /content/drive/MyDrive/dataset_aqg/evaluation_results/sample_outputs.json