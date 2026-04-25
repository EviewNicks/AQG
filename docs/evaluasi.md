# 2 Load Model with LoRA 

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


✓ Loaded 211 entries from /content/dataset_aqg/dataset-task-spesifc/test.jsonl

Dataset loaded:
  Train: 876 samples
  Val:   175 samples
  Test:  211 samples

```
# Validate and preview dataset
validation_results = loader.validate_dataset(train_dataset)

sample = train_dataset[0]
print('\n=== Sample Entry ===')
print(f"Input: {sample['input']}...")
print(f"Target: {sample['target']}...")

```

=== Dataset Validation Summary ===
Total Entries: 876
Duplicate Count: 649
Avg Input Length: 707.47 chars
Avg Target Length: 140.21 chars
Has Metadata: False
⚠ Warning: Found 649 duplicate entries

=== Sample Entry ===
Input: ### Perbandingan Penggunaan Memori

```python
import numpy
import sys

var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var_array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Ukuran keseluruhan elemen list dalam bytes =", sys.getsizeof(var_list) * len(var_list))
print("Ukuran keseluruhan elemen NumPy dalam bytes =", var_array.size * var_array.itemsize)

"""
Output:
Ukuran keseluruhan elemen list dalam bytes = 240
Ukuran keseluruhan elemen NumPy dalam bytes = 72
"""
```
Dengan matriks yang sama, NumPy hanya menggunakan **72 bytes** dibanding list Python yang menggunakan **240 bytes** — inilah alasan banyak programmer memilih NumPy untuk memproses matriks. > **Catatan:** Seluruh materi pada modul ini akan menggunakan list Python untuk mengimplementasikan matriks, agar kita memahami fundamental matriks tanpa melibatkan library apa pun....
Target: Sesuai catatan modul yang menggunakan list Python untuk matriks, lengkapi kode berikut untuk menghitung ukuran memori list: import sys; var_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; ukuran_memori = ________________.?...

# 4  Baseline Evaluation ( Pre-Training )


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
  BLEU:     0.0199
  BLEU-1:   0.1240
  BLEU-2:   0.0291
  BLEU-3:   0.0131
  BLEU-4:   0.0033

ROUGE Scores:
  ROUGE-1:  0.1037
  ROUGE-2:  0.0167
  ROUGE-L:  0.0940

Diversity:
  Distinct-1: 0.5016
  Distinct-2: 0.8070

============================================================

Baseline Metrics:
  BLEU-4:  0.0033
  ROUGE-L: 0.0940


# 5 Configure Training

```
from src.finetuned.training.task_trainer import TaskSpecificTrainer

CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/aqg'

# Initialize trainer (all logic in task_trainer.py)
trainer = TaskSpecificTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    output_dir=CHECKPOINT_DIR,
    max_length=512,
    metrics_calculator=metrics_calc
)

print('✓ Trainer initialized')
print(f'  Checkpoints will be saved to: {CHECKPOINT_DIR}')

```

✓ Trainer initialized
  Checkpoints will be saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/aqg

# 6 Start Training

```
import time

start_time = time.time()

print('Starting task-specific AQG training...')
print('='*60)

# Train (all logic in task_trainer.py)
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    early_stopping=True,
    early_stopping_patience=2
)

elapsed = (time.time() - start_time) / 3600
print(f'\n✓ Training completed in {elapsed:.2f} hours')
print(f'  Final training loss: {results["training_loss"]:.4f}')

```


Parameter 'function'=<function TaskSpecificTrainer.preprocess_dataset.<locals>.tokenize_function at 0x7f347561eca0> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
WARNING:datasets.fingerprint:Parameter 'function'=<function TaskSpecificTrainer.preprocess_dataset.<locals>.tokenize_function at 0x7f347561eca0> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
Starting task-specific AQG training...


Starting task-specific AQG training...
============================================================

============================================================
STARTING TASK-SPECIFIC AQG TRAINING
============================================================

Preprocessing datasets...
Preprocessing 876 samples...

✓ Preprocessed 175 samples
  Note: Padding and label masking will be handled by DataCollatorForSeq2Seq

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

=== Final Evaluation Metrics ===
eval_loss: nan
eval_bleu_1: 0.0240
eval_bleu_4: 0.0240
eval_rouge_l: 0.0000
eval_runtime: 101.6246
eval_samples_per_second: 1.7220
eval_steps_per_second: 0.2160
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_results.json

✓ Training completed in 0.19 hours
  Final training loss: 0.0000

# Evaluasi AKhir 

```
import json
from pathlib import Path

# Compare with baseline
comparison = evaluator_final.compare_with_baseline(
    finetuned_metrics=final_metrics,
    baseline_metrics=baseline_metrics
)

# Save evaluation report
Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
report = {
    'baseline_metrics': baseline_metrics,
    'final_metrics': final_metrics,
    'comparison': comparison,
    'training_time_hours': elapsed,
    'model_path': model_path,
    'config': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'epochs': 3,
        'lora_r': 8,
        'lora_alpha': 16
    }
}

with open(f'{EVAL_DIR}/evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Print summary
print('\n' + '='*60)
print('TASK-SPECIFIC AQG TRAINING SUMMARY')
print('='*60)
print(f'Training Time: {elapsed:.2f} hours')
print(f'Model saved: {model_path}')
print(f'\nMetrics Comparison:')
print(f"  BLEU-4:       {baseline_metrics.get('bleu_4',0):.4f} → {final_metrics.get('bleu_4',0):.4f}")
print(f"  ROUGE-L:      {baseline_metrics.get('rouge_l',0):.4f} → {final_metrics.get('rouge_l',0):.4f}")
print(f"  BERTScore F1: {baseline_metrics.get('bertscore_f1',0):.4f} → {final_metrics.get('bertscore_f1',0):.4f}")

bleu_improvement = comparison.get('bleu_4_improvement_pct', 0)
print(f'\nBLEU-4 Improvement: {bleu_improvement:+.1f}%')

if final_metrics.get('bleu_4', 0) >= 0.35:
    print('\n✓ SUCCESS: BLEU-4 target achieved (>= 0.35)')
else:
    print(f"\n⚠ BLEU-4 = {final_metrics.get('bleu_4',0):.4f} (target: >= 0.35)")
    print('  Consider: more epochs, lower lr, or larger dataset')

print('\n✓ Fine-tuning pipeline complete!')
print(f'  Evaluation report: {EVAL_DIR}/evaluation_report.json')
print(f'  Sample outputs: {EVAL_DIR}/sample_outputs.json')

```

============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0336       0.0303       -9.72%
bleu_1                          0.1578       0.1330      -15.69%
bleu_2                          0.0504       0.0388      -22.96%
bleu_3                          0.0198       0.0175      -11.44%
bleu_4                          0.0081       0.0093       15.50%
brevity_penalty                 1.0000       1.0000        0.00%
length_ratio                    1.6752       1.7371        3.70%
rouge_1                         0.1715       0.1638       -4.49%
rouge_2                         0.0573       0.0546       -4.79%
rouge_l                         0.1379       0.1335       -3.22%
rouge_1_fmeasure                0.1715       0.1638       -4.49%
rouge_2_fmeasure                0.0573       0.0546       -4.79%
rouge_l_fmeasure                0.1379       0.1335       -3.22%
distinct_1                      0.5610       0.2068      -63.14%
distinct_2                      0.8646       0.5051      -41.58%

============================================================
TASK-SPECIFIC AQG TRAINING SUMMARY
============================================================
Training Time: 0.19 hours
Model saved: /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/indot5-python-aqg

Metrics Comparison:
  BLEU-4:       0.0081 → 0.0093
  ROUGE-L:      0.1379 → 0.1335
  BERTScore F1: 0.0000 → 0.6383

BLEU-4 Improvement: +15.5%

⚠ BLEU-4 = 0.0093 (target: >= 0.35)
  Consider: more epochs, lower lr, or larger dataset

✓ Fine-tuning pipeline complete!
  Evaluation report: /content/drive/MyDrive/dataset_aqg/evaluation_results/evaluation_report.json
  Sample outputs: /content/drive/MyDrive/dataset_aqg/evaluation_results/sample_outputs.json