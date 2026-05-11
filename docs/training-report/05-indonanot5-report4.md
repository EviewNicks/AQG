# IndoNanoT5 Fine-tued + LoRA with Dataset V3 No Code Blocks   03

Note = letaknya di akun gmail @gmail.com

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


✓ Loaded 168 entries from /content/dataset_aqg/dataset-task-spesifc/test.jsonl

Dataset loaded:
  Train: 1332 samples
  Val:   166 samples
  Test:  168 samples

✓ Using output field: 'output'

✓ Using output field: 'output'

=== Dataset Validation Summary ===
Total Entries: 1332
Duplicate Count: 0
Avg Input Length: 153.71 chars
Avg Target Length: 195.2 chars
Has Metadata: True
✓ No duplicates found

=== Sample Entry ===
Input: buat_soal_pilihan_ganda: Perulangan while adalah indefinite iteration, artinya perulangan berhenti ketika kondisi tertentu terpenuhi. While digunakan ketika jumlah iterasi tidak diketahui sebelumnya....
Output: question: Apa yang dimaksud dengan indefinite iteration?
answer: Perulangan berhenti ketika kondisi terpenuhi
distractors: Perulangan dengan jumlah pasti | Perulangan tanpa kondisi | Perulangan satu kali...

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

Evaluating 10 samples...
  Processed 10/10 samples...
✓ Generated 10 predictions
Computing metrics for 10 samples...
  Computing BLEU...

Computing Diversity...
✓ All metrics computed

Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0476
  BLEU-1:   0.1483
  BLEU-2:   0.0609
  BLEU-3:   0.0342
  BLEU-4:   0.0166

ROUGE Scores:
  ROUGE-1:  0.2214
  ROUGE-2:  0.1031
  ROUGE-L:  0.1784

Diversity:
  Distinct-1: 0.4227
  Distinct-2: 0.7613

============================================================

Baseline Metrics:
  BLEU-4:  0.0166
  ROUGE-L: 0.1784




## 5 Start Training 

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

✓ Preprocessed 794 samples
  Note: Padding and label masking will be handled by DataCollatorForSeq2Seq

=== Training Configuration ===
Epochs: 10
Batch size: 8
Gradient accumulation: 4
Effective batch size: 32
Learning rate: 0.0001
Warmup steps: 50
FP16: True
Train samples: 6356
Eval samples: 794
Metrics: BLEU-4, ROUGE-L

Starting training...

![alt text](image/05-image.png)
![alt text](image.png)


=== Training Complete ===
Final training loss: 38.8484
Training time: 694.08 seconds
Training samples per second: 5.76

=== Final Evaluation Metrics ===
eval_loss: 9.9155
eval_bleu_1: 0.0244
eval_bleu_4: 0.0244
eval_rouge_l: 0.0000
eval_runtime: 70.4592
eval_samples_per_second: 2.3560
eval_steps_per_second: 0.2980
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/aqg/training_results.json

✓ Training completed in 0.21 hours
  Final training loss: 38.8484 


## 7 Final Evaluation 

```
# Re-initialize evaluator with trained model
evaluator_final = ModelEvaluator(
    model=peft_model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc
)

print('Running comprehensive evaluation on test set...')
final_metrics = evaluator_final.evaluate_on_test_set(
    test_dataset=test_dataset,
    num_beams=4,
    include_bertscore=True,
    max_samples=None
)

print('\n=== Evaluation Results ===')
for key, value in final_metrics.items():
    print(f'{key}: {value:.4f}')

```

Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0035
  BLEU-1:   0.0987
  BLEU-2:   0.0032
  BLEU-3:   0.0010
  BLEU-4:   0.0005

ROUGE Scores:
  ROUGE-1:  0.0158
  ROUGE-2:  0.0013
  ROUGE-L:  0.0155

BERTScore:
  Precision: 0.5392
  Recall:    0.5178
  F1:        0.5280

Diversity:
  Distinct-1: 0.0201
  Distinct-2: 0.0892

============================================================

=== Evaluation Results ===
bleu: 0.0035
bleu_1: 0.0987
bleu_2: 0.0032
bleu_3: 0.0010
bleu_4: 0.0005
brevity_penalty: 1.0000
length_ratio: 1.1360
rouge_1: 0.0158
rouge_2: 0.0013
rouge_l: 0.0155
rouge_1_fmeasure: 0.0158
rouge_2_fmeasure: 0.0013
rouge_l_fmeasure: 0.0155
bertscore_precision: 0.5392
bertscore_recall: 0.5178
bertscore_f1: 0.5280
distinct_1: 0.0201
distinct_2: 0.0892


## 9 Generate Smaple Outputs 

Generating 20 sample outputs...

================================================================================
Sample 1/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Duck typing tidak berkaitan langsung dengan dynamic typing atau loosely typed. Konsep duck typing lebih erat dengan pemrograman berorientasi objek (OOP) dan fokus pada kemampuan object melakukan operasi tertentu.

✅ REFERENCE:
question: Dengan konsep apa duck typing lebih erat kaitannya?
answer: Pemrograman berorientasi objek (OOP)
distractors: Dynamic typing | Loosely typed | Static typing

🤖 PREDICTION:
) - - - - - - - : - - - - - -- | -> -1 - - - - - - - - - - - - - -editing () :) – - - - -

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 2/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Notebook seperti Jupyter atau Colab menyediakan lingkungan pengembangan interaktif dengan sel-sel yang dapat dijalankan satu per satu.

✅ REFERENCE:
question: Apa keunggulan sistem sel pada Notebook?
answer: Dapat menjalankan kode satu per satu
distractors: Lebih cepat dari script | Tidak perlu Python | Otomatis menyimpan

🤖 PREDICTION:
- - - - – - - -- : -) - - - - - -> - -: + - :: – | = - - - - - - - - - -. - - - - - 

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 3/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Dalam NumPy, kita dapat membuat matriks dengan nilai default menggunakan fungsi numpy.zeros() untuk matriks berisi 0, atau numpy.ones() untuk matriks berisi 1.

✅ REFERENCE:
question: Fungsi NumPy apa yang digunakan untuk membuat matriks berisi nilai 0?
answer: numpy.zeros()
distractors: numpy.empty() | numpy.zero() | numpy.fill(0)

🤖 PREDICTION:
1 - 1 – 1, - atau 1.1 | 10 -: 1 atau 2 -, | atau : -) -. - - - - - - - - - - - - - - - - - - - - - - -

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 4/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Method overriding adalah kemampuan child class untuk memberikan implementasi berbeda dari method yang diwarisi dari parent class. Method di child class harus memiliki nama dan signature yang sama dengan method di parent class.

✅ REFERENCE:
question: Apa yang dimaksud dengan method overriding?
answer: Child class memberikan implementasi berbeda dari method yang diwarisi dari parent class
distractors: Menghapus method dari parent class | Membuat method baru di parent class | Mengubah nama method

🤖 PREDICTION:
- - - - - - - – - - - - : -) -> -- | - -: + -= -1 - - yang - - - - - - - - - - - - - - -

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 5/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Unpacking memungkinkan assignment nilai dari list atau tuple ke beberapa variabel sekaligus.

✅ REFERENCE:
question: Apa yang dimaksud dengan unpacking?
answer: Assignment nilai dari list/tuple ke beberapa variabel
distractors: Menghapus variabel | Menggabungkan variabel | Membuat list

🤖 PREDICTION:
- - - - - - | - - - - - : -) -> -- – - -= -: –- || | - -. - - - - - - - - - - - - - 

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 6/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Prosedur dapat digunakan untuk denormalization, yaitu menggabungkan data untuk meningkatkan performa query. Denormalization prosedur trade-off antara storage dan speed.

✅ REFERENCE:
question: Apa trade-off dari denormalization prosedur?
answer: Antara storage dan speed
distractors: Antara security dan usability | Tidak ada trade-off | Antara size dan color

🤖 PREDICTION:
. | - : - - - - - -) - - - - - -- – -> -1 -2 - yang - :) 1. 2. - - - - - - - - - - - -  | 

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 7/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Function body adalah blok kode yang diindentasi dan menentukan apa yang dilakukan fungsi. Body berisi instruksi-instruksi yang akan dieksekusi ketika fungsi dipanggil.

✅ REFERENCE:
question: Apa yang dimaksud dengan function body?
answer: Blok kode yang diindentasi berisi instruksi yang dieksekusi
distractors: Nama fungsi yang digunakan | Nilai yang dikembalikan fungsi | Parameter yang diterima fungsi

🤖 PREDICTION:
- : - - - - - -editing - - - - - -- – -> -) - | -| –- :- | - - - - - - - - - - - - - - -

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 8/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Dalam implementasi perkalian matriks dengan konstanta menggunakan list Python, kita memerlukan nested loop untuk mengiterasi setiap elemen. Loop pertama untuk baris dan loop kedua untuk kolom.

✅ REFERENCE:
question: Mengapa kita memerlukan nested loop (dua perulangan) untuk mengalikan matriks dengan konstanta menggunakan list Python?
answer: Karena matriks adalah struktur 2 dimensi, sehingga perlu loop untuk baris dan loop untuk kolom
distractors: Karena Python tidak mendukung operasi langsung pada list 2D | Karena satu loop hanya bisa memproses maksimal 10 elemen | Karena nested loop membuat kode lebih cepat dieksekusi

🤖 PREDICTION:
- – - - - - - - - - - : -) -> - -- + -1 - -= -2 - atau -3 - - -_ - - - - - - - - - - |   -

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 9/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Case-sensitive juga berlaku untuk nama modul dan package dalam Python. Import statement harus menggunakan kapitalisasi yang tepat sesuai dengan nama file atau package.

✅ REFERENCE:
question: Apakah case-sensitive berlaku untuk nama modul?
answer: Ya, harus sesuai dengan nama file
distractors: Tidak, bisa menggunakan kapitalisasi apa saja | Hanya untuk variabel | Hanya untuk fungsi

🤖 PREDICTION:
- - - - - - – | -editing -) -. - - - - - : -d -- –- | -: – - -> - - - - - - - - - - - |  

📊 BLEU Score: 0.0000
================================================================================

================================================================================
Sample 10/20
================================================================================

📥 INPUT:
buat_soal_pilihan_ganda: Ketika menggunakan black, formatter akan memformat kode secara konsisten bahkan untuk kode yang sangat kompleks. Black menggunakan algoritma yang dapat menangani berbagai pola kode Python dan menghasilkan format yang optimal.

✅ REFERENCE:
question: Apakah black dapat memformat kode Python yang sangat kompleks dengan benar?
answer: Ya, black menggunakan algoritma yang dapat menangani berbagai pola kode Python yang kompleks
distractors: Tidak, black hanya efektif untuk kode Python yang sederhana | Ya, tetapi hanya jika kode tidak menggunakan fitur Python lanjutan | Tidak, kode yang kompleks perlu diformat secara manual

🤖 PREDICTION:
| - - - - - - - - : -) - - - --> ->> – -_- > —- –> |- :: –- || |) | - - - - - - - - -   

📊 BLEU Score: 0.0000
================================================================================


✓ Samples saved to /content/drive/MyDrive/dataset_aqg/evaluation_results/05-indonanoot5-report/sample_outputs.json
✓ 20 sample outputs generated and saved
✓ Full output displayed above with BLEU scores



## 10 Final Summary 

============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0474       0.0035      -92.51%
bleu_1                          0.1769       0.0987      -44.21%
bleu_2                          0.0654       0.0032      -95.11%
bleu_3                          0.0309       0.0010      -96.68%
bleu_4                          0.0141       0.0005      -96.53%
brevity_penalty                 1.0000       1.0000        0.00%
length_ratio                    1.9112       1.1360      -40.56%
rouge_1                         0.2555       0.0158      -93.80%
rouge_2                         0.0978       0.0013      -98.71%
rouge_l                         0.1976       0.0155      -92.15%
rouge_1_fmeasure                0.2555       0.0158      -93.80%
rouge_2_fmeasure                0.0978       0.0013      -98.71%
rouge_l_fmeasure                0.1976       0.0155      -92.15%
distinct_1                      0.3672       0.0201      -94.54%
distinct_2                      0.7241       0.0892      -87.68%

============================================================
TASK-SPECIFIC AQG TRAINING SUMMARY
============================================================
Training Time: 0.33 hours
Model saved: /content/drive/MyDrive/dataset_aqg/checkpoints/05-indonanoot5-report/indot5-python-aqg

Metrics Comparison:
  BLEU-4:       0.0141 → 0.0005
  ROUGE-L:      0.1976 → 0.0155
  BERTScore F1: 0.0000 → 0.5280

BLEU-4 Improvement: -96.5%

⚠ BLEU-4 = 0.0005 (target: >= 0.35)
  Consider: more epochs, lower lr, or larger dataset

✓ Fine-tuning pipeline complete!
  Evaluation report: /content/drive/MyDrive/dataset_aqg/evaluation_results/05-indonanoot5-report/evaluation_report.json
  Sample outputs: /content/drive/MyDrive/dataset_aqg/evaluation_results/05-indonanoot5-report/sample_outputs.json