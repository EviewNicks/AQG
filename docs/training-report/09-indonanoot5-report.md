# IndonanoT5 fine-tuned D=64 With Dataset V3  No-Code 

Note = letaknya di akun gmail dianysahardi@gmail.com

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=6) ⬆️
Trainable:       ~9.5M params (3.8%) ⬆️
Dataset:         dataset-task-v3/00-dataset/ (5,560 train) ⬆️
Epochs:          10 ⬆️
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   5e-5 ⬇️ (lebih kecil untuk model lebih besar)
Warmup:          100 steps ⬆️

Expected Results:
  BLEU-4:        0.32-0.35 (+23-35%)
  ROUGE-L:       0.52-0.58 (+8-20%)
  Training Time: 6-8 hours



## 1 setup environtment 

Python:  3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
OS:      Linux
Torch:   2.10.0+cu128
CUDA:    True

=== Library Versions ===
  adapters             1.3.0
  transformers         4.57.6
  datasets             4.0.0
  accelerate           1.13.0
  evaluate             0.4.6
  torch                2.10.0+cu128
  tokenizers           0.22.2
  rouge_score          unknown
  bert_score           0.3.12

  cuda version         12.8
  gpu name             Tesla T4

## 2 Load Model with Adapters Layers 

```

from src.finetuned.utils.adapter_loader import load_model_with_adapter, print_adapter_info

# Load model with adapter layers
model, tokenizer = load_model_with_adapter(
    model_name='LazarusNLP/IndoNanoT5-base',
    adapter_name='mcq_generation',
    adapter_config='pfeiffer',
    reduction_factor=6,  # d=128
    device='cuda'
)

# Print detailed info
trainable, total = print_adapter_info(model, tokenizer)

```

✓ Base model loaded with transformers + adapters.init()
✓ Adapter added: pfeiffer config, d=128
✓ Adapter activated for training
✓ Model moved to GPU
  GPU allocated: 1.01 GB

============================================================
MODEL INFORMATION
============================================================

Parameters:
  Trainable: 4,740,096 (1.88%)
  Total:     252,317,952
  Frozen:    247,577,856

Tokenizer:
  Vocab size: 32000
  Pad token:  <pad> (ID: 0)
  EOS token:  </s> (ID: 1)

✓ Loaded 567 entries from /content/dataset_aqg/dataset-task-spesifc/test.jsonl

Dataset loaded:
  Train: 2812 samples
  Val:   566 samples
  Test:  567 samples
✓ Using output field: 'output'

=== Dataset Validation Summary ===
Total Entries: 4529
Duplicate Count: 0
Avg Input Length: 195.65 chars
Avg Target Length: 239.35 chars
Has Metadata: True
✓ No duplicates found

=== Sample Entry ===
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
var_mat = [[10, 20],
           [30, 40],
           [50, 60]]
print(var_mat[0][1] + var_mat[2][1])
```
Kode ini menjumlahkan elemen kolom kedua dari baris pertama dan baris terakhir....
Output: question: Perhatikan kode berikut:
```python
var_mat = [[10, 20],
           [30, 40],
           [50, 60]]
print(var_mat[0][1] + var_mat[2][1])
```
Apa output dari kode tersebut?
answer: 80
distractors: 70 | 90 | 60...

✓ Dataset ready (supports both v2 and v3 formats)

## 4 baseline Evaluation

```

from src.finetuned.evaluation.metrics_calculator import MetricsCalculator
from src.finetuned.evaluation.model_evaluator import ModelEvaluator

metrics_calc = MetricsCalculator()
evaluator = ModelEvaluator(
    model=model,
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

Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0262
  BLEU-1:   0.1068
  BLEU-2:   0.0374
  BLEU-3:   0.0127
  BLEU-4:   0.0092

ROUGE Scores:
  ROUGE-1:  0.1981
  ROUGE-2:  0.0739
  ROUGE-L:  0.1673

Diversity:
  Distinct-1: 0.4169
  Distinct-2: 0.6959

============================================================

Baseline Metrics:
  BLEU-4:  0.0092
  ROUGE-L: 0.1673


## 5 Configure Training

```

from src.finetuned.training.adapter_trainer import AdapterTrainer

CHECKPOINT_DIR = '/content/drive/MyDrive/dataset_aqg/checkpoints/08-indonanoot5-report'

# Initialize trainer
trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc,
    output_dir=CHECKPOINT_DIR,
    max_length=512
)

# Setup training configuration
training_args = trainer.setup_training(
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01
)

print('\n✓ Trainer configured')
print(f'  Checkpoints will be saved to: {CHECKPOINT_DIR}')

```

============================================================
TRAINING CONFIGURATION
============================================================
Epochs: 10
Batch size: 4
Effective batch size: 8
Learning rate: 5e-05
Warmup steps: 100
FP16: True
Gradient checkpointing: True

✓ Trainer configured
  Checkpoints will be saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report

## 6 Start Training

```

# ✅ SIMPLIFIED: Let trainer handle checkpoint detection
resume = True  # Set to False if you want fresh training

# Train - trainer will auto-detect and resume from last checkpoint
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_args=training_args,
    early_stopping_patience=2,
    resume_from_checkpoint=resume
)

```

============================================================
PREPROCESSING DATASETS
============================================================

✓ Datasets tokenized
✓ Data collator configured
✓ Trainer initialized (with transformers 4.46+ compatibility fix)
📂 Found 2 checkpoint(s): ['checkpoint-352', 'checkpoint-1056']
🔄 Resuming from: checkpoint-1056

============================================================
STARTING TRAINING
============================================================
Training with Adapter Layers (d=64, ~3.6% trainable params)
Expected time: 6-8 hours on T4 GPU
Total epochs: 10
============================================================

There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
WARNING:adapters.models.t5.modeling_t5:`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...

![alt text](image/09-image.png)

===========================================================================================

## 7 Save adapter & Visualize 

```

# Save adapter weights
adapter_save_path = trainer.save_adapter(
    adapter_name='mcq_generation',
    save_config={
        "model_name": "LazarusNLP/IndoNanoT5-base",
        "adapter_config": "pfeiffer",
        "reduction_factor": 12,
        "trainable_params": trainable,
        "total_params": total,
        "num_train_epochs": 8,
        "learning_rate": 1e-4,
        "training_time_hours": elapsed
    }
)

# Plot training curves
trainer.plot_training_curves(
    save_path=f'{CHECKPOINT_DIR}/training_curves.png'
)

```

============================================================
SAVING ADAPTER WEIGHTS
============================================================
✓ Adapter weights saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report/adapter_mcq_generation
✓ Tokenizer saved
✓ Config saved
✓ Plot saved to /content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report/training_curves.png

![alt text](image/091-image.png)

##  8 final Evaluation

```
# Re-initialize evaluator with trained model
evaluator_final = ModelEvaluator(
    model=model,
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
  BLEU:     0.1951
  BLEU-1:   0.5504
  BLEU-2:   0.3094
  BLEU-3:   0.1608
  BLEU-4:   0.0939

ROUGE Scores:
  ROUGE-1:  0.5099
  ROUGE-2:  0.2773
  ROUGE-L:  0.4425

BERTScore:
  Precision: 0.7933
  Recall:    0.7760
  F1:        0.7842

Diversity:
  Distinct-1: 0.1507
  Distinct-2: 0.5305

============================================================

=== Evaluation Results ===
bleu: 0.1951
bleu_1: 0.5504
bleu_2: 0.3094
bleu_3: 0.1608
bleu_4: 0.0939
brevity_penalty: 0.8662
length_ratio: 0.8744
rouge_1: 0.5099
rouge_2: 0.2773
rouge_l: 0.4425
rouge_1_fmeasure: 0.5099
rouge_2_fmeasure: 0.2773
rouge_l_fmeasure: 0.4425
bertscore_precision: 0.7933
bertscore_recall: 0.7760
bertscore_f1: 0.7842
distinct_1: 0.1507
distinct_2: 0.5305

## 9 generate sample outputs

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
question: apa yang dimaksud dengan duck typing? answer: dynamic typing lebih erat dengan python dan fokus pada kemampuan object melakukan operasi tertentu distractors: python lebih lambat | python lebih cepat | tidak ada perbedaan

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
question: apa yang dimaksud dengan notebook seperti jupyter atau colab? answer: lingkungan pengembangan interaktif dengan sel-sel yang dapat dijalankan satu per satu distractors: fungsi yang hanya bisa dijalankan sekali | fungsi yang tidak bisa dijalankan

📊 BLEU Score: 0.1329
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
question: apa fungsi numpy.zeros() untuk matriks berisi 0 distractors: membuat matriks dengan nilai default | menghapus matriks | membuat matriks baru

📊 BLEU Score: 0.1267
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
question: apa yang dimaksud dengan method overriding? answer: kemampuan child class untuk memberikan implementasi berbeda dari method di parent class distractors: kemampuan membuat class berjalan lebih cepat | kemampuan membuat method berjalan lebih lambat

📊 BLEU Score: 0.3510
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
question: apa yang dimaksud dengan unpacking? answer: assignment nilai dari list atau tuple ke beberapa variabel sekaligus distractors: hanya satu variabel | tidak ada perbedaan

📊 BLEU Score: 0.3744
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
question: apa fungsi denormalization prosedur? answer: menggabungkan data untuk meningkatkan performa query distractors: membuat query lebih lambat | menghapus query | membuat query menjadi lebih lambat

📊 BLEU Score: 0.1566
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
question: apa yang dimaksud dengan function body? answer: blok kode yang diindentasi dan menentukan apa yang dilakukan fungsi distractors: kode yang tidak bisa dipanggil | kode yang hanya bisa dipanggil

📊 BLEU Score: 0.2800
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
question: apa yang diperlukan untuk mengiterasi setiap elemen dalam perkalian matriks dengan konstanta? answer: nested loop distractors: semua elemen | semua elemen harus sama

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
question: apa yang dimaksud dengan case-sensitive dalam python? answer: nama modul dan package dalam python distractors: nama file dan package | nama file atau package yang harus diimpor

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
question: bagaimana black memformat kode secara konsisten bahkan untuk kode yang sangat kompleks? answer: algoritma yang dapat menangani berbagai pola kode python dan menghasilkan format yang optimal distractors: algoritma tidak dapat menangani semua kode python | algoritma yang tidak bisa menangani kode python

📊 BLEU Score: 0.1622
================================================================================


✓ 20 samples generated and saved
✓ Full output displayed above with BLEU scores

## 10 final summary 

============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0275       0.1951      609.84%
bleu_1                          0.1145       0.5504      380.87%
bleu_2                          0.0404       0.3094      666.04%
bleu_3                          0.0148       0.1608      987.86%
bleu_4                          0.0083       0.0939     1025.46%
brevity_penalty                 1.0000       0.8662      -13.38%
length_ratio                    1.8023       0.8744      -51.48%
rouge_1                         0.2157       0.5099      136.36%
rouge_2                         0.0788       0.2773      252.04%
rouge_l                         0.1832       0.4425      141.59%
rouge_1_fmeasure                0.2157       0.5099      136.36%
rouge_2_fmeasure                0.0788       0.2773      252.04%
rouge_l_fmeasure                0.1832       0.4425      141.59%
distinct_1                      0.3493       0.1507      -56.86%
distinct_2                      0.6407       0.5305      -17.20%

============================================================
ADAPTER-BASED AQG TRAINING SUMMARY
============================================================
Method: Adapter Layers (d=64)
Training Time: 0.01 hours
Trainable: 1.88%

Metrics Comparison:
  BLEU-4:  0.0083 → 0.0939
  ROUGE-L: 0.1832 → 0.4425

BLEU-4 Improvement: +1025.5%

⚠ BLEU-4 = 0.0939 (target: >= 0.20)
  Consider: more epochs or adjust hyperparameters

✓ Fine-tuning pipeline complete!
  Adapter: /content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report/adapter_mcq_generation
  Report: /content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report/evaluation_report.json
  Samples: /content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report/sample_outputs.json

============================================================
HOW TO LOAD TRAINED ADAPTER
============================================================
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

model = AutoAdapterModel.from_pretrained("LazarusNLP/IndoNanoT5-base")
tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
model.load_adapter("/content/drive/MyDrive/dataset_aqg/checkpoints/09-indonanoot5-report/adapter_mcq_generation")
model.set_active_adapters("mcq_generation")

# Generate
inputs = tokenizer("generate_mcq: [CONTEXT]", return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))