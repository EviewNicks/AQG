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
  ROUGE-1:  0.5097
  ROUGE-2:  0.2778
  ROUGE-L:  0.4431

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
rouge_1: 0.5097
rouge_2: 0.2778
rouge_l: 0.4431
rouge_1_fmeasure: 0.5097
rouge_2_fmeasure: 0.2778
rouge_l_fmeasure: 0.4431
bertscore_precision: 0.7933
bertscore_recall: 0.7760
bertscore_f1: 0.7842
distinct_1: 0.1507
distinct_2: 0.5305

## 9 generate sample outputs

Generating 20 sample outputs...

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Duck typing tidak berkaitan langsung dengan dynamic typing atau loosely typed. Konsep duck typing lebih erat dengan pemrogram...
Reference: question: Dengan konsep apa duck typing lebih erat kaitannya?
answer: Pemrograman berorientasi objek (OOP)
distractors: Dynamic typing | Loosely typed...
Prediction: question: apa yang dimaksud dengan duck typing? answer: dynamic typing lebih erat dengan python dan fokus pada kemampuan object melakukan operasi tert...
BLEU: 0.0000

--- Sample 2 ---
Input: buat_soal_pilihan_ganda: Notebook seperti Jupyter atau Colab menyediakan lingkungan pengembangan interaktif dengan sel-sel yang dapat dijalankan satu ...
Reference: question: Apa keunggulan sistem sel pada Notebook?
answer: Dapat menjalankan kode satu per satu
distractors: Lebih cepat dari script | Tidak perlu Pyt...
Prediction: question: apa yang dimaksud dengan notebook seperti jupyter atau colab? answer: lingkungan pengembangan interaktif dengan sel-sel yang dapat dijalanka...
BLEU: 0.1329

--- Sample 3 ---
Input: buat_soal_pilihan_ganda: Dalam NumPy, kita dapat membuat matriks dengan nilai default menggunakan fungsi numpy.zeros() untuk matriks berisi 0, atau nu...
Reference: question: Fungsi NumPy apa yang digunakan untuk membuat matriks berisi nilai 0?
answer: numpy.zeros()
distractors: numpy.empty() | numpy.zero() | nump...
Prediction: question: apa fungsi numpy.zeros() untuk matriks berisi 0 distractors: membuat matriks dengan nilai default | menghapus matriks | membuat matriks baru...
BLEU: 0.1267

--- Sample 4 ---
Input: buat_soal_pilihan_ganda: Method overriding adalah kemampuan child class untuk memberikan implementasi berbeda dari method yang diwarisi dari parent cl...
Reference: question: Apa yang dimaksud dengan method overriding?
answer: Child class memberikan implementasi berbeda dari method yang diwarisi dari parent class
...
Prediction: question: apa yang dimaksud dengan method overriding? answer: kemampuan child class untuk memberikan implementasi berbeda dari method di parent class ...
BLEU: 0.3510

--- Sample 5 ---
Input: buat_soal_pilihan_ganda: Unpacking memungkinkan assignment nilai dari list atau tuple ke beberapa variabel sekaligus....
Reference: question: Apa yang dimaksud dengan unpacking?
answer: Assignment nilai dari list/tuple ke beberapa variabel
distractors: Menghapus variabel | Menggabu...
Prediction: question: apa yang dimaksud dengan unpacking? answer: assignment nilai dari list atau tuple ke beberapa variabel sekaligus distractors: hanya satu var...
BLEU: 0.3744

--- Sample 6 ---
Input: buat_soal_pilihan_ganda: Prosedur dapat digunakan untuk denormalization, yaitu menggabungkan data untuk meningkatkan performa query. Denormalization p...
Reference: question: Apa trade-off dari denormalization prosedur?
answer: Antara storage dan speed
distractors: Antara security dan usability | Tidak ada trade-o...
Prediction: question: apa fungsi denormalization prosedur? answer: menggabungkan data untuk meningkatkan performa query distractors: membuat query lebih lambat | ...
BLEU: 0.1566

--- Sample 7 ---
Input: buat_soal_pilihan_ganda: Function body adalah blok kode yang diindentasi dan menentukan apa yang dilakukan fungsi. Body berisi instruksi-instruksi yan...
Reference: question: Apa yang dimaksud dengan function body?
answer: Blok kode yang diindentasi berisi instruksi yang dieksekusi
distractors: Nama fungsi yang di...
Prediction: question: apa yang dimaksud dengan function body? answer: blok kode yang diindentasi dan menentukan apa yang dilakukan fungsi distractors: kode yang t...
BLEU: 0.2800

--- Sample 8 ---
Input: buat_soal_pilihan_ganda: Dalam implementasi perkalian matriks dengan konstanta menggunakan list Python, kita memerlukan nested loop untuk mengiterasi ...
Reference: question: Mengapa kita memerlukan nested loop (dua perulangan) untuk mengalikan matriks dengan konstanta menggunakan list Python?
answer: Karena matri...
Prediction: question: apa yang diperlukan untuk mengiterasi setiap elemen dalam perkalian matriks dengan konstanta? answer: nested loop distractors: semua elemen ...
BLEU: 0.0000

--- Sample 9 ---
Input: buat_soal_pilihan_ganda: Case-sensitive juga berlaku untuk nama modul dan package dalam Python. Import statement harus menggunakan kapitalisasi yang t...
Reference: question: Apakah case-sensitive berlaku untuk nama modul?
answer: Ya, harus sesuai dengan nama file
distractors: Tidak, bisa menggunakan kapitalisasi ...
Prediction: question: apa yang dimaksud dengan case-sensitive dalam python? answer: nama modul dan package dalam python distractors: nama file dan package | nama ...
BLEU: 0.0000

--- Sample 10 ---
Input: buat_soal_pilihan_ganda: Ketika menggunakan black, formatter akan memformat kode secara konsisten bahkan untuk kode yang sangat kompleks. Black menggu...
Reference: question: Apakah black dapat memformat kode Python yang sangat kompleks dengan benar?
answer: Ya, black menggunakan algoritma yang dapat menangani ber...
Prediction: question: bagaimana black memformat kode secara konsisten bahkan untuk kode yang sangat kompleks? answer: algoritma yang dapat menangani berbagai pola...
BLEU: 0.1622


✓ Samples saved to /content/drive/MyDrive/dataset_aqg/evaluation_results/09-indonanoot5-report/sample_outputs.json
✓ 20 samples generated

## 10 final summary 

============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0262       0.1951      645.19%
bleu_1                          0.1068       0.5504      415.24%
bleu_2                          0.0374       0.3094      726.50%
bleu_3                          0.0127       0.1608     1165.47%
bleu_4                          0.0092       0.0939      916.48%
brevity_penalty                 1.0000       0.8662      -13.38%
length_ratio                    1.6361       0.8744      -46.56%
rouge_1                         0.1981       0.5097      157.33%
rouge_2                         0.0739       0.2778      275.71%
rouge_l                         0.1673       0.4431      164.83%
rouge_1_fmeasure                0.1981       0.5097      157.33%
rouge_2_fmeasure                0.0739       0.2778      275.71%
rouge_l_fmeasure                0.1673       0.4431      164.83%
distinct_1                      0.4169       0.1507      -63.86%
distinct_2                      0.6959       0.5305      -23.77%