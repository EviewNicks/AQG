# 1. Eksperiment Pertama Parameter LoRA 

Dataset :

✓ Metadata dropped
  Columns: ['input', 'output', 'metadata']
  Train: 1332 | Val: 166 | Test: 168


 peft_model, tokenizer = load_model_with_lora(
    model_name='LazarusNLP/IndoNanoT5-base',  
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q', 'v']
)

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
  Total: 

Jika pelru detail coba lihat : src\finetuned\notebooks\03_task_specific_training_v2.ipynb 


# 2 Eksperimen Keduan Adapter 

## Dataset Loaded V3 

Dataset loaded Versi Completed  :
  Train: 4529 samples
  Val:   566 samples
  Test:  566 samples
✓ Using output field: 'output'

Dataset loaded Versi No Code :
  Train: 2812 samples
  Val:   351 samples
  Test:  352 samples
✓ Using output field: 'output'

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=64 (reduction_factor=12)
Trainable:       2.38M params (0.95%)
Dataset:         dataset-task-spesifc/ (4,529 train)
Epochs:          8
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   1e-4
Warmup:          50 steps

# 3 Eksperimen Ketiga Adapter 

## Dataset Loaded V3 

Dataset loaded Versi Completed  :
  Train: 4529 samples
  Val:   566 samples
  Test:  566 samples
✓ Using output field: 'output'

Dataset loaded Versi No Code :
  Train: 2812 samples
  Val:   351 samples
  Test:  352 samples
✓ Using output field: 'output'

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=6) ⬆️
Trainable:       ~9.5M params (3.8%) ⬆️
Dataset:         dataset-task-v3/00-dataset/ (5,560 train) ⬆️
Epochs:          10 ⬆️
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   5e-5 ⬇️ (lebih kecil untuk model lebih besar)
Warmup:          100 steps ⬆️


# 4 Eksperimen Keempat Adapter 

## Dataset Loaded V4  


Dataset loaded Versi Completed  :
  Train: 8680 samples
  Val:   1085 samples
  Test:  1086 samples
✓ Using output field: 'output'

Dataset loaded Versi No Code :
  Train: 6356 samples
  Val:   794 samples
  Test:  794 samples
✓ Using output field: 'output'

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=6) ⬆️
Trainable:       ~9.5M params (3.8%) ⬆️
Dataset:         dataset-task-v3/00-dataset/ (5,560 train) ⬆️
Epochs:          10 ⬆️
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   5e-5 ⬇️ (lebih kecil untuk model lebih besar)
Warmup:          100 steps ⬆️

