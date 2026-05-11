# IndoNanoT5 Fine-tued + LoRA with Dataset V3  03 

Detail :
✓ Base model loaded
✓ LoRA applied: r=8, alpha=16, target=['q', 'v']
  Trainable: 884,736 (0.36%)
  Total:     248,462,592
✓ Model device: cuda:0
  GPU allocated: 1.00 GB


============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 19.09   , 8.89
  2. 17.62 , 8.55
  3. 17.25 , 8.39
  4. 16.98 , 8.27
  5. 16.83 , 8.19
  6. 16.70 , 8.15
  7. 16.55 , 8.12
  8. 16.34 , 8.10
  9. 16.23 , 8.06
  10. 15.88 , 8.02



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

# IndoNanoT5 Fine-tued + LoRA with Dataset V3 No Code Blocks   03

✓ Base model loaded
✓ LoRA applied: r=8, alpha=16, target=['q', 'v']
  Trainable: 884,736 (0.36%)
  Total:     248,462,592
✓ Model device: cuda:0
  GPU allocated: 1.00 GB

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 20.05 , 9.36
  2. 18.44 , 8.83 
  3. 17.77 , 8.64
  4. 17.46 , 8.52 
  5. 17.35 , 8.42
  6. 17.18 , 8.26
  7. 17.07 , 8.33
  8. 17.03 , 8.30
  9. 16.99 , 8.29
  10. 15.67 , 8.27 

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

# IndonanoT5 fine-tuned D=64 With Dataset V3  04

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=64 (reduction_factor=12)
Trainable:       2.38M params (0.95%)
Dataset:         dataset-task-spesifc/ (4,529 train)
Epochs:          10
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   1e-4
Warmup:          100 steps

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 2.96 , 1.22
  2. 2.81 , 1.09
  3. 2.51 , 1.02
  4. 2.34 , 0.99
  5. 2.08 , 0.95
  6. 1.82 , 0.93
  7. 2.15 , 0.92
  8. 1.88 , 0.91
  9. 2.05 , 0.90
  10. 1.93 , 0.89


============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.2909
  BLEU-1:   0.6286
  BLEU-2:   0.4333
  BLEU-3:   0.3160
  BLEU-4:   0.2598

ROUGE Scores:
  ROUGE-1:  0.5285
  ROUGE-2:  0.3488
  ROUGE-L:  0.4809

BERTScore:
  Precision: 0.8040
  Recall:    0.7837
  F1:        0.7933

Diversity:
  Distinct-1: 0.1498
  Distinct-2: 0.4470

# IndonanoT5 fine-tuned D=64 With Dataset V3  no-code  04

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=12)
Trainable:       2.38M params (0.95%)
Dataset:         dataset-task-spesifc/ (4,529 train)
Epochs:          10
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   1e-4
Warmup:          100 steps

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 3.86 , 1.71
  2. 3.60 , 1.58
  3. 3.24 , 1.52 
  4. 3.18 , 1.47
  5. 2.78 , 1.43
  6. 3.11 , 1.41
  7. 3.01 , 1.39
  8. 2.88 , 1.38
  9. 3.10 , 1.37
  10. 2.65 , 1.37

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.1902
  BLEU-1:   0.5516
  BLEU-2:   0.3061
  BLEU-3:   0.1581
  BLEU-4:   0.0899

ROUGE Scores:
  ROUGE-1:  0.5112
  ROUGE-2:  0.2796
  ROUGE-L:  0.4451

BERTScore:
  Precision: 0.7947
  Recall:    0.7761
  F1:        0.7850

Diversity:
  Distinct-1: 0.1515
  Distinct-2: 0.5340

# IndonanoT5 fine-tuned D=128 With Dataset V3  full  05

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=6) ⬆️
Trainable:       ~9.5M params (3.8%) ⬆️
Dataset:         dataset-task-v3/00-dataset/ (5,560 train) ⬆️
Epochs:          10 ⬆️
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   5e-5 ⬇️ (lebih kecil untuk model lebih besar)
Warmup:          100 steps ⬆️

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 3.08 , 1.26 
  2. 2.93 , 1.13
  3. 2.59 , 1.06
  4. 2.44 , 1.01
  5. 2.15 , 0.98
  6. 1.89 , 0.95 
  7. 2.24 , 0.94
  8. 1.93 , 0.94
  9. 2.12 , 0.93
  10. 1.91 , 0.93


============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.2907
  BLEU-1:   0.6357
  BLEU-2:   0.4389
  BLEU-3:   0.3196
  BLEU-4:   0.2632

ROUGE Scores:
  ROUGE-1:  0.5325
  ROUGE-2:  0.3472
  ROUGE-L:  0.4826

BERTScore:
  Precision: 0.8048
  Recall:    0.7841
  F1:        0.7939

Diversity:
  Distinct-1: 0.1470
  Distinct-2: 0.4510

# IndonanoT5 fine-tuned D=128 With Dataset V3  No-Code  05

Model:           IndoNanoT5-base (248M params)
Adapter:         Pfeiffer, d=128 (reduction_factor=6) ⬆️
Trainable:       ~9.5M params (3.8%) ⬆️
Dataset:         dataset-task-v3/00-dataset/ (5,560 train) ⬆️
Epochs:          10 ⬆️
Batch Size:      4 (effective: 8 with grad_accum=2)
Learning Rate:   5e-5 ⬇️ (lebih kecil untuk model lebih besar)
Warmup:          100 steps ⬆️

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 3.51 , 1.62
  2. 3.42 , 1.59
  3. 3.34 , 1.56
  4. 3.30 , 1.52
  5. 2.87 , 1.47
  6. 3.24 , 1.45
  7. 3.11 , 1.43
  8. 2.99 , 1.42
  9. 3.22 , 1.41
  10. 2.78 , 1.41

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

# IndonanoT5 fine-tuned D=512 With Dataset V3  full 07

✓ Adapter added: pfeiffer config, d=512.0
✓ Adapter activated for training
✓ Model moved to GPU
  GPU allocated: 1.08 GB

============================================================
MODEL INFORMATION
============================================================

Parameters:
  Trainable: 18,905,088 (7.09%)
  Total:     266,482,944
  Frozen:    247,577,856

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 2.8 , 1.13
  2. 2.55 , 0.99
  3. 2.23 , 0.92
  4. 1.99 , 0.88
  5. 1.78 , 0.85
  6. 1.50 , 0.82
  7. 1.73 ,  0.81
  8. 1.52 , 0.80 
  9. 1.46 , 0.75
  10. 1.52 , 0.73 

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.3052
  BLEU-1:   0.6159
  BLEU-2:   0.4164
  BLEU-3:   0.3026
  BLEU-4:   0.2476

ROUGE Scores:
  ROUGE-1:  0.5405
  ROUGE-2:  0.3547
  ROUGE-L:  0.4909

BERTScore:
  Precision: 0.8060
  Recall:    0.7920
  F1:        0.7984

Diversity:
  Distinct-1: 0.1422
  Distinct-2: 0.4342


# IndonanoT5 fine-tuned D=512 With Dataset V3  No Code 07

✓ Adapter added: pfeiffer config, d=512.0
✓ Adapter activated for training
✓ Model moved to GPU
  GPU allocated: 1.08 GB

============================================================
MODEL INFORMATION
============================================================

Parameters:
  Trainable: 18,905,088 (7.09%)
  Total:     266,482,944
  Frozen:    247,577,856

============================================================
EPOCH TRAINING Loass and Validation Loss 
============================================================
  1. 3.68 , 1.6
  2. 3.36 , 1.50
  3. 2.93 , 1.42
  4. 2.82 , 1.38 
  5. 2.42 , 1.33
  6. 2.65 , 1.31
  7. 2.54 , 1.30
  8. 2.32 , 1.29 
  9. 2.20 , 1.27
  10. 2.17 , 1.23

BLEU Scores:
  BLEU:     0.2042
  BLEU-1:   0.5312
  BLEU-2:   0.2908
  BLEU-3:   0.1550
  BLEU-4:   0.0926

ROUGE Scores:
  ROUGE-1:  0.5066
  ROUGE-2:  0.2810
  ROUGE-L:  0.4433

BERTScore:
  Precision: 0.7924
  Recall:    0.7818
  F1:        0.7867

Diversity:
  Distinct-1: 0.1313
  Distinct-2: 0.4677