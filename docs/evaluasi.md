# 4 configure training

============================================================
TRAINING CONFIGURATION
============================================================

✓ Hyperparameters (Spec Defaults):
  Epochs:              6
  Learning Rate:       0.0002
  Warmup Steps:        10
  Batch Size:          4
  Gradient Accum:      4
  Effective Batch:     16
  FP16:                True

✓ Training Estimates:
  Steps per epoch:     ~10
  Total steps:         ~60
  Warmup duration:     ~1.0 epochs
  Estimated time:      30-45 minutes on T4 GPU

✓ Checkpoint Directory:
  /content/drive/MyDrive/dataset_aqg/checkpoints/domain

============================================================
✅ Configuration ready for training
============================================================

# 5 Start Training

Starting domain adaptation training...
Checkpoints will be saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain
============================================================

============================================================
STARTING DOMAIN ADAPTATION TRAINING
============================================================

✓ Model moved to GPU: Tesla T4
  Model device: cuda:0
Preprocessing datasets...
Preprocessing 161 samples...
  Columns: ['input', 'target']
  Batch size: 32
  Removing columns: ['input', 'target']

✓ Preprocessed 22 samples
  Sample label check: 31 valid tokens, 0 masked (-100)

=== Dataset Size After Preprocessing ===
Train samples (actual): 161
Eval samples (actual):  22

=== Training Configuration ===
Epochs: 6
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16
Learning rate: 0.0002
Warmup steps: 10
FP16: True
Train samples: 161
Eval samples: 22
Starting training...

=== Training Complete ===
Final training loss: 36.6791
Training time: 107.23 seconds
Training samples per second: 9.01
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/domain/training_results.json

Training completed in 0.03 hours
Final training loss: 36.6791



# 6 Save Best Model 

✓ Best model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain/indot5-python-domain
Model saved to: /content/drive/MyDrive/dataset_aqg/checkpoints/domain/indot5-python-domain
✓ Training curves saved to /content/drive/MyDrive/dataset_aqg/checkpoints/domain/training_curves.png
✓ Training curves saved

# 7  Evaluate on Validation Set

============================================================
INFERENCE TEST
============================================================
Input:  Apa itu list dalam Python?
Output: (),"-'/?`. di dari: dalam hal ini adalah sebagai berikut; dan yang lainnya juga bisa!—ta= untukh#123456* atau anda dapat menggunakan fitur pada> langsung saja ke halaman dengan kata lain kita akan mengetahui apa itu’”[0]–“terms\» yaitu sendiri baru sebelumnya tentang caraed_ seperti tersebut lagi tertentu
------------------------------------------------------------
Input:  Jelaskan fungsi dalam Python.
Output: (1). dalam hal ini, ada beberapa cara untuk melakukannya:"[2]-— di atas= dengan demikian kita dapat memahaminya lebih lanjut lagi tentang apa yang akan terjadi selanjutnya pada suatu waktu nanti tersebut berikut adalah penjelasan singkat mengenai pengertian dari kata/etc;”>“ dan’s all about me?'*`#\ sebagai misalnya saja menggunakan atau hanya satu selain itu juga»nh
------------------------------------------------------------
Input:  Apa itu variable dalam Python?
Output: (),"-nya.'/?` di dari dalam: yang dapat kita gunakan untuk memahami cara kerja dengan menggunakan bahasa indonesia sebagai* dan juga bisa digunakan oleh pengguna lain seperti#10 atau lebih baik=htc+’s_ef—“>” pada langsung saja berikut ini adalah penjelasan tentang apa itu tersebut!– yaitu; sendiri jika anda ingin|[]2»n sebelumnyad
------------------------------------------------------------

✅ Inference test complete
============================================================

