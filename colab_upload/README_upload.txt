INSTRUKSI UPLOAD KE COLAB
========================

File yang perlu di-upload ke Colab:

1. src_finetuned.zip
   → Upload via cell "Upload Source Code" di notebook
   → Akan di-extract ke /content/src/finetuned/

2. Dataset files (upload terpisah):
   Dari: dataset_aqg/output_domain/
   → train.jsonl
   → validation.jsonl
   → test.jsonl

   Dari: dataset_aqg/dataset-task-spesifc/
   → train.jsonl
   → validation.jsonl
   → test.jsonl

URUTAN NOTEBOOK:
1. 01_setup_and_validation.ipynb
2. 02_domain_adaptation.ipynb
3. 03_task_specific_training.ipynb
