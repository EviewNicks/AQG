---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[8], line 2
      1 # Modul 10
----> 2 run_pipeline(
      3     materi_dir='dataset_aqg/materi',
      4     output_dir=output_dir,
      5     section_filter='10-style-guide',

File d:\2-Project\AQG\src\pipeline\dataset_pipeline.py:206, in run_pipeline(materi_dir, output_dir, max_per_chunk, section_filter, difficulties, question_types, dry_run, max_chunks_per_section)
    200 print(f"[INFO] {len(concepts)} konsep: {concepts[:3]}{'...' if len(concepts) > 3 else ''}")
    202 # Build prompt inputs
    203 # Setiap chunk menghasilkan 1 prompt per kombinasi (difficulty × question_type).
    204 # max_per_chunk mengontrol berapa kali chunk yang sama di-sample ulang dengan
    205 # konsep berbeda — default 1 (tidak di-sample ulang).
--> 206 from src.dataset.prompt_constructor import extract_concept_from_chunk
    207 prompt_inputs = []
    208 for chunk in all_chunks:
    209     # Pilih konsep paling relevan dengan isi chunk (context grounding)

ModuleNotFoundError: No module named 'src.dataset.prompt_constructor'