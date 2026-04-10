File output:
  accumulated.jsonl              94725 bytes
  checkpoint.json                106 bytes
  dataset_info.json              325 bytes
  test.jsonl                     9901 bytes
  train.jsonl                    74181 bytes
  validation.jsonl               10643 bytes
  validation_failures.jsonl      835 bytes


Train records: 72

Contoh record pertama:
  Keys    : ['input', 'target', 'metadata']
  Format  : qa_generic
  Input   : Apa itu membuat objek baru dalam Python?...
  Target  : Perhatikan contoh berikut:

Ketika Anda melakukan inisialisasi ulang variabel, Python sebenarnya **membuat objek baru** dengan nilai baru — bukan meng...


Train records: 72

Contoh record pertama:
  Keys    : ['input', 'target', 'metadata']
  Format  : qa_generic
  Input   : Apa itu membuat objek baru dalam Python?...
  Target  : Perhatikan contoh berikut:

Ketika Anda melakukan inisialisasi ulang variabel, Python sebenarnya **membuat objek baru** dengan nilai baru — bukan meng...


{
  "total": 90,
  "splits": {
    "train": 72,
    "validation": 9,
    "test": 9
  },
  "format_distribution": {
    "qa_generic": 60,
    "span_corruption": 30
  },
  "module_distribution": {
    "01-berkenalan-dengan-python": 47,
    "02-berinteraksi-dengan-data": 43
  },
  "generated_at": "2026-04-07"
}


DatasetDict({
    train: Dataset({
        features: ['input', 'target', 'metadata'],
        num_rows: 72
    })
    validation: Dataset({
        features: ['input', 'target', 'metadata'],
        num_rows: 9
    })
    test: Dataset({
        features: ['input', 'target', 'metadata'],
        num_rows: 9
    })
})

Contoh dari train split:
{'input': 'Apa itu membuat objek baru dalam Python?', 'target': 'Perhatikan contoh berikut:\n\nKetika Anda melakukan inisialisasi ulang variabel, Python sebenarnya **membuat objek baru** dengan nilai baru — bukan mengubah nilai yang sudah ada.', 'metadata': {'format': 'qa_generic', 'source_file': 'D:\\2-Project\\AQG\\dataset_aqg\\materi\\02-berinteraksi-dengan-data\\03-type-data.md', 'module_name': '02-berinteraksi-dengan-data', 'section_heading': '### Numbers', 'token_count': 335, 'has_code': True}}