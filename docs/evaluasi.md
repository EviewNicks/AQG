# Analisis BERTScore Model Loading - Section 8 Final Evaluation

## 1. Konteks: Apa itu BERTScore?

BERTScore adalah metrik evaluasi untuk text generation yang menggunakan **contextual embeddings** dari pre-trained BERT models untuk mengukur similarity antara generated text dan reference text.

### Perbedaan dengan Metrik Tradisional:
- **BLEU/ROUGE**: Hanya menghitung exact n-gram matches (surface-level)
- **BERTScore**: Mengukur semantic similarity menggunakan embeddings (meaning-level)

### Contoh:
```
Reference:  "Mobil berwarna merah"
Prediction: "Kendaraan berwarna merah"

BLEU: Rendah (karena "mobil" ≠ "kendaraan")
BERTScore: Tinggi (karena secara semantik mirip)
```

---

## 2. BERTScore Model Loading Report

### Lokasi dalam Code:
File: `src/finetuned/evaluation/metrics_calculator.py`

```python
def compute_bertscore(
    self, 
    predictions: List[str], 
    references: List[str],
    lang: Optional[str] = None,
    model_type: Optional[str] = None
) -> Dict[str, float]:
    """Compute BERTScore (Precision, Recall, F1)."""
    
    lang = lang or self.lang  # Default: "id" (Indonesian)
    
    results = self.bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang,              # ← Ini yang trigger model loading
        model_type=model_type   # ← Optional: specify model explicitly
    )
```

### Proses Loading:

1. **Lazy Loading** (saat pertama kali dipanggil):
```python
@property
def bertscore(self):
    """Lazy load BERTScore metric."""
    if self._bertscore is None:
        self._bertscore = load("bertscore")  # ← Download dari HuggingFace
    return self._bertscore
```

2. **Model Selection** (berdasarkan `lang="id"`):
   - Library `bert_score` akan otomatis memilih model BERT yang sesuai untuk bahasa Indonesia
   - Default untuk Indonesian: `indobenchmark/indobert-base-p1` atau `cahya/bert-base-indonesian-1.5G`

3. **Download & Cache**:
   - Model BERT (~400-500MB) di-download dari HuggingFace Hub
   - Disimpan di cache: `~/.cache/huggingface/`

---

## 3. "UNEXPECTED" Results - Apa yang Terjadi?

### Output yang Muncul:
```
Some weights of the model checkpoint at ... were not used when initializing BertModel: 
['cls.predictions.bias', 'cls.predictions.transform.dense.weight', ...]
- This IS expected for this model, but UNEXPECTED for BertModel.
```

### Penjelasan:

#### A. Mengapa Muncul Warning Ini?

Model BERT yang di-download adalah **BertForMaskedLM** (untuk pre-training task), tetapi BERTScore hanya butuh **BertModel** (base encoder saja).

**Arsitektur BertForMaskedLM:**
```
BertModel (encoder)
├── Embeddings
├── 12 Transformer Layers
└── Pooler
    
+ MLM Head (untuk masked language modeling)
  ├── cls.predictions.transform
  ├── cls.predictions.decoder
  └── cls.predictions.bias
```

**Yang Dibutuhkan BERTScore:**
```
BertModel (encoder) ← Hanya ini!
├── Embeddings
├── 12 Transformer Layers
└── Pooler
```

#### B. Apakah Ini Masalah?

**TIDAK!** Ini adalah behavior yang normal dan expected:

1. **"This IS expected for this model"**: Model checkpoint memang punya MLM head
2. **"but UNEXPECTED for BertModel"**: Karena kita load sebagai BertModel (tanpa head)
3. **Solusi otomatis**: PyTorch/Transformers otomatis **drop** layer yang tidak dipakai

#### C. Analogi:

Seperti membeli mobil lengkap dengan spoiler racing, tapi kamu hanya butuh mesinnya:
- Spoiler dibuang ✓
- Mesin tetap berfungsi sempurna ✓
- Warning: "Spoiler tidak dipakai" (informational saja)

---

## 4. Cara Kerja BERTScore

### Step-by-Step Process:

```python
# 1. Tokenization
predictions = ["Mobil berwarna merah"]
references = ["Kendaraan berwarna merah"]

# 2. Get BERT Embeddings
pred_embeddings = bert_model.encode(predictions)    # Shape: [seq_len, 768]
ref_embeddings = bert_model.encode(references)      # Shape: [seq_len, 768]

# 3. Compute Cosine Similarity (token-level)
similarity_matrix = cosine_similarity(pred_embeddings, ref_embeddings)
# Matrix: [pred_tokens × ref_tokens]

# 4. Greedy Matching
# Untuk setiap token di prediction, cari token paling mirip di reference
precision = max_similarity_per_pred_token.mean()
recall = max_similarity_per_ref_token.mean()
f1 = 2 * (precision * recall) / (precision + recall)
```

### Visualisasi:

```
Prediction: ["Mobil", "berwarna", "merah"]
Reference:  ["Kendaraan", "berwarna", "merah"]

Similarity Matrix:
                Kendaraan  berwarna  merah
Mobil           0.85       0.12      0.08
berwarna        0.10       1.00      0.15
merah           0.05       0.15      1.00

Precision: (0.85 + 1.00 + 1.00) / 3 = 0.95
Recall:    (0.85 + 1.00 + 1.00) / 3 = 0.95
F1:        0.95
```

---

## 5. Code Implementation dalam Notebook

### Section 8: Final Evaluation

```python
# Re-initialize evaluator with trained model
evaluator_final = ModelEvaluator(
    model=peft_model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc  # ← Ini punya bertscore
)

print('Running comprehensive evaluation on test set...')
final_metrics = evaluator_final.evaluate_on_test_set(
    test_dataset=test_dataset,
    num_beams=4,
    include_bertscore=True,  # ← Trigger BERTScore computation
    max_samples=None
)
```

### Flow Execution:

```
evaluate_on_test_set()
  ↓
compute_all_metrics(include_bertscore=True)
  ↓
compute_bertscore(predictions, references, lang="id")
  ↓
[FIRST TIME ONLY]
  ↓
load("bertscore")  # Download library
  ↓
Download BERT model for Indonesian (~500MB)
  ↓
Load BertModel (drop MLM head) ← WARNING MUNCUL DI SINI
  ↓
Compute embeddings & similarity
  ↓
Return {precision, recall, f1}
```

---

## 6. Tujuan Menggunakan BERTScore

### A. Mengatasi Keterbatasan BLEU/ROUGE

**Problem dengan BLEU:**
```python
Reference:  "Python adalah bahasa pemrograman"
Prediction: "Python merupakan bahasa pemrograman"

BLEU: 0.60 (karena "adalah" ≠ "merupakan")
BERTScore: 0.95 (karena semantically equivalent)
```

### B. Evaluasi Semantic Quality

BERTScore lebih baik untuk:
- **Paraphrasing**: Kalimat dengan struktur berbeda tapi makna sama
- **Synonym usage**: "mobil" vs "kendaraan"
- **Word order variations**: "merah mobil" vs "mobil merah"

### C. Correlation dengan Human Judgment

Research menunjukkan BERTScore memiliki **korelasi lebih tinggi** dengan human evaluation dibanding BLEU/ROUGE.

---

## 7. Hasil Evaluasi (dari docs/evaluasi.md)

### Sample Outputs:

```
--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Abstraksi data memastikan...
Reference: question: Apa dampak dari memiliki tujuan variabel...
Prediction: tujuan dari setiap variabel adalah untuk mencapai...
BLEU: 0.0000
```

### Analisis:

**Mengapa BLEU = 0?**
- Model menghasilkan text yang **semantically related** tapi **structurally different**
- BLEU hanya menghitung exact matches → 0.0000

**BERTScore akan lebih tinggi karena:**
- "tujuan variabel" dan "tujuan yang jelas" semantically similar
- Embeddings akan capture relationship ini

---

## 8. Konfigurasi BERTScore

### Default Settings:

```python
class MetricsCalculator:
    def __init__(self, lang: str = "id"):  # ← Indonesian
        self.lang = lang
```

### Model Selection untuk Indonesian:

Library `bert_score` akan otomatis pilih salah satu:
1. `indobenchmark/indobert-base-p1` (recommended)
2. `cahya/bert-base-indonesian-1.5G`
3. `indolem/indobert-base-uncased`

### Custom Model (optional):

```python
bertscore_metrics = self.compute_bertscore(
    predictions, 
    references,
    lang="id",
    model_type="indobenchmark/indobert-base-p1"  # ← Explicit
)
```

---

## 9. Performance Considerations

### Computational Cost:

```python
# BLEU/ROUGE: Fast (string matching)
compute_bleu()    # ~0.1s for 100 samples

# BERTScore: Slow (neural network inference)
compute_bertscore()  # ~10-30s for 100 samples (GPU)
                     # ~60-120s for 100 samples (CPU)
```

### Memory Usage:

- BERT Model: ~500MB in memory
- Embeddings: ~768 dimensions per token
- Batch processing untuk efficiency

### Optimization dalam Code:

```python
def evaluate_on_test_set(
    self,
    test_dataset: Dataset,
    include_bertscore: bool = True,  # ← Optional flag
    max_samples: Optional[int] = None
):
    # Baseline evaluation: include_bertscore=False (fast)
    # Final evaluation: include_bertscore=True (comprehensive)
```

---

## 10. Troubleshooting

### A. Warning: "Some weights not used"

**Status**: ✅ Normal, bisa diabaikan

**Reason**: MLM head di-drop karena tidak dipakai

**Action**: None required

### B. Out of Memory

**Symptom**: CUDA OOM saat compute BERTScore

**Solution**:
```python
# Reduce batch size atau process in chunks
for i in range(0, len(predictions), batch_size):
    batch_preds = predictions[i:i+batch_size]
    batch_refs = references[i:i+batch_size]
    batch_scores = compute_bertscore(batch_preds, batch_refs)
```

### C. Slow Computation

**Solution**: Use GPU
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# BERTScore akan otomatis gunakan GPU jika available
```

---

## 11. Summary

### BERTScore Model Loading:

1. **Tujuan**: Evaluasi semantic similarity menggunakan BERT embeddings
2. **Model**: IndoBERT (~500MB) untuk bahasa Indonesia
3. **Warning "UNEXPECTED"**: Normal behavior, MLM head di-drop
4. **Cara Kerja**: Token-level cosine similarity dengan greedy matching
5. **Keuntungan**: Lebih akurat untuk paraphrasing dan synonym usage
6. **Trade-off**: Lebih lambat tapi lebih meaningful dibanding BLEU

### Recommendation:

```python
# Baseline evaluation (fast)
baseline_metrics = evaluator.evaluate_on_test_set(
    val_dataset,
    include_bertscore=False,  # Skip BERTScore
    max_samples=10
)

# Final evaluation (comprehensive)
final_metrics = evaluator.evaluate_on_test_set(
    test_dataset,
    include_bertscore=True,   # Include BERTScore
    max_samples=None
)
```

---

## Sample Outputs Analysis

Generating 5 sample outputs...

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Abstraksi data memastikan bahwa setiap variabel memiliki tujuan yang jelas dalam program....
Reference: question: Apa dampak dari memiliki tujuan variabel yang jelas melalui abstraksi?
answer: Meminimalkan kesalahan penggunaan variabel dalam kode
distrac...
Prediction: tujuan dari setiap variabel adalah untuk mencapai tujuan yang jelas dalam program. tujuan dari sebuah program adalah untuk menghasilkan output yang di...
BLEU: 0.0000

--- Sample 2 ---
Input: buat_soal_pilihan_ganda: String dapat menggunakan escape character seperti \n untuk newline, \t untuk tab, dan \\ untuk backslash....
Reference: question: Apa fungsi \n dalam string?
answer: Membuat baris baru (newline)
distractors: Membuat tab | Membuat backslash | Membuat spasi...
Prediction: untuk backslash dapat menggunakan escape character seperti \n untuk newline, \\ untuk tab, dan \ \ untuk atau bisa juga menggunakan embassy atau bisa ...
BLEU: 0.0000

--- Sample 3 ---
Input: buat_soal_pilihan_ganda: Pengguna Mac OS atau Ubuntu umumnya sudah memiliki Python yang terinstal secara otomatis....
Reference: question: Sistem operasi mana yang umumnya sudah memiliki Python terinstal secara otomatis?
answer: Mac OS dan Ubuntu
distractors: Windows dan DOS | A...
Prediction: selain itu juga sudah memiliki  dengan kata lain sudah memiliki aplikasi   dengan demikian, akan lebih mudah bagi pengguna untuk melakukan pengaturan ...
BLEU: 0.0000

--- Sample 4 ---
Input: buat_soal_pilihan_ganda: Pemrosesan sekuensial dapat digunakan untuk menghitung mode (nilai yang paling sering muncul). Ini memerlukan frequency count...
Reference: question: Bagaimana cara menghitung mode dengan pemrosesan sekuensial?
answer: Hitung frekuensi setiap nilai kemudian cari nilai dengan frekuensi maks...
Prediction: untuk menghitung mode (nilai yang paling sering muncul), kita dapat menggunakan frequency counting (request counter) dan frequency.  untuk menghitung ...
BLEU: 0.0000

--- Sample 5 ---
Input: buat_soal_pilihan_ganda: Aksi sekuensial adalah fondasi dari semua paradigma pemrograman. Bahkan dalam pemrograman paralel, setiap thread menjalankan ...
Reference: question: Apakah aksi sekuensial relevan dalam pemrograman paralel?
answer: Ya, setiap thread menjalankan instruksi secara sekuensial
distractors: Tid...
Prediction: setiap thread menjalankan instruksi secara sekuensial. setiap th setiap thtread menjalankan perintah secara sekual. thread dijalankan secara paralel, ...
BLEU: 0.0742

✓ Samples saved to /content/drive/MyDrive/dataset_aqg/evaluation_results/sample_outputs.json
✓ 5 sample outputs generated and saved

=== Sample Output ===
Input: buat_soal_pilihan_ganda: Abstraksi data memastikan bahwa setiap variabel memiliki tujuan yang jelas dalam program....