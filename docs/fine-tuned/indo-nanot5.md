# Penjelasan Konsep: Model Pre-trained vs. Fine-tuned pada IndoT5

Dalam konteks *Natural Language Processing* (NLP) modern, terutama dengan model-model besar seperti T5, konsep *pre-trained* dan *fine-tuned* adalah fundamental. Mari kita bedah perbedaannya dan bagaimana hal ini relevan dengan skrip `run_summarization.py` untuk IndoT5.

## 1. Model Pre-trained (LazarusNLP/IndoNanoT5-base)

Model *pre-trained* adalah model yang telah dilatih pada korpus data yang sangat besar dan beragam untuk mempelajari representasi bahasa secara umum. Tujuan utama dari *pre-training* adalah agar model dapat memahami struktur bahasa, tata bahasa, semantik, dan bahkan beberapa pengetahuan dunia.

### Karakteristik Utama:
-   **Pelatihan Skala Besar**: Dilatih pada miliaran token teks dari berbagai sumber (misalnya, Common Crawl, Wikipedia, buku, dll.).
-   **Tugas Umum**: Seringkali dilatih dengan tugas-tugas *self-supervised* seperti *Masked Language Modeling* (MLM) atau *Text-to-Text Transfer Transformer* (T5) yang mengubah semua tugas NLP menjadi format *text-to-text*.
-   **Fondasi Pengetahuan**: Model ini menjadi fondasi yang kaya akan pengetahuan linguistik dan faktual, yang kemudian dapat diadaptasi untuk tugas-tugas spesifik.
-   **Contoh**: `LazarusNLP/IndoNanoT5-base` adalah model T5 yang telah di-*pre-trained* secara ekstensif pada data bahasa Indonesia. Ini berarti model ini sudah memiliki pemahaman yang kuat tentang bahasa Indonesia sebelum diterapkan pada tugas tertentu.

### Relevansi dengan Skrip (`--model-checkpoint LazarusNLP/IndoNanoT5-base`):
Parameter `--model-checkpoint LazarusNLP/IndoNanoT5-base` dalam skrip Anda merujuk pada model *pre-trained* ini. Ini adalah titik awal Anda. Anda mengambil model yang sudah "pintar" dalam bahasa Indonesia dan akan mengajarkannya tugas yang lebih spesifik.

## 2. Model Fine-tuned (Hugging Face 🤗 Hub repo name: LazarusNLP/IndoNanoT5-base-IndoSum)

*Fine-tuning* adalah proses mengambil model *pre-trained* dan melatihnya lebih lanjut pada dataset yang lebih kecil dan spesifik untuk tugas tertentu (misalnya, ringkasan, tanya jawab, terjemahan mesin, dll.). Tujuannya adalah untuk mengadaptasi pengetahuan umum model *pre-trained* agar sangat efektif dalam menyelesaikan tugas target.

### Karakteristik Utama:
-   **Pelatihan Tugas Spesifik**: Dilatih pada dataset yang relatif kecil yang secara langsung berkaitan dengan tugas yang diinginkan (misalnya, dataset pasangan artikel-ringkasan untuk tugas ringkasan).
-   **Penyesuaian Parameter**: Selama *fine-tuning*, parameter model *pre-trained* disesuaikan sedikit untuk mengoptimalkan kinerja pada tugas baru.
-   **Efisiensi**: Jauh lebih efisien daripada melatih model dari awal (*from scratch*) karena model sudah memiliki pemahaman bahasa yang kuat.
-   **Contoh**: Jika Anda mengambil `LazarusNLP/IndoNanoT5-base` dan melatihnya pada dataset ringkasan bahasa Indonesia (misalnya, IndoSum), model yang dihasilkan disebut model *fine-tuned* untuk tugas ringkasan. Nama seperti `LazarusNLP/IndoNanoT5-base-IndoSum` menunjukkan bahwa model dasar (`IndoNanoT5-base`) telah di-*fine-tuned* untuk tugas IndoSum.

### Relevansi dengan Skrip (`--hub-model-id LazarusNLP/IndoNanoT5-base-IndoSum`):
Parameter `--hub-model-id LazarusNLP/IndoNanoT5-base-IndoSum` adalah nama yang akan diberikan kepada model Anda setelah proses *fine-tuning* selesai dan Anda mengunggahnya ke Hugging Face Hub. Ini adalah identitas model *fine-tuned* Anda.

## 3. Bedah Parameter Skrip `run_summarization.py`

Skrip `run_summarization.py` adalah contoh implementasi untuk melakukan *fine-tuning* model T5 untuk tugas ringkasan. Mari kita lihat parameter-parameter pentingnya:

| Parameter                                           | Deskripsi                                                                                                      | Keterangan & Penyesuaian untuk Notebook Anda                                                                                                                                                                                         |
| :----------------------------------------------------| :---------------------------------------------------------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--model-checkpoint LazarusNLP/IndoNanoT5-base`     | **Model Pre-trained**: Jalur atau nama model *pre-trained* yang akan digunakan sebagai titik awal.             | **Penting**: Ini adalah model dasar yang akan Anda latih. Pastikan Anda menggunakan model yang sesuai dengan bahasa dan arsitektur yang Anda inginkan (misalnya, IndoNanoT5 untuk bahasa Indonesia).                                 |
| `--dataset-name LazarusNLP/indonlg`                 | **Nama Dataset**: Nama dataset dari Hugging Face Hub yang akan digunakan.                                      | Sesuaikan dengan dataset yang Anda gunakan. Jika dataset Anda lokal, Anda perlu memuatnya secara manual dengan `load_dataset("json", data_files="path/to/your_dataset.json")` atau sejenisnya.                                       |
| `--dataset-config indosum`                          | **Konfigurasi Dataset**: Sub-konfigurasi spesifik dari dataset (jika ada).                                     | Untuk IndoNLG, ini bisa berupa `indosum` (ringkasan) atau `question_answering` (tanya jawab). Sesuaikan dengan tugas Anda.                                                                                                           |
| `--input-column-name input`                         | **Kolom Input**: Nama kolom di dataset yang berisi teks input (misalnya, artikel).                             | Sesuaikan dengan nama kolom di dataset Anda yang berisi teks sumber.                                                                                                                                                                 |
| `--target-column-name target`                       | **Kolom Target**: Nama kolom di dataset yang berisi teks target (misalnya, ringkasan).                         | Sesuaikan dengan nama kolom di dataset Anda yang berisi teks keluaran yang diharapkan.                                                                                                                                               |
| `--input-max-length 512`                            | **Panjang Maksimal Input**: Panjang maksimal token untuk input model.                                          | Sesuaikan berdasarkan panjang rata-rata input Anda dan kapasitas memori GPU. Nilai yang lebih besar membutuhkan lebih banyak memori.                                                                                                 |
| `--target-max-length 512`                           | **Panjang Maksimal Target**: Panjang maksimal token untuk output model (target).                               | Sesuaikan berdasarkan panjang rata-rata output yang Anda harapkan.                                                                                                                                                                   |
| `--num-beams 5`                                     | **Jumlah Beam**: Lebar *beam search* selama inferensi (generasi teks).                                         | Nilai yang lebih tinggi dapat menghasilkan output yang lebih baik tetapi lebih lambat. Umumnya 3-5 adalah nilai yang baik.                                                                                                           |
| `--output-dir outputs/indo-nanot5-indosum`          | **Direktori Output**: Lokasi untuk menyimpan *checkpoint* model, log, dan hasil lainnya.                       | Sesuaikan dengan lokasi penyimpanan yang Anda inginkan.                                                                                                                                                                              |
| `--num-train-epochs 5`                              | **Jumlah Epoch**: Berapa kali model akan melihat seluruh dataset pelatihan.                                    | Sesuaikan berdasarkan ukuran dataset dan konvergensi model. Lebih banyak epoch tidak selalu lebih baik (risiko *overfitting*).                                                                                                       |
| `--optim adamw_torch_fused`                         | **Optimizer**: Algoritma optimasi yang digunakan.                                                              | `adamw_torch_fused` adalah versi yang dioptimalkan dari AdamW. Anda bisa menggunakan `adamw_torch` jika tidak ada masalah performa.                                                                                                  |
| `--learning-rate 1e-3`                              | **Learning Rate**: Ukuran langkah yang diambil optimizer saat memperbarui bobot model.                         | Sangat penting! Untuk *fine-tuning* model *pre-trained*, *learning rate* biasanya lebih kecil (misalnya, `1e-5` hingga `5e-5`) dibandingkan dengan melatih dari awal. `1e-3` mungkin terlalu tinggi untuk *fine-tuning* yang stabil. |
| `--weight-decay 0.01`                               | **Weight Decay**: Regularisasi untuk mencegah *overfitting*.                                                   | Nilai umum antara 0.01 hingga 0.1.                                                                                                                                                                                                   |
| `--per-device-train-batch-size 8`                   | **Ukuran Batch Pelatihan**: Jumlah sampel per *batch* untuk setiap GPU/perangkat selama pelatihan.             | Sesuaikan berdasarkan kapasitas memori GPU Anda. Kurangi jika Anda mengalami *out-of-memory* (OOM) error.                                                                                                                            |
| `--per-device-eval-batch-size 16`                   | **Ukuran Batch Evaluasi**: Jumlah sampel per *batch* untuk setiap GPU/perangkat selama evaluasi.               | Bisa lebih besar dari *batch size* pelatihan karena tidak ada perhitungan gradien.                                                                                                                                                   |
| `--hub-model-id LazarusNLP/IndoNanoT5-base-IndoSum` | **ID Model di Hugging Face Hub**: Nama repositori di Hugging Face Hub tempat model *fine-tuned* akan diunggah. | **Penting**: Ini adalah nama unik untuk model *fine-tuned* Anda. Sesuaikan dengan nama yang deskriptif untuk model dan tugas Anda.                                                                                                   |

## 4. Penyesuaian untuk Notebook Fine-tuning Anda

Jika Anda bekerja di lingkungan notebook (seperti Jupyter atau Google Colab), Anda tidak akan menjalankan skrip `.sh` secara langsung. Sebaliknya, Anda akan menerjemahkan parameter-parameter ini ke dalam argumen Python untuk `Seq2SeqTrainingArguments` dan fungsi-fungsi lainnya.

Berikut adalah contoh bagaimana Anda dapat mengadaptasi parameter di atas ke dalam kode Python di notebook Anda:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate

# 1. Muat Tokenizer dan Model Pre-trained
model_checkpoint = "LazarusNLP/IndoNanoT5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 2. Muat Dataset Anda
# Jika dari Hugging Face Hub:
dataset_name = "LazarusNLP/indonlg"
dataset_config = "indosum"
dataset = load_dataset(dataset_name, dataset_config)

# Jika dataset lokal (contoh format JSON Lines):
# dataset = load_dataset("json", data_files={"train": "path/to/train.jsonl", "validation": "path/to/val.jsonl"})

# Definisikan nama kolom input dan target sesuai dataset Anda
input_column_name = "input" # Sesuaikan jika nama kolom berbeda
target_column_name = "target" # Sesuaikan jika nama kolom berbeda

# 3. Fungsi Preprocessing
max_input_length = 512
max_target_length = 128 # Sesuaikan dengan panjang ringkasan yang diharapkan

def preprocess_function(examples):
    inputs = examples[input_column_name]
    # Tambahkan task prefix jika diperlukan oleh model T5 Anda
    # Contoh: inputs = ["summarize: " + text for text in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples[target_column_name], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 5. Metrik Evaluasi
metric = evaluate.load("rouge") # Untuk summarization

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Ganti -100 di labels menjadi pad_token_id untuk decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a list of str for predictions and references
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

# 6. Konfigurasi Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_indosum", # Sesuaikan dengan direktori output Anda
    evaluation_strategy="epoch",
    learning_rate=5e-5, # Learning rate yang lebih konservatif untuk fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True, # Gunakan jika GPU mendukung
    push_to_hub=True, # Untuk mengunggah model ke Hugging Face Hub
    hub_model_id="LazarusNLP/IndoNanoT5-base-IndoSum", # Sesuaikan dengan ID model Anda
    # Optional: logging_steps, save_steps, etc.
)

# 7. Inisialisasi dan Latih Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # Tambahkan fungsi metrik
)

trainer.train()

# Setelah pelatihan, Anda bisa mengunggah model secara manual jika push_to_hub=False
# trainer.push_to_hub()
```

## Kesimpulan

Memahami perbedaan antara model *pre-trained* dan *fine-tuned* adalah kunci untuk memanfaatkan kekuatan *transfer learning* di NLP. Model *pre-trained* memberikan fondasi pengetahuan bahasa yang kuat, sementara *fine-tuning* mengadaptasi pengetahuan tersebut untuk tugas spesifik Anda. Parameter-parameter dalam skrip `run_summarization.py` secara langsung memetakan ke konfigurasi yang Anda perlukan saat menyiapkan proses *fine-tuning* di notebook Anda, memungkinkan Anda untuk mengontrol setiap aspek pelatihan secara detail.
