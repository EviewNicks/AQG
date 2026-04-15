# Panduan Teknis: Domain Adaptation (Stage 1) IndoT5 untuk Domain Python

Tahap ini bertujuan untuk membiasakan model IndoT5 dengan terminologi, sintaksis, dan konsep spesifik Python sebelum masuk ke tahap *Fine-tuning* instruksi atau QA yang lebih kompleks.

## 1. Strategi Domain Adaptation: Span Corruption vs. Input-Output Pairs

Berdasarkan arsitektur T5 (Raffel et al., 2019), terdapat dua pendekatan utama untuk adaptasi domain:

### A. Span Corruption (Denoising) - **Sangat Direkomendasikan**
Ini adalah metode asli yang digunakan saat *pre-training* T5. Model belajar memprediksi bagian teks yang hilang (*masked spans*).
- **Kelebihan**: Mengajarkan model tentang struktur bahasa dan hubungan antar kata dalam domain Python secara mendalam.
- **Format**: Menggunakan sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, dst.).
- **Tanpa Context Field**: Benar, metode ini hanya membutuhkan satu blok teks yang kemudian diproses menjadi pasangan input-target.

### B. Input-Output Pairs (Simple Mapping)
Menggunakan pasangan tanya-jawab sederhana atau definisi.
- **Kelebihan**: Lebih mudah disiapkan jika Anda sudah memiliki daftar glosarium.
- **Format**: `"Apa itu list?"` → `"List adalah struktur data..."`.
- **Prefix**: Bisa menggunakan prefix `"question: "` atau `"explain: "`.

---

## 2. Apakah Prefix "question: {text}" Cukup?

**Ya, tetapi dengan catatan**:
- Untuk **Domain Adaptation**, model lebih membutuhkan paparan terhadap data mentah (korpus) daripada sekadar instruksi.
- Jika Anda menggunakan format *input-output pairs*, prefix membantu model memahami bahwa ia sedang dalam mode "menjawab" atau "menjelaskan".
- Namun, jika tujuannya adalah *Continued Pre-training* (mengajarkan pengetahuan dasar), **Span Corruption** jauh lebih efektif karena memaksa model memahami konteks teknis Python secara utuh [1].

---

## 3. Contoh Dataset JSONL (Domain Python)

Berikut adalah contoh konkret format JSONL untuk kedua metode tersebut:

### Opsi 1: Format Span Corruption (Untuk Pengetahuan Mendalam)
```jsonl
{"input": "Python adalah bahasa <extra_id_0> yang dirilis pada tahun 1991 oleh <extra_id_1>.", "target": "<extra_id_0> pemrograman <extra_id_1> Guido van Rossum <extra_id_2>"}
{"input": "List dalam Python digunakan untuk <extra_id_0> koleksi item dalam satu <extra_id_1>.", "target": "<extra_id_0> menyimpan <extra_id_1> variabel <extra_id_2>"}
```

### Opsi 2: Format Input-Output (Untuk Glosarium/Definisi)
```jsonl
{"input": "question: Apa itu list dalam Python?", "target": "List adalah tipe data terurut yang dapat diubah (mutable) dan digunakan untuk menyimpan sekumpulan item."}
{"input": "question: Apa fungsi dari print()?", "target": "Fungsi print() digunakan untuk mencetak pesan atau output ke layar atau konsol."}
```

---

## 4. Perbedaan Continued Pre-training vs. Domain Adaptation

| Fitur         | Continued Pre-training                                  | Domain Adaptation (Stage 1)                                            |
| :--------------| :--------------------------------------------------------| :-----------------------------------------------------------------------|
| **Tujuan**    | Menambah pengetahuan umum bahasa baru atau domain luas. | Menyesuaikan model dengan terminologi spesifik (Python).               |
| **Objective** | Hampir selalu *Span Corruption*.                        | Bisa *Span Corruption* atau *Supervised Task*.                         |
| **Data**      | Korpus teks mentah yang sangat besar.                   | Teks teknis, dokumentasi, atau glosarium.                              |
| **Hasil**     | Model dasar yang lebih pintar di domain tersebut.       | Model yang siap untuk *fine-tuning* tugas spesifik (QA/Summarization). |

---

## 5. Rekomendasi Langkah Selanjutnya

1. **Gunakan Span Corruption** jika Anda memiliki banyak teks dokumentasi Python mentah. Ini akan membuat model "paham" Python secara intuitif.
2. **Gunakan Input-Output Pairs** jika Anda memiliki daftar tanya-jawab atau glosarium yang sudah matang.
3. **Gunakan TRL SFTTrainer**: Anda dapat menggunakan `SFTTrainer` dari pustaka TRL untuk melatih model dengan format `prompt` dan `completion` yang sederhana [2].

## Referensi
[1] [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.](https://arxiv.org/abs/1910.10683)
[2] [Hugging Face TRL Documentation - Dataset Formats](https://huggingface.co/docs/trl/en/dataset_formats)
