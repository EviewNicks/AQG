# Analisis dan Review: IndoT5 vs IndoNanoT5 (LazarusNLP)

## Ringkasan Eksekutif

Laporan ini menyajikan perbandingan mendalam antara dua model bahasa berbasis T5 dari LazarusNLP: `indo-t5-base` dan `IndoNanoT5-base`. Berdasarkan analisis arsitektur, data pelatihan, dan performa benchmark pada tugas *Natural Language Generation* (NLG) bahasa Indonesia, **IndoNanoT5-base** direkomendasikan sebagai pilihan utama untuk proyek *Automatic Question Generation* (AQG) pemrograman Python Anda. Model ini menawarkan efisiensi parameter yang lebih baik dan performa yang lebih unggul pada tugas-tugas generatif bahasa Indonesia dibandingkan model berbasis mT5 standar.

---

## Perbandingan Spesifikasi Teknis

Berikut adalah perbandingan data teknis antara kedua model tersebut:

| Fitur | IndoT5-base (LazarusNLP) | IndoNanoT5-base (LazarusNLP) |
| :--- | :--- | :--- |
| **Model Dasar** | `google/mt5-base` | Pre-trained dari nol (*from scratch*) |
| **Arsitektur** | Encoder-Decoder (T5) | Encoder-Decoder (T5) |
| **Parameter** | ~580 Juta | ~248 Juta |
| **Data Pelatihan** | Fine-tuned pada dataset Alkitab-Sabda | Pre-trained pada **CulturaX** (23M dokumen) |
| **Sifat Bahasa** | Multilingual (Indonesian-centric fine-tune) | **Monolingual (Indonesia)** |
| **Tujuan Utama** | Machine Translation / Religious Domain | General Purpose Indonesian NLG |
| **Framework** | HuggingFace Trainer | nanoT5 (Optimized for budget/speed) |

---

## Analisis Performa dan Benchmark (IndoNLG)

Berdasarkan data benchmark resmi dari proyek LazarusNLP pada tugas **IndoNLG**, IndoNanoT5-base menunjukkan keunggulan yang signifikan dibandingkan model *baseline* lainnya:

### 1. Ringkasan (IndoSum)
IndoNanoT5-base mencapai skor tertinggi dibandingkan model dengan ukuran serupa, bahkan melampaui mBART Large yang memiliki parameter jauh lebih besar.
*   **IndoNanoT5-base**: R1: **75.29**, R2: **71.23**, RL: **73.30**
*   **mT5 Small**: R1: 74.04, R2: 69.64, RL: 71.89
*   **IndoBART**: R1: 70.67, R2: 65.59, RL: 68.18

### 2. Tanya Jawab (TyDiQA)
Pada tugas pemahaman bacaan dan ekstraksi jawaban, model ini menunjukkan stabilitas yang tinggi.
*   **IndoNanoT5-base**: F1: **72.19**, EM: 58.94
*   **mT5 Small**: F1: 51.90, EM: 35.67

### 3. Chatbot/Persona (XPersona)
Dalam tugas yang membutuhkan kreativitas bahasa dan konteks (mirip dengan AQG), IndoNanoT5-base unggul telak.
*   **IndoNanoT5-base**: BLEU: **4.07**
*   **IndoBART**: BLEU: 2.93
*   **mT5 Small**: BLEU: 1.89

---

## Mengapa IndoNanoT5-base Lebih Baik untuk Proyek Anda?

Berdasarkan dokumen proyek Anda yang berjudul *"Automatic Generation of Python Programming Quiz Questions and Distractors"*, berikut adalah alasan teknis mengapa **IndoNanoT5-base** adalah pilihan yang lebih tepat:

1.  **Fokus Monolingual**: IndoNanoT5 dilatih khusus pada korpus bahasa Indonesia (CulturaX) dari nol. Hal ini membuatnya memiliki *vocabulary* yang lebih efisien untuk struktur kalimat bahasa Indonesia dibandingkan `indo-t5-base` yang merupakan hasil *fine-tuning* dari model multilingual (mT5).
2.  **Efisiensi Parameter & LoRA**: Dengan parameter yang lebih kecil (~248M vs ~580M), proses *fine-tuning* menggunakan LoRA akan jauh lebih cepat, hemat memori (VRAM), dan kecil kemungkinan mengalami *overfitting* pada dataset Anda yang berukuran kecil hingga menengah (400-3000 pasang data).
3.  **Kualitas Generasi (NLG)**: Tugas menghasilkan distraktor kuis membutuhkan logika bahasa yang kuat agar tidak terjadi halusinasi. Benchmark menunjukkan IndoNanoT5 memiliki skor ROUGE dan BLEU yang lebih stabil untuk tugas generatif di bahasa Indonesia.
4.  **Kesesuaian Domain**: `indo-t5-base` yang ada di HuggingFace LazarusNLP saat ini lebih banyak dioptimasi untuk domain spesifik (seperti terjemahan teks keagamaan), sedangkan IndoNanoT5 dirancang sebagai model dasar (*base model*) untuk berbagai tugas NLG umum.

---

## Rekomendasi Implementasi

Untuk proyek AQG Python Anda, saya menyarankan langkah-langkah berikut:

*   **Model Utama**: Gunakan `LazarusNLP/IndoNanoT5-base`.
*   **Fallback**: Tetap pertimbangkan `google/mt5-base` hanya jika Anda menemukan banyak istilah teknis Python dalam bahasa Inggris yang tidak tertangani dengan baik oleh model monolingual (meskipun IndoNanoT5 biasanya cukup kuat untuk *code-mixed text* ringan).
*   **Strategi Fine-tuning**:
    *   Gunakan **LoRA** dengan rank (r) 8 atau 16 untuk menjaga efisiensi.
    *   Pastikan prompt Anda menyertakan instruksi yang jelas (seperti yang ada di dokumen Anda: Materi + Tingkat Kesulitan + Jenis Soal).
    *   Lakukan *filtering* pasca-generasi untuk memvalidasi sintaks Python yang dihasilkan model.

## Referensi
[1] [LazarusNLP IndoT5 GitHub Repository](https://github.com/LazarusNLP/IndoT5)
[2] [HuggingFace Model Card: IndoNanoT5-base](https://huggingface.co/LazarusNLP/IndoNanoT5-base)
[3] [Cahyawijaya et al. (2021). IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation.](https://arxiv.org/abs/2104.08200)
