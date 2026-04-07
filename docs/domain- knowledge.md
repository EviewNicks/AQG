# Laporan Strategis: Penentuan Pendekatan Fine-tuning untuk Proyek AQG Python

Laporan ini menyajikan hasil riset literatur, sesi *brainstorming*, dan perancangan arsitektur alternatif untuk menentukan pendekatan *fine-tuning* yang paling efektif bagi proyek *Automatic Question Generation* (AQG) Python Anda. Tujuan utama adalah untuk menjawab pertanyaan krusial: apakah kita membutuhkan dataset *fine-tuned* khusus untuk kuis, atau cukup dengan *fine-tuning* model berdasarkan materi modul, atau kombinasi keduanya, mengingat tujuan proyek yang lebih luas (generasi kuis, penjelasan kode, dll.).

## 1. Task-Specific Fine-tuning (AQG) vs Domain-Specific Fine-tuning (Domain Knowledge)

Dalam pengembangan model *Large Language Model* (LLM) seperti IndoT5, terdapat dua pendekatan utama untuk *fine-tuning*:

### a. Task-Specific Fine-tuning (AQG)

*   **Definisi**: Melatih model langsung pada dataset yang diformat khusus untuk tugas tertentu, dalam hal ini, *Automatic Question Generation* (AQG) dengan *output* pertanyaan, jawaban, dan distraktor. Model belajar memetakan konteks materi ke format kuis yang spesifik.
*   **Kelebihan**: Model akan sangat teroptimalisasi untuk menghasilkan kuis dengan format yang akurat dan konsisten. Kontrol penuh terhadap struktur *output* memudahkan integrasi API ke *frontend*.
*   **Kekurangan**: Keterbatasan kemampuan di luar tugas kuis. Model mungkin tidak performa baik untuk tugas lain seperti *explain code* atau *summarization* tanpa *fine-tuning* tambahan. Kualitas *output* sangat bergantung pada kualitas dataset kuis, dan ada potensi *hallucination* konten jika *context grounding* tidak kuat.

### b. Domain-Specific Fine-tuning (Domain Knowledge)

*   **Definisi**: Melatih model pada korpus data yang spesifik untuk suatu domain (misalnya, edukasi Python) untuk meningkatkan pemahaman model terhadap terminologi, gaya bahasa, dan konsep-konsep domain tersebut. Data biasanya berupa teks mentah atau dengan tugas generik seperti *summarization*.
*   **Kelebihan**: Model akan memiliki pemahaman mendalam tentang materi Python, mengurangi *hallucination* konten dan meningkatkan relevansi informasi. Ini menjadi fondasi kuat untuk berbagai tugas NLP terkait Python di masa depan (misalnya, *explain code*, *summarization*).
*   **Kekurangan**: Tidak secara otomatis menghasilkan kuis terstruktur. Membutuhkan korpus domain yang besar dan berkualitas tinggi (seluruh 11 modul, 55 *lesson*). Proses *fine-tuning* awal bisa lebih mahal dan memakan waktu.

### Perbandingan Singkat

| Fitur / Tujuan | Task-Specific Fine-tuning (AQG) | Domain-Specific Fine-tuning (Domain Knowledge) |
| :--- | :--- | :--- |
| **Generasi Kuis Otomatis** | Sangat efektif, format terjamin. | Membutuhkan *prompt engineering* lanjutan atau *fine-tuning* tambahan. |
| **Explain Code Python** | Tidak langsung mendukung, perlu *fine-tuning* terpisah. | Sangat mendukung, model memahami konteks kode. |
| **Pemahaman Materi Python** | Terbatas pada konteks kuis. | Sangat mendalam, model "paham" materi. |
| **Kualitas Distraktor** | Bergantung pada dataset kuis. | Perlu *fine-tuning* tambahan untuk menghasilkan distraktor pedagogis. |
| **Fleksibilitas Tugas Lain** | Rendah. | Tinggi, dapat diadaptasi ke berbagai tugas. |
| **Ketergantungan Data** | Dataset kuis teranotasi. | Korpus teks domain mentah. |

## 2. Rekomendasi: Pendekatan Hibrida (Hybrid Approach)

Untuk mencapai tujuan proyek Anda yang multifaset (generasi kuis, *explain code*, dll.) dengan kualitas tinggi, pendekatan hibrida adalah strategi yang paling optimal. Pendekatan ini melibatkan dua tahap *fine-tuning* berurutan pada model IndoT5, memanfaatkan kelebihan dari kedua strategi di atas:

### a. Tahap 1: Domain Adaptation (IndoT5-Python)

*   **Tujuan**: Melatih IndoT5 pada seluruh 11 modul (55 *lesson*) materi Python Anda. Ini akan membuat model memiliki pemahaman yang mendalam tentang domain pendidikan Python berbahasa Indonesia.
*   **Data**: Seluruh teks materi Python (Markdown/YAML) yang telah di-*chunk* dan mungkin dirangkum. Format data dapat berupa *unsupervised pre-training* (MLM/T5 generik), *generative summarization*, atau *question answering* sederhana.
*   **Strategi LoRA**: Diterapkan pada tahap ini untuk efisiensi *fine-tuning* dan pengurangan ukuran model.
*   **Output**: Model `IndoT5-Python` yang memiliki pemahaman mendalam tentang materi Python Anda.

### b. Tahap 2: Task-Specific Fine-tuning (IndoT5-Python-AQG)

*   **Tujuan**: Mengajarkan model `IndoT5-Python` (dari Tahap 1) untuk menghasilkan pertanyaan, jawaban, dan distraktor kuis dengan format spesifik dari konteks materi Python.
*   **Model Dasar**: `IndoT5-Python` (hasil dari Tahap 1).
*   **Data**: Dataset AQG yang telah kita buat (`accumulated.jsonl`) dan telah diperbaiki berdasarkan rekomendasi evaluasi sebelumnya (peningkatan *context grounding*, `misconception_tags`, variasi kesulitan, dll.).
*   **Prompt Template**: Menggunakan *prompt template* yang telah dirancang untuk AQG, dengan *input* berupa `Konteks: <teks_materi>\n\nPrompt: <instruksi_generasi_kuis>` dan *target* berupa `Pertanyaan: ... Jawaban benar: ... Distraktor: ...`.
*   **Strategi LoRA**: Diterapkan juga pada tahap ini untuk memungkinkan model belajar tugas AQG tanpa melupakan pengetahuan domain yang telah diperoleh.
*   **Output**: Model `IndoT5-Python-AQG` yang mampu menghasilkan kuis Python berkualitas tinggi dan relevan secara pedagogis.

### Diagram Arsitektur Hibrida

```mermaid
graph TD
    A[IndoT5 Base Model] --> B{Domain Adaptation}
    B -- "Materi Python (11 Modul, 55 Lesson)" --> C[IndoT5-Python Model]
    C --> D{Task-Specific Fine-tuning (AQG)}
    D -- "Dataset AQG (Kuis Python)" --> E[IndoT5-Python-AQG Model]
    E --> F[API Endpoint]
    F --> G[Frontend Next.js]
    C --> H{Task-Specific Fine-tuning (Explain Code)}
    H -- "Dataset Explain Code" --> I[IndoT5-Python-ExplainCode Model]
    I --> J[API Endpoint]
    J --> K[Frontend Next.js]
```

## 3. Manfaat Pendekatan Hibrida

*   **Sinergi Optimal**: Pemahaman domain yang kuat (Tahap 1) akan membantu model menghasilkan kuis yang lebih relevan dan akurat secara kontekstual di Tahap 2. Sebaliknya, *fine-tuning* kuis (Tahap 2) akan mengajarkan model untuk memformat *output* secara spesifik.
*   **Fleksibilitas Jangka Panjang**: Model `IndoT5-Python` dapat digunakan sebagai *base model* untuk *fine-tuning* tugas lain di masa depan (misalnya, *explain code* dengan dataset *code-explanation*), menjadikannya asisten edukasi Python yang komprehensif.
*   **Kualitas Output Menyeluruh**: Kombinasi ini diharapkan menghasilkan kuis yang tidak hanya benar secara format tetapi juga kaya secara pedagogis dan *grounded* pada materi, serta mampu menjelaskan kode dengan akurasi tinggi.
*   **Efisiensi LoRA**: Penggunaan LoRA di kedua tahap memungkinkan *fine-tuning* yang efisien secara komputasi dan penyimpanan, serta mengurangi risiko *catastrophic forgetting*.

## 4. Kesimpulan dan Langkah Selanjutnya

Dataset *fine-tuned* kuis yang telah Anda buat (`accumulated.jsonl`) **tetap sangat dibutuhkan** dan merupakan komponen krusial dalam Tahap 2. Namun, untuk mencapai visi proyek yang lebih besar—yaitu model yang tidak hanya bisa membuat kuis tetapi juga menjelaskan kode dan memiliki pemahaman mendalam tentang materi Python—pendekatan hibrida adalah jalan terbaik.

**Rekomendasi Langkah Selanjutnya:**

1.  **Prioritaskan Persiapan Data Domain**: Kumpulkan dan siapkan seluruh 11 modul (55 *lesson*) materi Python Anda untuk *fine-tuning* IndoT5 pertama (Domain Adaptation). Fokus pada pembersihan, *chunking*, dan pemrosesan data ini.
2.  **Perbaiki dan Perkaya Dataset AQG**: Sambil melakukan *domain adaptation*, terus perbaiki dataset kuis (`accumulated.jsonl`) kita dengan fokus pada *context grounding*, `misconception_tags`, dan variasi kesulitan, seperti yang dibahas dalam laporan evaluasi sebelumnya.
3.  **Lakukan Fine-tuning Tahap 1 (Domain Adaptation)**: Latih IndoT5 pada korpus materi Python yang telah disiapkan.
4.  **Lakukan Fine-tuning Tahap 2 (Task-Specific AQG)**: Setelah model memiliki pemahaman domain yang kuat, gunakan dataset AQG yang sudah diperbaiki untuk *fine-tuning* tahap kedua.

Dengan strategi ini, kita akan membangun sistem AQG yang tidak hanya fungsional tetapi juga cerdas secara pedagogis dan fleksibel untuk kebutuhan edukasi Python di masa depan, memberikan *value proposition* yang jauh lebih kuat untuk proyek Anda.
