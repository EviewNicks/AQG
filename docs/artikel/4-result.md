# 4. Hasil dan Pembahasan

Bagian ini memaparkan hasil eksperimen fine-tuning IndoNanoT5 untuk task AQG-DG, mencakup analisis dataset, evaluasi performa model, dan perbandingan metode PEFT.

---

## 4.1. Hasil Dataset

Penelitian ini berhasil mengembangkan dataset berkualitas tinggi untuk domain pendidikan pemrograman Python berbahasa Indonesia dengan total **5.662 entri soal pilihan ganda**. Dataset didominasi oleh soal konseptual (62.1%), sementara soal yang melibatkan interpretasi kode mencakup 37.9%, sesuai dengan target distribusi (Knowledge ≥60%, Code ≤40%).

**Tabel 1. Distribusi Dataset Berdasarkan Tipe Konten**

| Kategori | Jumlah | Persentase |
|:---------|-------:|-----------:|
| Dataset Full (Knowledge + Code) | 5,662 | 100% |
| Dataset Knowledge Only (No-Code) | 3,515 | 62.1% |

Distribusi tingkat kesulitan dataset menunjukkan proporsi yang seimbang: **Mudah 35%**, **Sedang 45%**, dan **Sulit 20%**, mencerminkan kurva pembelajaran alami dalam pendidikan pemrograman. Dataset mencakup **60+ konsep pemrograman Python** dari **11 modul pembelajaran**.

Berikut disajikan dua contoh representatif dari masing-masing kategori dataset:

**Contoh 1: Soal Knowledge (No-Code)**

```
Input: buat_soal_pilihan_ganda: Python adalah bahasa pemrograman multifungsi 
yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR).

Output: question: Dari penjelasan di atas, Tahun berapakah bahasa pemrograman 
Python pertama kali dirilis?
answer: 1991


distractors: 1995 | 1989 | 2000

Metadata: {"difficulty": "Mudah", "type":"knowledge", "source_section": "01-perkenalan-python"}
``` 

2: Soal Code (Knowledge + Code)**

```
Input: buat_soal_pilihan_ganda: Kode program berikut: 
```python print("Hello World!") ```
 Sintaks tersebut memberikan perintah untuk menampilkan/mencetak pesan 
berupa teks ke layar komputer.

Output: question: Menurut materi tersebut, Apa hasil eksekusi dari kode 
```print("Hello World!")```? 
answer: Hello World!
distractors: Tidak ada output | print("Hello World!") | Error

Metadata: {"difficulty": "Mudah", "type": "code" , "source_section": "01-perkenalan-python"}
```


Distraktor dirancang berdasarkan miskonsepsi umum siswa untuk meningkatkan plausibility dan nilai pedagogis.

---

## 4.2. Performa Baseline (Model Pre-trained)

Evaluasi baseline dilakukan pada model IndoNanoT5-base tanpa fine-tuning menggunakan 10 sampel dari validation set dengan konfigurasi inference beam search (num_beams=4). Evaluasi ini penting untuk menunjukkan efektivitas fine-tuning task-specific dan memberikan titik referensi perbandingan.

**Tabel 2. Metrik Evaluasi Baseline (Model Pre-trained)**

| Metrik     | Nilai  | Interpretasi                                      |
| :-----------| -------:| :--------------------------------------------------|
| BLEU-4     | 0.0030 | Sangat rendah - hampir tidak ada kecocokan n-gram |
| ROUGE-L    | 0.1355 | Sangat rendah - overlap subsequence minimal       |
| Distinct-1 | 0.3139 | Sedang - keragaman unigram cukup                  |
| Distinct-2 | 0.5944 | Tinggi - keragaman bigram baik                    |

Hasil evaluasi menunjukkan performa yang sangat rendah pada metrik BLEU-4 (0.0030) dan ROUGE-L (0.1355), mengindikasikan bahwa model pre-trained tidak mampu menghasilkan output yang sesuai dengan format dan konten soal pilihan ganda. Meskipun metrik diversity (Distinct-1 dan Distinct-2) menunjukkan nilai tinggi, hal ini justru mengindikasikan model menghasilkan output yang bervariasi namun tidak relevan dengan task AQG-DG.

Untuk memberikan gambaran konkret tentang kegagalan model baseline, berikut disajikan contoh output yang dihasilkan:

**Contoh Output Baseline**

```
INPUT:
buat_soal_pilihan_ganda: Fungsi sorted() dalam Python digunakan untuk mengurutkan kumpulan nilai. Contoh: sorted([3, 1, 4, 1, 5]) akan menghasilkan [1, 1, 3, 4, 5].

REFERENCE:
question: Apa fungsi dari sorted() dalam Python?
answer: Mengurutkan kumpulan nilai
distractors: Mencari nilai terbesar | Mencari nilai terkecil | Menghitung jumlah

PREDICTION:
sorted() akan menghasilkan [3, 1, 4].[3] akan menghasilkan bilangan[2][3] berikut ini adalah fungsi sort() dalam python.[2, 3] perintah perintah perintah perintah.................................
```


baseline menunjukkan kegagalan dalam mengikuti format yang diharapkan (question, answer, distractors) dan menghasilkan teks yang tidak koheren dengan repetisi berlebihan. Hasil ini mengonfirmasi bahwa model pre-trained memerlukan fine-tuning task-specific untuk dapat menghasilkan soal kuis yang berkualitas. Model pre-trained IndoNanoT5-base dilatih pada task umum dan tidak memiliki pengetahuan spesifik tentang format soal pilihan ganda untuk domain pendidikan pemrograman Python.

Hasil Contoh output modle yang di berikan dapat dilihat abhwa prediction yang diberikan tidak sesuai dengan format question yang diharapkan yang mana question:{context } answer: (jawbaan benar) distractors : { pilihan yang lain } , sehingga Hasil baseline ini mengonfirmasi bahwa model pre-trained memerlukan fine-tuning task-specific untuk dapat menghasilkan soal kuis yang berkualitas. Model pre-trained IndoNanoT5-base dilatih pada task umum seperti text generation dan tidak memiliki pengetahuan spesifik tentang format dan karakteristik soal pilihan ganda untuk domain pendidikan pemrograman Python. Fine-tuning dengan dataset task-specific menjadi kebutuhan esensial untuk mengadaptasi model agar mampu menghasilkan output yang sesuai dengan kebutuhan AQG-DG.

## 4.3. Hasil Evaluasi Eksperimen Fine-tuning


## 4.3. Hasil Evaluasi Eksperimen Fine-tuning

Penelitian ini melakukan empat eksperimen fine-tuning dengan metode PEFT yang berbeda untuk mengidentifikasi konfigurasi optimal dalam menghasilkan soal pilihan ganda. Eksperimen mencakup satu metode LoRA (r=8, α=16) dan tiga variasi Adapter dengan dimensi bottleneck berbeda (d=64, d=128, d=512), semua menggunakan dataset v3 yang sama untuk memastikan perbandingan yang adil.

**Tabel 3. Perbandingan Hasil Eksperimen Fine-tuning**

| Eksperimen | Metode PEFT      | Trainable Params | BLEU-4     | ROUGE-L    | BERTScore F1 | Training Time |
| :-----------| :-----------------| :-----------------| -----------:| -----------:| -------------:| :--------------|
| Baseline   | -                | -                | 0.0030     | 0.1355     | -            | -             |
| Exp 1      | LoRA (r=8, α=16) | 884K (0.36%)     | 0.0074     | 0.0647     | 0.5897       | 2.51 jam      |
| Exp 2      | Adapter (d=64)   | 2.38M (0.95%)    | 0.2598     | 0.4809     | 0.7933       | 3.92 jam      |
| Exp 3      | Adapter (d=128)  | 9.5M (3.8%)      | **0.2632** | 0.4826     | 0.7939       | ~4 jam**      |
| Exp 4      | Adapter (d=512)  | 18.9M (7.09%)    | 0.2476     | **0.4909** | **0.7984**   | ~5 jam        |

Hasil evaluasi menunjukkan perbedaan performa yang sangat signifikan antara metode LoRA dan Adapter. **LoRA gagal total** dengan BLEU-4 hanya 0.0074, bahkan lebih buruk dari baseline pada metrik ROUGE-L (0.0647 vs 0.1355). Hal ini mengindikasikan bahwa parameter trainable yang terlalu sedikit (0.36%) tidak mampu mempelajari struktur output kompleks yang dibutuhkan untuk task AQG-DG (question + answer + 3 distractors). Training loss LoRA yang sangat tinggi (15.88) dan validation loss (8.02) mengonfirmasi kegagalan pembelajaran ini.

Sebaliknya, metode **Adapter menunjukkan peningkatan dramatis**. Adapter d=64 mencapai BLEU-4 0.2598 (peningkatan 3,410% dari LoRA), sementara **Adapter d=128 mencapai performa terbaik** dengan BLEU-4 0.2632 dan ROUGE-L 0.4826. Menariknya, Adapter d=512 justru mengalami penurunan performa (BLEU-4: 0.2476) meskipun memiliki parameter trainable 2x lipat lebih banyak dan validation loss terendah (0.73). Fenomena ini mengindikasikan adanya **sweet spot pada dimensi d=128** untuk dataset berukuran 5,560 samples—model yang terlalu kompleks menyebabkan overfitting dan penurunan performa pada metrik task-specific.

Eksperimen tambahan dengan dataset No-Code (hanya soal konseptual tanpa kode) menunjukkan **penurunan performa drastis** pada semua metode. Adapter d=128 pada dataset No-Code hanya mencapai BLEU-4 0.0939 (turun 64% dari 0.2632), sementara Adapter d=64 turun dari 0.2598 ke 0.0899 (turun 65%). Pola serupa terlihat pada Adapter d=512 yang turun dari 0.2476 ke 0.0926 (turun 63%). Temuan ini mengungkapkan bahwa **soal yang melibatkan kode (code-based questions) lebih mudah dipelajari model** karena memiliki struktur dan pola yang lebih konsisten, sedangkan soal konseptual murni memiliki variasi bahasa yang lebih tinggi dan lebih sulit untuk digeneralisasi. walaupun model indonanot5 tidak dilatih dnegan korpus domain knowledge programming language.

Berdasarkan evaluasi komprehensif, **Adapter d=128** dipilih sebagai model terbaik dengan pertimbangan: (1) BLEU-4 tertinggi (0.2632) menunjukkan kemampuan terbaik dalam menghasilkan output sesuai ground truth, (2) keseimbangan optimal antara efisiensi (3.8% parameter trainable) dan ekspresivitas model, (3) konvergensi stabil tanpa overfitting (validation loss 0.93), dan (4) peningkatan signifikan 8,673% dari baseline. Model ini mendemonstrasikan bahwa fine-tuning dengan Adapter berdimensi sedang memberikan hasil optimal untuk task AQG-DG pada domain pendidikan pemrograman Python berbahasa Indonesia.

## 4.4. Analisis Output Model Terbaik

Analisis kualitatif dilakukan terhadap 20 sample output dari model Adapter d=128 untuk mengidentifikasi pola kualitas dan kelemahan sistematis. Evaluasi mencakup kesesuaian format, kualitas pertanyaan, jawaban, dan distraktor.

**Contoh Output Sukses (Knowledge Type - BLEU: 0.3336):**

```
INPUT:
buat_soal_pilihan_ganda: Untuk membuat virtual environment, gunakan perintah python -m venv nama_env di terminal.

REFERENCE:
question: Bagaimana cara membuat virtual environment?
answer: python -m venv nama_env
distractors: create venv nama_env | new env nama_env | make venv nama_env

PREDICTION:
question: bagaimana cara membuat virtual environment? answer: menggunakan perintah python -m venv nama_env di terminal distractors: menggunakan loop | menggunakan perintah lse | tidak menggunakan perintah
```

Model Output:
- Question: "bagaimana cara membuat virtual environment?"
- Answer: "menggunakan perintah python -m venv nama_env di terminal"
- Distractors: "menggunakan loop | menggunakan perintah lse | tidak menggunakan perintah"

Analisis: Format output sesuai standar dataset. Pertanyaan dan jawaban relevan dengan konteks input, meskipun jawaban cenderung verbose dengan mengulangi frasa dari input. Distraktor menunjukkan kelemahan dengan pilihan yang kurang plausible ("menggunakan loop" tidak relevan untuk virtual environment).

**Contoh Output Bermasalah (Code Type - BLEU: 0.2214):**

```

INPUT:
buat_soal_pilihan_ganda: Fungsi len() mengembalikan integer yang merepresentasikan jumlah elemen.

REFERENCE:
question: Perhatikan kode berikut:
```python
data = [10, 20, 30, 40, 50]
print(type(len(data)))
```
Apa output dari kode di atas?
answer: <class 'int'>
distractors: <class 'list'> | <class 'str'> | 5

PREDICTION:
question: perhatikan kode berikut: ```python data = [1, 2, 3, 4] print(len(data)))`` apa output dari kode di atas? answer: 1 distractors: 2 | 3 | error

```

Model Output menghasilkan code block dengan syntax error parah: `class def = [1, 2, 3...]` dan struktur kode yang tidak valid. Ini menunjukkan model kesulitan menghasilkan code block Python yang syntactically correct, terutama untuk konsep OOP yang kompleks.

Dari 20 samples yang dianalisis, model menunjukkan performa yang tidak konsisten dengan variance BLEU tinggi (0.0000 hingga 0.6425). Kelemahan utama teridentifikasi pada tiga aspek: pertama, code block syntax dimana model sering menghasilkan penutupan tidak lengkap (```) atau syntax error (variabel tidak valid, operator salah), terutama pada code block kompleks yang melibatkan OOP atau nested structures; kedua, kualitas distraktor yang sering tidak plausible atau tidak relevan dengan konteks soal, dengan beberapa kasus menunjukkan duplikasi atau pilihan yang terlalu mudah dibedakan dari jawaban benar; ketiga, verbosity jawaban dimana model cenderung mengulang frasa lengkap dari konteks input, padahal reference dataset menggunakan jawaban yang lebih ringkas. Meskipun demikian, model berhasil mempertahankan format output yang konsisten sesuai struktur dataset (question | answer | distractors) pada hampir semua kasus, dengan performa terbaik dicapai pada soal knowledge type dengan konteks sederhana, sementara soal code type dengan kompleksitas tinggi masih menjadi tantangan signifikan.


## 4.5. Perbandingan dengan Model Lain ( OPSIONAL )

> **CATATAN:** Bagian ini akan diisi setelah eksperimen komparatif dengan model eksternal dilakukan. Eksperimen ini memerlukan resource tambahan (API access, compute) yang sedang dalam proses persiapan.

### Rencana Eksperimen Komparatif

**Model yang Akan Dibandingkan:**
1. **IndoNanoT5-Adapter-d128** (model terbaik dari penelitian ini)
2. **OpenRouter Free Tier Model** (e.g., Llama-3-8B-Instruct)
3. **GPT-4 / GLM-4** (commercial baseline)

**Metodologi:**
- Test set: 50 sampel representatif dari berbagai kategori
- Metrik: BLEU-4, ROUGE-L, BERTScore F1, Distinct-1/2
- Inference: beam_search dengan num_beams=4
- Prompt engineering: prompt yang sama untuk semua model

## 4.6. Diskusi dan Limitasi

Penelitian ini berhasil mendemonstrasikan efektivitas metode Adapter layers untuk fine-tuning IndoNanoT5 pada task AQG domain pemrograman Python berbahasa Indonesia. Temuan utama menunjukkan bahwa Adapter d=128 mencapai performa terbaik dengan BLEU-4 0.2632 (peningkatan 8,673% dari baseline), mengonfirmasi bahwa dimensi adapter memiliki sweet spot optimal—model dengan 3.8% trainable parameters memberikan keseimbangan terbaik antara efisiensi dan ekspresivitas. Sebaliknya, metode LoRA gagal total dengan BLEU-4 hanya 0.0074, membuktikan bahwa parameter trainable yang terlalu sedikit (0.36%) tidak mampu mempelajari struktur output kompleks untuk task AQG. Temuan penting lainnya adalah soal berbasis kode terbukti lebih mudah dipelajari model dibandingkan soal konseptual murni, dengan penurunan performa hingga 64% pada dataset No-Code, mengindikasikan bahwa struktur dan pola kode memberikan signal pembelajaran yang lebih konsisten bagi model.

Meskipun demikian, penelitian ini memiliki beberapa keterbatasan yang perlu diakui. Pertama, ukuran dataset yang relatif terbatas (5,662 samples) berpotensi menjadi faktor penyebab performa model yang tidak konsisten, dengan variance BLEU sangat tinggi (0.0000 hingga 0.6425) pada sample outputs. Dataset yang lebih besar dengan distribusi seimbang antara soal knowledge dan code dapat membantu model mempelajari pola generasi soal dengan lebih robust dan mengurangi variabilitas performa. Kedua, metode evaluasi otomatis yang digunakan memiliki limitasi fundamental—BLEU Score mengukur seluruh output (question + answer + distractors) sebagai satu text sequence tanpa membedakan kualitas masing-masing komponen. Pendekatan ini hanya mengukur lexical overlap dan tidak dapat menilai apakah jawaban benar sesuai konteks, apakah distraktor plausible dan berbeda dari jawaban, atau apakah pertanyaan relevan dengan materi input. Evaluasi yang lebih komprehensif seharusnya memisahkan penilaian untuk setiap komponen: menggunakan BLEU atau semantic similarity untuk mengukur relevansi pertanyaan dengan konteks, exact match atau semantic equivalence untuk memvalidasi kebenaran jawaban, serta diversity metrics dan plausibility scoring untuk menilai kualitas distraktor. Ketiga, analisis kualitatif mengungkapkan kelemahan sistematis model dalam menghasilkan code block syntax yang valid (penutupan ``` tidak lengkap, variabel invalid), distraktor yang tidak plausible atau tidak relevan, serta jawaban yang terlalu verbose dengan mengulangi frasa input.

Untuk penelitian lanjutan, prioritas utama adalah ekspansi dataset dengan tetap fokus pada domain pemrograman Python, meningkatkan jumlah samples untuk mencapai distribusi yang lebih representatif dan membantu model belajar dengan lebih stabil. Selain itu, implementasi sistem evaluasi otomatis yang lebih granular dengan memisahkan penilaian per komponen output akan memberikan insight lebih mendalam tentang kekuatan dan kelemahan model pada aspek-aspek spesifik generasi soal. Human evaluation oleh expert dan student testing juga disarankan untuk memvalidasi kualitas pedagogis soal yang dihasilkan, mengukur tingkat kesulitan aktual, dan mengidentifikasi area perbaikan yang tidak terdeteksi oleh metrik otomatis. Penelitian ini memberikan fondasi yang solid untuk pengembangan sistem AQG berbahasa Indonesia, dengan kontribusi utama berupa identifikasi metode PEFT optimal dan pemahaman mendalam tentang karakteristik pembelajaran model untuk task generasi soal pemrograman.
