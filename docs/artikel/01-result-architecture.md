# Architecture: Bagian 4. Hasil dan Pembahasan

> **Dokumen Blueprint**: Panduan struktur dan konten untuk penulisan Bagian 4. Hasil dan Pembahasan

---

## Overview

Bagian ini memaparkan hasil eksperimen fine-tuning IndoNanoT5 untuk task AQG-DG, mencakup analisis dataset, evaluasi performa model, dan perbandingan metode PEFT (LoRA vs Adapter).

**Tujuan Utama:**
- Menunjukkan kualitas dataset yang dikembangkan
- Membuktikan efektivitas fine-tuning dengan PEFT
- Mengidentifikasi konfigurasi model terbaik
- Menganalisis trade-off efisiensi vs performa

---

## 4.1. Hasil Dataset

### MUST HAVE:
1. **Statistik Dataset Final**
   - Total entri: 5,662 soal MCQ
   - Distribusi tipe: Knowledge+Code vs No-Code
   - Distribusi kesulitan: Mudah, Sedang, Sulit
   - Tabel distribusi per kategori

2. **Contoh Dataset**
   - Minimal 2 contoh: 1 Knowledge+Code, 1 No-Code
   - Format: Input (konteks) → Output (question, answer, distractors)
   - Highlight struktur JSONL yang digunakan

3. **Validasi Kualitas**
   - Hasil automated checks (format, kelengkapan, duplikasi)
   - Konfirmasi distribusi sesuai target (Knowledge ≥60%, Mudah ~40%, dll)

### SHOULD HAVE:
- Visualisasi distribusi (bar chart/pie chart)
- Statistik panjang teks (rata-rata token input/output)
- Cakupan topik (60+ konsep dari 11 modul)
- Perbandingan dengan dataset AQG lain (jika ada)

### Format Output:
```
## 4.1. Hasil Dataset

[Paragraf pembuka: ringkasan dataset]

### 4.1.1. Komposisi dan Distribusi Dataset

**Tabel 1. Distribusi Dataset Berdasarkan Tipe dan Kesulitan**
| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| ... | ... | ... |

[Analisis distribusi]

### 4.1.2. Contoh Dataset

**Contoh 1: Soal Knowledge+Code**
```json
{input, output, metadata}
```
[Penjelasan contoh]

**Contoh 2: Soal No-Code**
[...]

### 4.1.3. Validasi Kualitas Dataset
[Hasil automated checks dan konfirmasi kualitas]
```

---

## 4.2. Baseline Performance (Model Pre-trained)

### MUST HAVE:
1. **Setup Evaluasi Baseline**
   - Model: IndoNanoT5-base (tanpa fine-tuning)
   - Test set: 10 sampel dari validation set
   - Inference config: beam_search, num_beams=4

2. **Metrik Baseline**
   - BLEU-4, ROUGE-L, BERTScore F1, Distinct-1/2
   - Tabel hasil metrik
   - Interpretasi: mengapa baseline rendah

3. **Contoh Output Baseline**
   - Minimal 1-2 contoh output yang menunjukkan kegagalan model
   - Perbandingan dengan ground truth
   - Highlight masalah: format tidak sesuai, konten tidak relevan

### SHOULD HAVE:
- Analisis error patterns (apa yang salah dari output baseline)
- Justifikasi kebutuhan fine-tuning task-specific

### Format Output:
```
## 4.2. Baseline Performance (Model Pre-trained)

[Paragraf pembuka: pentingnya baseline]

### Setup Evaluasi
[Deskripsi setup]

### Hasil Metrik Baseline

**Tabel 2. Metrik Evaluasi Baseline (Pre-trained Model)**
| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| BLEU-4 | ~0.01 | Sangat rendah |
| ... | ... | ... |

[Analisis hasil]

### Contoh Output Baseline
**Input:** [konteks]
**Ground Truth:** [expected output]
**Baseline Output:** [actual output - menunjukkan kegagalan]

[Analisis: mengapa baseline gagal]
```

---

## 4.3. Hasil Evaluasi Eksperimen Fine-tuning

### MUST HAVE:
1. **Ringkasan 4 Eksperimen**
   - Tabel konfigurasi (sudah ada di Metodologi, bisa direferensikan)
   - Eksperimen 1: LoRA (r=8, α=16) - Dataset v2 (1,332)
   - Eksperimen 2: Adapter (d=64) - Dataset v3 (4,529)
   - Eksperimen 3: Adapter (d=128) - Dataset v3 (5,560)
   - Eksperimen 4: Adapter (d=128) - Dataset v4 (8,680)

2. **Tabel Komparatif Metrik**
   - Semua metrik (BLEU-1/2/3/4, ROUGE-1/2/L, BERTScore P/R/F1, Distinct-1/2)
   - Perbandingan dengan baseline
   - Highlight model terbaik (bold/color)

3. **Training Dynamics**
   - Loss curves (training & validation) untuk setiap eksperimen
   - Analisis convergence: kapan model converge, ada overfitting?
   - Training time dan resource usage

4. **Perbandingan LoRA vs Adapter**
   - Trade-off: efisiensi parameter vs performa
   - Kapan LoRA cukup, kapan Adapter lebih baik
   - Pengaruh dimensi bottleneck (d=64 vs d=128)
   - Pengaruh ukuran dataset (v2 vs v3 vs v4)

5. **Identifikasi Model Terbaik**
   - Berdasarkan metrik evaluasi
   - Pertimbangan efisiensi (trainable params, inference time)
   - Justifikasi pemilihan

### SHOULD HAVE:
- Visualisasi metrik (bar chart perbandingan)
- Analisis per-kategori (Knowledge vs Code, Mudah vs Sedang vs Sulit)
- Statistical significance testing (jika memungkinkan)
- Ablation study insights

### Format Output:
```
## 4.3. Hasil Evaluasi Eksperimen Fine-tuning

[Paragraf pembuka: overview eksperimen]

### 4.3.1. Ringkasan Konfigurasi Eksperimen
[Referensi ke Tabel 2 di Metodologi atau ringkasan singkat]

### 4.3.2. Hasil Metrik Evaluasi

**Tabel 3. Perbandingan Metrik Evaluasi 4 Eksperimen**
| Eksperimen | BLEU-4 | ROUGE-L | BERTScore F1 | Distinct-1 | Distinct-2 |
|------------|--------|---------|--------------|------------|------------|
| Baseline   | 0.01   | 0.02    | 0.45         | 0.20       | 0.30       |
| Exp 1 (LoRA) | ... | ... | ... | ... | ... |
| Exp 2 (Adapter d=64) | ... | ... | ... | ... | ... |
| **Exp 3 (Adapter d=128)** | **...** | **...** | **...** | **...** | **...** |
| Exp 4 (Adapter d=128 v4) | ... | ... | ... | ... | ... |

[Analisis hasil: model mana yang terbaik, mengapa]

### 4.3.3. Training Dynamics

**Gambar 1. Training & Validation Loss Curves**
[Grafik loss curves untuk 4 eksperimen]

[Analisis: convergence, overfitting, stabilitas training]

**Tabel 4. Training Statistics**
| Eksperimen | Training Time | Peak Memory | Final Train Loss | Final Val Loss |
|------------|---------------|-------------|------------------|----------------|
| ... | ... | ... | ... | ... |

### 4.3.4. Perbandingan LoRA vs Adapter

**Trade-off Efisiensi vs Performa:**
- **LoRA (Exp 1):** 
  - Trainable params: 884K (0.36%)
  - BLEU-4: [nilai]
  - Kelebihan: [...]
  - Kekurangan: [...]

- **Adapter d=64 (Exp 2):**
  - Trainable params: 2.38M (0.95%)
  - BLEU-4: [nilai]
  - Improvement vs LoRA: [...]

- **Adapter d=128 (Exp 3 & 4):**
  - Trainable params: 9.5M (3.8%)
  - BLEU-4: [nilai]
  - Improvement vs d=64: [...]

**Pengaruh Ukuran Dataset:**
[Analisis Exp 3 vs Exp 4: apakah dataset lebih besar = performa lebih baik?]

### 4.3.5. Identifikasi Model Terbaik

Berdasarkan evaluasi komprehensif, **[Model X]** dipilih sebagai model terbaik dengan pertimbangan:
1. Metrik evaluasi tertinggi: [...]
2. Trade-off efisiensi: [...]
3. Stabilitas training: [...]

[Justifikasi detail]
```

---

## 4.4. Analisis Output Model Terbaik

### MUST HAVE:
1. **Contoh Input-Output**
   - Minimal 3-5 contoh dari berbagai kategori:
     - Knowledge+Code (Mudah, Sedang, Sulit)
     - No-Code (Mudah, Sedang, Sulit)
   - Format: Input → Ground Truth → Model Output
   - Highlight perbedaan (jika ada)

2. **Analisis Kualitas Pertanyaan**
   - Apakah pertanyaan sesuai konteks?
   - Apakah pertanyaan jelas dan tidak ambigu?
   - Apakah tingkat kesulitan sesuai?
   - Apakah format bahasa Indonesia benar (EYD)?

3. **Analisis Kualitas Jawaban**
   - Apakah jawaban benar sesuai konteks?
   - Apakah jawaban konsisten dengan pertanyaan?

4. **Analisis Kualitas Distraktor**
   - Apakah distraktor plausible (masuk akal)?
   - Apakah distraktor berbeda dari jawaban benar?
   - Apakah distraktor berbasis miskonsepsi umum?
   - Diversity: apakah 3 distraktor cukup berbeda?

5. **Perbandingan dengan Ground Truth**
   - Semantic similarity (BERTScore per-sample)
   - Lexical overlap (BLEU per-sample)
   - Kasus sukses vs kasus gagal

### SHOULD HAVE:
- Kategorisasi error types (jika ada)
- Analisis kuantitatif: berapa % output yang sempurna, acceptable, poor
- Visualisasi (word cloud distraktor, dll)

### Format Output:
```
## 4.4. Analisis Output Model Terbaik

[Paragraf pembuka: tujuan analisis kualitatif]

### 4.4.1. Contoh Output Model

**Contoh 1: Soal Knowledge+Code (Tingkat Mudah)**

**Input (Konteks):**
```
[konteks materi]
```

**Ground Truth:**
```
question: [...]
answer: [...]
distractors: [...] | [...] | [...]
```

**Model Output:**
```
question: [...]
answer: [...]
distractors: [...] | [...] | [...]
```

**Analisis:**
- Pertanyaan: [evaluasi kualitas]
- Jawaban: [evaluasi kualitas]
- Distraktor: [evaluasi kualitas]
- BERTScore: [nilai]
- BLEU-4: [nilai]

[Ulangi untuk 4-5 contoh lainnya dari berbagai kategori]

### 4.4.2. Analisis Kualitas Pertanyaan

[Analisis agregat dari semua contoh]
- Kesesuaian konteks: [...]
- Kejelasan: [...]
- Tingkat kesulitan: [...]
- Bahasa: [...]

### 4.4.3. Analisis Kualitas Jawaban

[Analisis agregat]
- Kebenaran: [...]
- Konsistensi: [...]

### 4.4.4. Analisis Kualitas Distraktor

[Analisis agregat]
- Plausibility: [...]
- Distinctiveness: [...]
- Miskonsepsi: [...]
- Diversity: [...]

### 4.4.5. Ringkasan Performa Kualitatif

**Tabel 5. Kategorisasi Kualitas Output (Sample: 50 output)**
| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Sempurna (semua komponen benar) | ... | ...% |
| Acceptable (minor issues) | ... | ...% |
| Poor (major issues) | ... | ...% |

[Interpretasi hasil]
```

---

## 4.5. Perbandingan dengan Model Lain

### MUST HAVE:
1. **Placeholder/Note**
   - Catatan bahwa bagian ini akan diisi setelah eksperimen komparatif
   - Daftar model yang akan dibandingkan:
     - IndoNanoT5 fine-tuned (model terbaik kita)
     - OpenRouter free tier model
     - GPT-4/GLM-4

2. **Template Tabel Komparatif**
   - Format tabel untuk perbandingan metrik
   - Format untuk perbandingan output kualitatif

3. **Metodologi Perbandingan (Planned)**
   - Test set yang sama untuk semua model
   - Metrik yang sama
   - Inference config yang comparable

### Format Output:
```
## 4.5. Perbandingan dengan Model Lain

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

### Template Hasil (To Be Filled)

**Tabel 6. Perbandingan Metrik dengan Model Lain**
| Model | BLEU-4 | ROUGE-L | BERTScore F1 | Distinct-1 | Distinct-2 | Inference Time |
|-------|--------|---------|--------------|------------|------------|----------------|
| IndoNanoT5-Adapter-d128 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| OpenRouter Free | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| GPT-4 / GLM-4 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Tabel 7. Perbandingan Kualitatif Output**
| Aspek | IndoNanoT5 | OpenRouter | GPT-4/GLM-4 |
|-------|------------|------------|-------------|
| Kualitas Pertanyaan | [TBD] | [TBD] | [TBD] |
| Kualitas Jawaban | [TBD] | [TBD] | [TBD] |
| Kualitas Distraktor | [TBD] | [TBD] | [TBD] |
| Adherence to Format | [TBD] | [TBD] | [TBD] |

### Contoh Output Komparatif (To Be Filled)

**Input:** [konteks]

**Ground Truth:** [...]

**IndoNanoT5 Output:** [TBD]

**OpenRouter Output:** [TBD]

**GPT-4/GLM-4 Output:** [TBD]

**Analisis:** [TBD]
```

---

## 4.6. Diskusi dan Limitasi

### MUST HAVE:
1. **Temuan Utama**
   - Ringkasan hasil eksperimen
   - Jawaban research questions (jika ada)
   - Kontribusi penelitian

2. **Implikasi Praktis**
   - Bagaimana hasil ini bisa digunakan dalam pendidikan?
   - Siapa yang bisa memanfaatkan sistem ini?
   - Deployment considerations

3. **Keterbatasan Penelitian**
   - Limitasi dataset (ukuran, domain, bahasa)
   - Limitasi evaluasi (hanya metrik otomatis, belum human evaluation)
   - Limitasi model (ukuran, kapasitas)
   - Limitasi resource (GPU, waktu training)

4. **Saran Penelitian Lanjutan**
   - Ekspansi dataset (lebih banyak topik, bahasa lain)
   - Human evaluation
   - Deployment dan user study
   - Eksplorasi metode PEFT lain (QLoRA, IA3, dll)

### SHOULD HAVE:
- Perbandingan dengan state-of-the-art (jika ada)
- Diskusi tentang generalizability
- Ethical considerations (bias, fairness)

### Format Output:
```
## 4.6. Diskusi dan Limitasi

### 4.6.1. Temuan Utama

Penelitian ini berhasil menunjukkan bahwa:
1. [Temuan 1]
2. [Temuan 2]
3. [Temuan 3]

[Elaborasi temuan]

### 4.6.2. Implikasi Praktis untuk Pendidikan

[Diskusi bagaimana sistem ini bisa digunakan]
- Use case 1: [...]
- Use case 2: [...]

[Deployment considerations]

### 4.6.3. Keterbatasan Penelitian

**Limitasi Dataset:**
- [Limitasi 1]
- [Limitasi 2]

**Limitasi Evaluasi:**
- [Limitasi 1]
- [Limitasi 2]

**Limitasi Model:**
- [Limitasi 1]
- [Limitasi 2]

**Limitasi Resource:**
- [Limitasi 1]
- [Limitasi 2]

### 4.6.4. Saran untuk Penelitian Lanjutan

1. **Ekspansi Dataset:** [...]
2. **Human Evaluation:** [...]
3. **Deployment dan User Study:** [...]
4. **Eksplorasi Metode PEFT Lain:** [...]
5. **Multilingual Extension:** [...]

[Elaborasi saran]
```

---

## Checklist Penulisan

### Sebelum Mulai Menulis:
- [ ] Ekstrak data metrik dari training reports (04, 05, 07, 08, 09, 10, 11, 12)
- [ ] Kumpulkan contoh dataset dari `dataset_aqg/dataset-task-v3/00-dataset/accumulated.jsonl`
- [ ] Siapkan loss curves/training logs
- [ ] Identifikasi model terbaik berdasarkan metrik

### Saat Menulis:
- [ ] 4.1: Tulis hasil dataset dengan contoh konkret
- [ ] 4.2: Tulis baseline performance dengan metrik
- [ ] 4.3: Tulis hasil evaluasi 4 eksperimen dengan tabel komparatif
- [ ] 4.4: Tulis analisis output model terbaik dengan 3-5 contoh
- [ ] 4.5: Tulis placeholder untuk perbandingan model lain
- [ ] 4.6: Tulis diskusi dan limitasi

### Setelah Menulis:
- [ ] Review konsistensi dengan bagian Metodologi
- [ ] Pastikan semua tabel dan gambar memiliki caption
- [ ] Pastikan semua referensi ke tabel/gambar benar
- [ ] Proofread bahasa Indonesia (EYD)
- [ ] Pastikan flow antar sub-bagian smooth

---

## Data Sources

### Untuk 4.1 (Dataset):
- `dataset_aqg/dataset-task-v3/00-dataset/accumulated.jsonl`
- `dataset_aqg/dataset-task-v4/00-dataset/accumulated.jsonl` (jika ada)
- Scripts validasi: `scripts/validate_dataset_design.py`

### Untuk 4.2 (Baseline):
- Baseline evaluation results (jika sudah ada)
- Atau perlu run baseline evaluation dulu

### Untuk 4.3 (Eksperimen):
- `docs/training-report/04-indonanot5-report4.md` (Exp 1: LoRA)
- `docs/training-report/05-indonanot5-report4.md` (Exp 2: Adapter d=64?)
- `docs/training-report/07-indonanoot5-report.md` (Exp 3?)
- `docs/training-report/08-indonanoot5-report.md` (Exp 4?)
- `docs/training-report/09-indonanoot5-report.md`
- `docs/training-report/10-indonanoot5-report.md`
- `docs/training-report/11-indonanoot5-report.md`
- `docs/training-report/12-indonanoot5-report.md`

### Untuk 4.4 (Output Analysis):
- Model terbaik checkpoint
- Inference results pada test set
- Perlu generate output examples

---

## Notes

- **Prioritas:** Tulis 4.1, 4.2, 4.3 terlebih dahulu (data sudah ada)
- **4.4:** Perlu generate output examples dari model terbaik
- **4.5:** Placeholder saja, akan diisi nanti
- **4.6:** Tulis setelah 4.1-4.4 selesai

- **Tone:** Formal, akademis, objektif
- **Bahasa:** Indonesia (EYD)
- **Visualisasi:** Gunakan tabel dan grafik untuk clarity
- **Referensi:** Cite metodologi dan related work jika relevan
