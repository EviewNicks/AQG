Baik, saya akan jelaskan cara penilaian BLEU Score dan bagaimana sistem evaluasi kita bekerja secara ringkas.

Baik, saya akan jelaskan secara ringkas cara penilaian BLEU Score dan menjawab pertanyaan Anda:

## Penjelasan BLEU Score dalam Sistem Kita

### 1. **Apa yang Dibandingkan BLEU?**

BLEU Score membandingkan **SELURUH OUTPUT** (bukan hanya input):

```
REFERENCE (Ground Truth):
"question: Bagaimana cara membuat virtual environment?
answer: python -m venv nama_env
distractors: create venv nama_env | new env nama_env | make venv nama_env"

PREDICTION (Model Output):
"question: bagaimana cara membuat virtual environment?
answer: menggunakan perintah python -m venv nama_env di terminal
distractors: menggunakan loop | menggunakan perintah lse | tidak menggunakan perintah"
```

**BLEU membandingkan keseluruhan string** (question + answer + distractors) sebagai **SATU TEXT SEQUENCE**.

### 2. **Cara Kerja BLEU Score**

BLEU mengukur **n-gram overlap** (kesamaan kata/frasa):

- **BLEU-1**: Unigram (kata tunggal) - "bagaimana", "cara", "membuat"
- **BLEU-2**: Bigram (2 kata) - "bagaimana cara", "cara membuat"
- **BLEU-3**: Trigram (3 kata) - "bagaimana cara membuat"
- **BLEU-4**: 4-gram - "bagaimana cara membuat virtual"

**Formula sederhana:**
```
BLEU = (jumlah n-gram yang cocok) / (total n-gram di prediction)
```

### 3. **Apakah BLEU Melihat Question, Answer, Distractors Terpisah?**

**TIDAK!** BLEU melihat **SELURUH OUTPUT sebagai satu string panjang**.

**Contoh Sample 4:**
- Reference: `"question: Bagaimana cara... answer: python -m venv... distractors: create venv..."`
- Prediction: `"question: bagaimana cara... answer: menggunakan perintah... distractors: menggunakan loop..."`
- **BLEU: 0.3336** → Artinya ~33% n-gram cocok di SELURUH output

### 4. **Apakah Ini Cukup untuk Evaluasi MCQ?**

**TIDAK CUKUP!** Ini adalah **KETERBATASAN** sistem evaluasi kita saat ini:

✅ **Yang BLEU ukur:**
- Kesamaan lexical (kata-kata yang sama)
- Format output (apakah ada "question:", "answer:", "distractors:")

❌ **Yang BLEU TIDAK ukur:**
- Apakah answer **benar** sesuai konteks?
- Apakah distractors **plausible** (masuk akal)?
- Apakah distractors **berbeda** dari answer?
- Apakah question **relevan** dengan input?

### 5. **Mengapa Kita Perlu Evaluasi Terpisah?**

**Anda BENAR!** Kita seharusnya evaluasi:

1. **Question Quality**: Apakah pertanyaan sesuai konteks input?
2. **Answer Correctness**: Apakah jawaban benar?
3. **Distractor Quality**: 
   - Apakah plausible (tidak terlalu mudah)?
   - Apakah berbeda dari answer?
   - Apakah ada duplikasi?

**Ini yang kita lakukan di Section 4.4** (analisis kualitatif manual), tapi **BELUM otomatis**.

### 6. **Solusi untuk Evaluasi Lebih Baik**

Untuk penelitian lanjutan, kita bisa:

1. **Parse output** menjadi komponen:
   ```python
   parsed = {
       "question": "...",
       "answer": "...",
       "distractors": ["...", "...", "..."]
   }
   ```

2. **Evaluasi terpisah**:
   - BLEU untuk question vs reference question
   - Exact match untuk answer vs reference answer
   - Diversity score untuk distractors
   - Semantic similarity (BERTScore) untuk masing-masing komponen

3. **Human evaluation** (gold standard):
   - Expert review untuk quality
   - Student testing untuk difficulty

### Kesimpulan

**BLEU Score saat ini:**
- Mengukur **keseluruhan output** sebagai satu string
- Tidak membedakan question, answer, distractors
- Hanya mengukur **lexical overlap**, bukan **semantic correctness**

**Ini adalah LIMITASI** yang harus kita sebutkan di Section 4.6 (Diskusi dan Limitasi) artikel!

Apakah penjelasan ini membantu? Apakah Anda ingin saya tambahkan penjelasan ini ke dokumentasi atau artikel?