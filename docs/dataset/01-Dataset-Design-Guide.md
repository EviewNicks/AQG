# Rancangan Skema JSON untuk Dataset

Berdasarkan materi `perkenalan-pythn.md` dan format dataset yang telah didefinisikan, berikut adalah rancangan skema JSON yang dapat digunakan untuk setiap pasang data (input-target) dalam dataset Anda. Skema ini dirancang untuk memfasilitasi proses *fine-tuning* model IndoT5 dan integrasi dengan *frontend* melalui API.

```json
{
  "input": "<konteks materi dari perkenalan-pythn.md>\n\n<prompt instruksi untuk generasi pertanyaan>",
  "target": "Pertanyaan: <pertanyaan kuis>? Jawaban benar: <jawaban>. Distraktor: 1) <distraktor 1> 2) <distraktor 2> 3) <distraktor 3> 4) <distraktor 4>",
  "metadata": {
    "difficulty": "<easy|medium|hard>",
    "question_type": "<MCQ|Code Completion>",
    "concept": "<konsep yang diuji, misal: Definisi Python, Versi Python, PEP>",
    "misconception_tags": [
      "<tag miskonsepsi 1>",
      "<tag miskonsepsi 2>"
    ]
  }
}
```

## Penjelasan Komponen Skema JSON

> **Catatan Penting (Evaluasi)**: `target` HARUS berupa **plain string**, bukan nested JSON object. Model IndoT5 adalah seq2seq yang belajar memetakan teks ke teks. Jika `target` berupa JSON object, model tidak bisa belajar dengan benar. `metadata` disimpan sebagai kolom terpisah dan tidak dimasukkan ke model saat training — hanya digunakan untuk filtering, analisis, dan human validation.

*   **`input` (String)**:
    *   Berisi gabungan dari **konteks materi** dan **prompt instruksi**. Konteks materi akan diambil dari paragraf atau bagian relevan dari file Markdown. Prompt instruksi akan memandu model untuk menghasilkan pertanyaan dan *distractor*.
    *   **Contoh Konteks**: "Python adalah bahasa pemrograman multifungsi yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR). Beliau membuat Python sebagai bahasa pemrograman yang mudah dibaca dan dimengerti (*readable*) serta memiliki kemampuan penanganan kesalahan (*exception handling*)."
    *   **Contoh Prompt**: "Buatlah soal pilihan ganda tentang [Definisi Python] dengan tingkat kesulitan [mudah]. Sertakan jawaban benar dan 3 distraktor pedagogis."

*   **`target` (String — plain text)**:
    *   Teks lengkap yang diharapkan dihasilkan model, dalam format natural yang konsisten.
    *   Format: `"Pertanyaan: ...? Jawaban benar: ... Distraktor: 1) ... 2) ... 3) ... 4) ..."`
    *   Ini adalah satu-satunya kolom yang digunakan sebagai label saat fine-tuning.

*   **`metadata` (Object — tidak masuk ke model)**:
    *   **`difficulty`**: Tingkat kesulitan soal (`easy`, `medium`, `hard`).
    *   **`question_type`**: Tipe pertanyaan (`MCQ` atau `Code Completion`).
    *   **`concept`**: Konsep utama yang diuji. Contoh: "Definisi Python", "Sejarah Python", "Fitur Python 2.x".
    *   **`misconception_tags`**: Tag miskonsepsi umum yang diatasi oleh *distractor*. Contoh: `["salah_paham_versi", "salah_paham_pep"]`.

## Contoh Implementasi dari `perkenalan-pythn.md`

Berikut adalah contoh bagaimana satu paragraf dari materi dapat diubah menjadi format JSON ini:

**Konteks Materi:**
"Python adalah bahasa pemrograman multifungsi yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR). Beliau membuat Python sebagai bahasa pemrograman yang mudah dibaca dan dimengerti (readable) serta memiliki kemampuan penanganan kesalahan (exception handling)."

**JSON Output (Format yang Benar untuk Fine-tuning):**
```json
{
  "input": "Konteks: Python adalah bahasa pemrograman multifungsi yang dirilis pada tahun 1991 oleh Guido van Rossum (GvR). Beliau membuat Python sebagai bahasa pemrograman yang mudah dibaca dan dimengerti (readable) serta memiliki kemampuan penanganan kesalahan (exception handling).\n\nPrompt: Buatlah satu soal pilihan ganda (MCQ) tentang pencipta bahasa Python berdasarkan konteks di atas. Tingkat kesulitan: mudah. Sertakan 1 jawaban benar dan 3 distraktor yang masuk akal.",
  "target": "Pertanyaan: Siapakah tokoh yang menciptakan bahasa pemrograman Python pada tahun 1991? Jawaban benar: Guido van Rossum. Distraktor: 1) Dennis Ritchie 2) James Gosling 3) Bjarne Stroustrup",
  "metadata": {
    "difficulty": "easy",
    "question_type": "MCQ",
    "concept": "Sejarah Python",
    "misconception_tags": ["tokoh_pemrograman_lain"]
  }
}
```

Format JSONL ini ideal karena:
1. **Kompatibel dengan fine-tuning**: `target` sebagai plain string langsung bisa dipakai sebagai label oleh Hugging Face Trainer.
2. **Fleksibel**: Mudah dikonversi ke Hugging Face `datasets.Dataset` dengan `load_dataset("json", ...)`.
3. **Metadata terpisah**: Tidak mencemari training signal — hanya dipakai untuk filtering dan analisis.
4. **Cukup untuk tahap dataset**: Integrasi API (Next.js, FastAPI) adalah urusan nanti, bukan sekarang.
