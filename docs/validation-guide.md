Biar saya baca dulu kode aktual yang ada sebelum menjawab.

Oke, sekarang saya punya gambaran lengkap dari kode aktual. Berikut jawabannya:

---

## Tahap 3.2: Preprocessing & Rekayasa Prompt

### 1. Chunking Strategy

**Algoritma (dari `chunker.py`):**

```
File .md → Split by heading (#/##/###) → Per section:
  ├── Pisahkan segmen: teks biasa vs code block (regex: ```...```)
  ├── Code block → TIDAK pernah dipotong, selalu utuh
  ├── Teks biasa → split di batas kalimat (regex: (?<=[.!?])\s+)
  │   └── Blockquote → split per baris
  └── Chunk < 50 token → merge ke chunk sebelumnya
```

**Contoh konkret chunk boundary:**

```
# Tipe Data Python          ← heading = section boundary
Teks paragraf 1...          ← chunk 1 (teks biasa)
                            ← boundary: token > 400
Teks paragraf 2...          ← chunk 2
```python                   ← code block: TIDAK dipotong
x = 10
```                         ← selalu satu chunk utuh
```

**Estimasi token:** `len(text.split()) × 1.3` (tanpa tokenizer penuh)

**Batas:** min=50 token, max=400 token per chunk

**Statistik distribusi** — tidak ada angka pasti di kode, tapi berdasarkan parameter:
- Min: 50 token (setelah merge)
- Max: 400 token (hard limit)
- Target range: 250–400 token (sweet spot)

---

### 2. Rekayasa Prompt (dari `prompt_constructor.py`)

**Template tunggal:**
```
Konteks: {context}

Prompt: Buat satu soal {question_type} tentang {concept},
tingkat kesulitan: {difficulty}, bahasa Indonesia.
```

**Tahap pembuatan prompt:**

1. **Context grounding** — pilih konsep paling relevan dengan isi chunk via `extract_concept_from_chunk()`:
   - Tokenisasi nama konsep → hitung keyword match di teks chunk
   - Bonus score jika nama konsep muncul sebagai substring
   - Fallback ke `section_heading` jika score = 0

2. **Kombinasi parameter** — setiap chunk menghasilkan N prompt:
   - `difficulty` × `question_type` = 3 × 1 = 3 prompt per chunk (default)
   - Contoh: chunk 1 → [easy/MCQ, medium/MCQ, hard/MCQ]

3. **Normalisasi path** — `source_file` dinormalisasi ke forward slash (cross-platform)

---

### 3. Validasi Output (dari `validator.py`)

Setiap output LLM divalidasi sebelum masuk dataset:

| Cek | Aturan |
|-----|--------|
| Panjang input | 50–600 token |
| Format target | Wajib ada: `Pertanyaan:`, `Jawaban benar:`, `Distraktor:` |
| Metadata fields | `difficulty`, `question_type`, `concept`, `misconception_tags` |
| Enum difficulty | hanya `easy/medium/hard` |
| Enum question_type | hanya `MCQ/Code Completion` |
| misconception_tags | list non-empty, minimal 1 tag |

---

### 4. Dataset Split (dari `dataset_writer.py`)

**Stratified split** berdasarkan `difficulty`:
- Task-specific: 70/15/15 (train/val/test)
- Domain: 80/10/10

Stratifikasi memastikan distribusi `easy/medium/hard` merata di setiap split. Jika group terlalu kecil (n=1 → semua ke train, n=2 → train+val).

---

### 5. Checkpoint & Resume

Pipeline menyimpan progress per-section sehingga bisa resume jika terputus:
- `checkpoint.jsonl` — section yang sudah selesai
- `progress.json` — index prompt terakhir yang diproses
- `accumulated.jsonl` — data valid yang sudah terkumpul


---

# flowcahrt Data Pipeline 

flowchart LR
    MD["📄 Markdown\nFiles"] 
    --> CHUNK["🔪 Chunking\n50–400 token\ncode block utuh"]
    --> PROMPT["✍️ Build Prompt\nKonteks + concept\n+ difficulty + type"]
    --> LLM["🤖 LLM Generate\nsoal MCQ/Code"]
    --> VAL{✅ Valid?}

    VAL -->|No| SKIP["🗑️ failures.jsonl"]
    VAL -->|Yes| SPLIT["📊 Split\n70/15/15\nstratified"]

    SPLIT --> TRAIN["train.jsonl\n~340 domain\n~1262 task"]

    TRAIN --> S1["🧠 Stage 1\nDomain Adaptation\n6 epoch lr=2e-4"]
    S1 --> M1["indot5-python\n-domain"]
    M1 --> S2["🎯 Stage 2\nTask-Specific AQG\n3 epoch lr=1e-4"]
    S2 --> M2["✨ indot5-python\n-aqg"]
