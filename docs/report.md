# 3. Metodologi

## 3.1. Akuisisi & Deskripsi Dataset

### Tujuan
Mempersiapkan dataset berkualitas tinggi untuk fine-tuning model IndoT5 pada tugas pembuatan soal pilihan ganda (MCQ) dan pertanyaan pemrograman Python dalam bahasa Indonesia.

### Deskripsi Dataset

Dataset dikumpulkan dari platform pembelajaran dan diproses menjadi format terstruktur untuk task-specific instruction tuning. Dataset terdiri dari dua kategori utama:

**Komposisi Dataset:**
- **Knowledge + Code**: 5,662 entri (soal dengan kode dan konsep)
- **No-Code**: 3,515 entri (soal konseptual tanpa mengambil dataset sample yang punya kode blocks )
- **Total**: 5,662 entri

tujuanna mengaa kita memiliki 2 kategoru ytama datatste adlaha karna modle indonanot5 hanya dilatih dnegan domain bahasa indoensia saja , kurnag memiliki pemahaman terkiat text bahasa ingris ( untuk istilah python) dan pemahaman code yang baik , dengan membagi 2 jenis datset ini , kita melakuakn experiment apakha modle lazaaruzNLP/indonanot5 tetap dapat emmahami dna mmembuat strctured code di generated nya dengan baik atau tidak 

### Struktur Data

Setiap entri dataset memiliki struktur JSONL dengan tiga komponen utama:

```json
{
  "input": "buat_soal_pilihan_ganda: [konteks materi 1-2 kalimat]",
  "output": "question: [soal]\nanswer: [jawaban]\ndistractors: [opsi1] | [opsi2] | [opsi3]",
  "metadata": {
    "difficulty": "Mudah|Sedang|Sulit",
    "type": "knowledge|code"
  }
}
```

**Penjelasan Komponen:**
- **Input**: Konteks materi dengan instruksi tugas, berisi 1-2 kalimat penjelasan (50-200 kata)
- **Output**: Soal terstruktur dengan pertanyaan, jawaban benar, dan tiga distraktor yang plausibel
- **Metadata**:
  - `difficulty`: Tingkat kesulitan (Mudah = recall langsung, Sedang = aplikasi, Sulit = sintesis)
  - `type`: Jenis soal (knowledge = konseptual, code = dengan blok kode)

### Distribusi dan Validasi

**Distribusi Tipe Soal:**
- Knowledge: ≥ 60% (mencegah overfitting pada pola kode)
- Code: ≤ 40% (menjaga keseimbangan dengan soal konseptual)

**Distribusi Kesulitan:**
- Mudah: ~40%
- Sedang: ~45%
- Sulit: ~15%

**Validasi Kualitas:**
Dataset divalidasi melalui automated checks yang memverifikasi:
- Format struktur JSONL dan kelengkapan field (input, output, metadata)
- Kehadiran semua komponen output (question, answer, distractors)
- Kualitas distraktor (plausibel, distinct, berbasis miskonsepsi umum)
- Konsistensi bahasa Indonesia (EYD, tone formal/edukatif)
- Tidak ada duplikasi dalam input atau pertanyaan
- Distribusi tipe dan kesulitan sesuai target

Proses validasi mengikuti pipeline yang didokumentasikan yang  mencakup tahap analisis, pembersihan, penggabungan batch, dan penambahan metadata tipe.

### Sumber dan Cakupan

Dataset mencakup 60+ konsep pemrograman Python dari 11 modul pembelajaran, dengan stratifikasi berdasarkan topik untuk memastikan representasi seimbang di setiap kategori. Semua soal dirancang untuk konteks pendidikan tingkat menengah dengan fokus pada pemahaman konsep dan kemampuan coding praktis.
