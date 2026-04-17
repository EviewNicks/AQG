Research report-nya solid dan arahnya sudah benar. Tapi ada beberapa hal yang perlu dipertimbangkan lebih dalam, terutama mengingat hasil training kita yang loss-nya 36+.

**Yang sudah tepat dari research:**

Strategi dua tahap (domain adaptation → task-specific) memang best practice untuk kasus ini. IndoNanoT5 sebagai pilihan arsitektur juga justified — encoder-decoder lebih cocok untuk structured output seperti QA generation dibanding decoder-only.

---

**Yang perlu kita diskusikan lebih dalam:**

**1. Apakah span corruption benar-benar perlu untuk kita?**

Research menyebut span corruption sebagai objective utama domain adaptation T5. Tapi ada masalah praktis: IndoNanoT5 sudah di-pre-train dengan span corruption pada CulturaX. Melakukan span corruption lagi dengan dataset 161 sampel kemungkinan besar tidak akan mengubah representasi internal model secara signifikan — terlalu sedikit sinyal.

Alternatif yang lebih pragmatis: fokus ke `qa_generic` dan `summarization` saja, karena keduanya supervised dan lebih langsung mengajarkan pola domain Python.

**2. Ukuran dataset kita jauh di bawah threshold**

Research menyebut "Indonesian Python tutorials (Markdown/YAML)" sebagai sumber data. Kita punya 11 modul, tapi setelah filter + dedup hanya 161 sampel train. Ini masalah utama. Beberapa opsi untuk menambah data:

- Aktifkan format `summarization` via LLM — bisa menambah 200-400 sampel dari materi yang sama
- Augmentasi `qa_generic`: satu chunk bisa menghasilkan lebih dari 1 QA pair (bold term + inline code + heading sekaligus)
- Pertimbangkan sumber eksternal: dokumentasi Python resmi dalam bahasa Indonesia, artikel Dicoding/Petani Kode

**3. Ada pertanyaan strategis yang perlu dijawab dulu**

Apakah domain adaptation ini benar-benar diperlukan untuk proyek ini, atau kita bisa langsung ke Stage 2 (task-specific AQG)?

Argumen untuk skip domain adaptation: IndoNanoT5 sudah cukup "tahu" bahasa Indonesia. Dengan dataset AQG yang cukup berkualitas, model mungkin bisa langsung belajar pola QA Python tanpa perlu Stage 1. Beberapa paper (termasuk referensi no. 2 di report) berhasil dengan LoRA langsung tanpa domain adaptation.

Argumen untuk tetap lakukan: jika dataset AQG kita juga kecil, domain adaptation membantu model tidak "kaget" dengan terminologi Python saat Stage 2.

---

**Rekomendasi saya:**

Sebelum invest lebih banyak waktu di domain adaptation, coba dulu eksperimen kecil: langsung fine-tune IndoNanoT5 base dengan dataset AQG Stage 2 yang ada. Bandingkan hasilnya dengan model yang sudah melalui domain adaptation. Kalau perbedaannya tidak signifikan, kita bisa simplifikasi pipeline dan fokus ke kualitas dataset AQG.

Mau lanjut ke arah mana — perbaiki domain adaptation dulu, atau coba eksperimen langsung ke Stage 2