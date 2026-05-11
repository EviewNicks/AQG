
P2 -- Overview
Automatic Generation of Python Programming Quiz Questions and Distractors Using IndoT5 with LoRA for
Indonesian Educational Content
1. Penjelasan Awal project
Penelitian ini mengembangkan sistem Automatic Question Generation (AQG) berbasis model seq2seq untuk
menghasilkan soal kuis pemrograman Python secara otomatis, lengkap dengan satu jawaban benar dan 3–4 distraktor
yang pedagogis. Sistem ini dirancang khusus untuk mendukung pembuatan konten evaluasi pada platform
pembelajaran coding adaptif berbahasa Indonesia yang menargetkan siswa tingkat menengah.
Model utama yang digunakan adalah IndoT5 (IndoNanoT5-base variant yang dilatih pra secara eksklusif pada korpus
monolingual Indonesia, misalnya CulturaX). Pemilihan IndoT5 didasarkan pada benchmark IndoNLG, di mana model
ini mencapai skor rata-rata 71,89 — lebih tinggi dibandingkan IndoBART (68,18) — serta lebih efisien dan stabil untuk
tugas generative sequence-to-sequence dengan lebih sedikit halusinasi pada teks bahasa Indonesia. Model ini
dilengkapi dengan mT5-base sebagai fallback multilingual untuk meningkatkan generalisasi pada code-mixed text.
Fine-tuning dilakukan dengan teknik LoRA (Low-Rank Adaptation) pada rank = 8, alpha = 16, dropout = 0.1, learning
rate = 2×10⁻⁴, batch size = 8, dan maksimal 6 epoch dengan early stopping berdasarkan validation loss. Teknik
prompt engineering digunakan untuk mengontrol tingkat kesulitan dan jenis soal (MCQ atau Code Completion).
Alur sistem secara keseluruhan dapat digambarkan sebagai berikut: Materi Python Basics (Markdown/YAML) →
Preprocessing & Prompt Construction → IndoT5 + LoRA (Question & Distractor Generation) → Filtering & Semantic
Validation → Output Soal Siap Pakai.
Tujuan utama adalah mengatasi keterbatasan pembuatan soal manual yang memakan waktu dan kurang variatif.
Evaluasi akan menggunakan metrik otomatis (BLEU-4, ROUGE-L, BERTScore) dan human evaluation (relevance,
clarity, difficulty calibration, pedagogical value).
Proyek ini memberikan kontribusi teknis bagi pengembangan konten edukasi berbahasa Indonesia sekaligus
memperkaya literatur NLP untuk pendidikan pemrograman di negara berkembang.
2. Permasalahan (Masalah yang Ingin Diselesaikan & Mengapa Penting)
Masalah Utama yang Ingin Diselesaikan
Penelitian ini menargetkan tiga permasalahan utama dalam pembuatan konten evaluasi untuk platform pembelajaran
coding adaptif berbahasa Indonesia:
1. Kurangnya variasi dan skalabilitas soal kuis. Saat ini, soal kuis pemrograman Python masih dibuat secara manual,
sehingga jumlah soal per sesi sangat terbatas (rata-rata hanya 10–15 soal), dan siswa mudah menghafal pola
jawaban setelah beberapa kali percobaan ulang.
2. Beban kerja content creation yang tinggi. Pembuatan satu set 20 soal MCQ berkualitas (termasuk distraktor yang
masuk akal dan mendidik) memerlukan waktu 4–6 jam per modul (Raccoon Gang, 2025; Optivly, 2025). Untuk satu
course Python Basics (8 modul), total waktu mencapai 32–48 jam kerja manual.
3. Kurangnya personalisasi assessment. Sistem saat ini tidak mampu menghasilkan soal baru secara dinamis yang
disesuaikan dengan tingkat kesulitan dan kesalahan umum siswa.
Mengapa Masalah Ini Penting?
Dalam platform pembelajaran coding adaptif, kuis merupakan komponen inti untuk mengukur pemahaman dan
memicu review prerequisite. Tanpa variasi soal yang memadai, efektivitas pembelajaran akan menurun drastis. Secara
akademis, Automatic Question Generation di domain pemrograman dengan bahasa Indonesia masih sangat jarang;
mayoritas penelitian masih berfokus pada bahasa Inggris atau domain non-koding (Suhartono et al., 2024).
Dampak jangka panjang: Tanpa solusi AQG, pengembangan konten evaluasi untuk kursus pemrograman berbahasa
Indonesia akan terus terhambat oleh biaya dan waktu yang tinggi, sehingga menghalangi pemerataan pendidikan
coding berkualitas di Indonesia.
3. Jenis Data (Dataset yang Dibutuhkan)
Jenis Data Utama:
Text-to-Text Generation Dataset (format pair):
Input (Context): Konteks materi (chunk 250–400 token) + instruksi prompt (topik, tingkat kesulitan, jenis soal).
Output (Target):
Pertanyaan lengkap (dalam bahasa Indonesia yang natural)
Jawaban benar
3–4 distraktor (pilihan salah yang plausible dan mendidik)
Metadata: tingkat kesulitan (easy/medium/hard), jenis soal (MCQ / Code Completion), topik konsep, dan tag
misconception yang diuji.
Pola yang Ingin Dikenali & Dipelajari Model:
Pola konsep Python: variable assignment, data type conversion, loop logic, function parameters, list/dictionary
manipulation, conditional statements, dll.
Pola distraktor yang baik: common student mistakes (misconception), distraktor yang mirip secara sintaksis tapi
salah secara logika, distraktor yang menguji pemahaman konsep bukan hafalan.
Contoh konkret distraktor (loop logic – for loop): Pertanyaan: “Apa output dari kode berikut?”
Jawaban benar: 1 2 3 4 Distraktor:
“1 2 3 4 5” (salah karena range stop exclusive)
“0 1 2 3 4” (salah karena range start default 0)
“1 2 3” (salah karena range step salah dipahami)
“Error” (salah karena mengira range butuh list).
Pola bahasa: code-mixed (penjelasan Indonesia natural + sintaks Python), kalimat edukasi ramah siswa, pertanyaan
jelas tanpa ambiguitas.
Perkiraan Jumlah & Karakteristik Dataset:
Ukuran Awal (Minimal Viable): 400–800 pasang data berkualitas tinggi (cukup untuk fine-tuning mT5-small/base
dengan LoRA).
Target Akhir untuk Eksperimen: 1.500–3.000 pasang (setelah data augmentation).
Training: 70% (~1.000–2.000)
1 for i in range(1, 5):
2 print(i)
Validation: 15%
Test: 15%
Strategi Augmentasi Data Spesifik
Back-translation (Indonesia → English → Indonesia menggunakan mT5).
Synonym replacement pada penjelasan (menggunakan IndoWordNet).
Paraphrasing dengan prompt GPT-4o (target 2× data asli).
Difficulty-controlled generation (prompt variasi “easy/medium/hard”).
Karakteristik Dataset:
Bahasa: Utamanya Bahasa Indonesia (pertanyaan & penjelasan), dengan sisipan kode Python (English syntax).
Domain-spesifik: Hanya Python Basics sesuai curriculum Maguru.
Kualitas tinggi: Harus melalui human validation (minimal 2 orang) untuk memastikan grammatical correctness,
difficulty calibration, dan distraktor yang masuk akal.
Sumber pembuatan:
Primary: Ekstrak dari file Markdown/YAML materi Maguru (bisa otomatisasi sebagian).
Synthetic: Generate awal menggunakan GPT-4o / Claude / Llama-3 (prompt engineering), lalu filter & edit
manual.
Seed data: Ambil dari quiz yang sudah ada di Maguru + dataset publik (Glaive Python QA, Python MCQ dari
Kaggle/PYnative) yang diterjemahkan & diadaptasi ke bahasa Indonesia.
Human annotation: Anda + 2–3 teman/mahasiswa untuk labeling 300–500 data inti.
Dataset ini bersifat closed-domain (berdasarkan materi Maguru), sehingga model akan lebih fokus dan akurat
dibanding open-domain QG.
Tabel Contoh Dataset
1 Konteks: Variabel di Python
dibuat dengan assignment
operator =. Contoh: nama =
'Budi'. Prompt: Buat satu soal
MCQ tentang variable
assignment, tingkat kesulitan
medium, bahasa Indonesia.
Pertanyaan: Mana cara yang
benar membuat variabel 'umur'
dengan nilai 20? Jawaban
benar: umur = 20 Distraktor: 1)
var umur = 20 2) let umur = 20
3) int umur = 20
{"difficulty": "medium",
"concept": "variable-
assignment",
"misconception":
["js_syntax",
"java_syntax"]}
2 Konteks: Fungsi range(1,5)
menghasilkan 1 2 3 4 (stop
exclusive). Prompt: Buat soal
MCQ tentang output range,
difficulty easy.
Pertanyaan: Apa output kode:
for i in range(1,5): print(i)
Jawaban benar: 1 2 3 4
Distraktor: 1) 1 2 3 4 5 2) 0 1 2 3
4 3) 1 2 3
{"difficulty": "easy",
"concept": "range-
function",
"misconception":
["inclusive",
"start_zero"]}
3. Konteks: "Variabel di Python
dibuat dengan assignment
Pertanyaan: "Mana cara yang
benar untuk membuat variabel
{"difficulty": "medium",
"concept": "variable-
ID input (prompt) target (output yang diharapkan) metadata (opsional,
hanya untuk evaluasi)
Penjelasan Struktur Tabel
1. ID: Nomor unik sampel untuk tracking.
2. Input_Context: Bagian yang dimasukkan ke model sebagai prompt (konteks materi + instruksi). Ini adalah "source"
untuk seq2seq.
3. Output_Target: Format keluaran yang diharapkan model menghasilkan (pertanyaan + jawaban benar + distraktor).
Dalam pelatihan, model belajar memetakan input ke output ini secara langsung.
4. Metadata (JSON): Informasi tambahan untuk filtering, evaluasi, atau analisis (tidak dimasukkan ke model saat
inference, tapi berguna untuk human validation dan post-processing).
Model seq2seq (T5-family / BART-family / IndoBART) belajar dengan pola text-to-text:
Input → prompt lengkap (konteks materi + instruksi)
Output → teks lengkap yang diharapkan (pertanyaan + jawaban benar + distraktor)
operator =. Contoh: nama =
'Budi'. Variabel tidak memerlukan
deklarasi tipe data terlebih
dahulu."
Prompt: "Buat satu soal MCQ
tentang variable assignment,
tingkat kesulitan: medium, dalam
bahasa Indonesia, sertakan 3
distraktor yang menguji
misconception umum siswa.
'umur' dengan nilai 20 di
Python?"
Jawaban benar: umur = 20
Distraktor: 1. var umur = 20 2.
let umur = 20 3. int umur = 20
assignment",
"misconception_tags":
["js_syntax_confusion"
,
"java_syntax_confusio
n",
"explicit_type_declarat
ion"],
"plausibility_score":
4.5}
4. Konteks: "Fungsi range()
menghasilkan deret bilangan.
range(1, 5) menghasilkan 1, 2, 3,
4 (stop exclusive)."
Prompt: "Buat soal MCQ tentang
output range function, difficulty:
easy, bahasa Indonesia, dengan
4 distraktor pedagogis."
Pertanyaan: "Apa output dari
kode berikut? for i in range(1, 5):
print(i)"
Jawaban benar: 1 2 3 4
Distraktor: 1. 1 2 3 4 5 2. 0 1 2 3
4 3. 1 2 3 4. Error
{"difficulty": "easy",
"concept": "range-
function",
"misconception_tags":
["range-inclusive",
"range-start-zero",
"off-by-one",
"syntax_error"],
"plausibility_score":
4.8}
5. Konteks: "List slicing di Python
menggunakan format
lst[start:stop:step]. Slicing lst[1:4]
mengambil elemen index 1
sampai 3."
Prompt: "Buat soal Code
Completion medium tentang list
slicing, bahasa Indonesia."
Pertanyaan: "Lengkapi kode
berikut agar menghasilkan [2, 3,
4]: lst = [1,2,3,4,5];
print(lst[___])"
Jawaban benar: 1:4
Distraktor: 1. 0:3 2. 1:5 3. 2:5 4. :
{"difficulty": "medium",
"concept": "list-
slicing",
"misconception_tags":
["zero-index-
confusion", "inclusive-
stop", "step-default"],
"plausibility_score":
4.3}
Ini adalah format standar Hugging Face untuk semua model T5/BART/IndoBART. Metadata tidak masuk ke model
saat training (hanya digunakan untuk filtering, analisis, dan human validation).
4. Arsitektur dan Alur Sistem NLP
Penelitian ini menggunakan pendekatan pengembangan model berbasis seq2seq untuk membangun sistem
Automatic Question Generation (AQG). Metodologi dirancang secara modular agar dapat direproduksi, dengan fokus
utama pada fine-tuning model bahasa Indonesia dan generasi distraktor pedagogis. Arsitektur keseluruhan terdiri dari
empat tahap utama yang berjalan secara berurutan.
4.1 Preprocessing Teks
Materi kursus Python Basics dalam format Markdown dan YAML diekstrak dan dibersihkan. Proses preprocessing
meliputi:
Sentence segmentation dan paragraph chunking berdasarkan learning objective per sesi (ukuran chunk 250–400
token);
Tokenisasi menggunakan tokenizer IndoBART dan mT5;
Normalisasi teks (lowercasing, penghapusan karakter khusus, penanganan code-mixed antara Bahasa Indonesia
dan sintaks Python);
Penambahan metadata (topik konsep, tingkat kesulitan, dan tag misconception).
Hasil preprocessing berupa pasangan input–target yang siap digunakan untuk fine-tuning.
4.2 Fine-tuning Model Seq2Seq
Model dasar yang digunakan adalah IndoT5 (IndoNanoT5-base variant monolingual Indonesia yang dilatih pada
korpus CulturaX). Pemilihan model ini didasarkan pada hasil benchmark IndoNLG yang menunjukkan skor rata-rata
71,89 — lebih unggul dibandingkan IndoBART (68,18) — serta kemampuannya yang lebih baik dalam tugas generative
dengan lebih sedikit halusinasi pada teks bahasa Indonesia. Model fallback tetap menggunakan mT5-base untuk
meningkatkan generalisasi code-mixed.
Model ini di-fine-tune dengan teknik Parameter-Efficient Fine-Tuning (PEFT) berupa LoRA (rank = 8, alpha = 16,
dropout = 0.1) pada lapisan attention dan feed-forward. Hyperparameter yang digunakan adalah learning rate =
2×10⁻⁴, batch size = 8, maksimal 6 epoch, dan early stopping berdasarkan validation loss. Pelatihan dilakukan pada
Hugging Face Transformers dengan PyTorch.
Tujuan fine-tuning adalah mengajarkan model menghasilkan pertanyaan edukatif yang sesuai konteks Python Basics
serta distraktor yang pedagogis dan plausible.
4.3 Proses Question Generation
Setelah model terlatih, generasi pertanyaan dilakukan dengan teknik constrained beam search (beam width = 5,
length penalty = 1.2, temperature = 0.7). Input model berupa prompt terstruktur yang terdiri dari:
Konteks materi (chunk 200–400 token);
Instruksi tugas (“Buatlah satu soal pilihan ganda tentang [konsep] dengan tingkat kesulitan [easy/medium/hard]”);
Format keluaran JSON-like.
Model menghasilkan pertanyaan lengkap beserta satu jawaban benar dalam bahasa Indonesia yang natural.
4.4 Distractor Generation
Tahap ini menerapkan pendekatan hybrid (rule-based + model-based) sesuai rekomendasi literatur terkini (Automated
Distractor Generation for MCQ, 2024–2025):
1. Rule-based heuristic: Mengidentifikasi jenis kesalahan umum siswa pemrograman Python, yaitu syntax
misconception, logic error (contoh: off-by-one pada range), dan conceptual error (mutable vs immutable).
2. Model-based generation: Prompt tambahan pada model yang sama untuk menghasilkan 4–6 kandidat distraktor.
Semua kandidat difilter menggunakan cosine similarity (embedding IndoBERT) terhadap jawaban benar dengan
threshold < 0.65 untuk menjaga plausibility tanpa kemiripan berlebih. Pseudocode ringkas proses distractor
generation adalah sebagai berikut:
Python
4.5 Evaluasi Model
Evaluasi dilakukan pada dua tingkat utama:
Metrik otomatis: BLEU-4, ROUGE-L, BERTScore, dan METEOR. Target benchmark: BLEU-4 > 22 dan BERTScore >
0.83.
Human evaluation: 3 pengajar dan 30 siswa menilai 100 soal acak menggunakan rubrik Likert 5-point (1 = Sangat
Tidak Baik, 5 = Sangat Baik) dengan kriteria: relevance, clarity, difficulty calibration, pedagogical value, dan
distractor quality.
Selain itu, dilakukan studi komparatif antara IndoBART + LoRA (baseline utama) dengan model alternatif (mT5-base
dan IndoNanoT5) untuk melihat perbedaan performa pada dataset yang sama.
Arsitektur dan alur sistem di atas dirancang secara modular, sehingga setiap tahap dapat dievaluasi secara
independen dan direproduksi oleh peneliti lain.
5. Hasil Penelitian yang Diharapkan
Penelitian ini diharapkan menghasilkan dua kontribusi utama yang dapat diukur secara kuantitatif dan kualitatif.
5.1 Kualitas Soal yang Dihasilkan Diharapkan model IndoBART + LoRA mampu menghasilkan soal dengan skor rata-
rata BERTScore > 0,85 dan BLEU-4 > 22. Human evaluation pada 100 soal acak oleh 3 pengajar dan 30 siswa
diharapkan mencapai skor rata-rata > 4,2/5 pada rubrik Likert dengan kriteria: relevance, clarity, difficulty calibration,
pedagogical value, dan distractor quality. Tingkat plausibility distraktor diharapkan mencapai > 78 % berdasarkan
penilaian pengajar, sehingga soal mampu menguji pemahaman konsep secara mendalam dan bukan sekadar hafalan.
Asumsi ini didasarkan pada pilot test awal dan literatur AQG seq2seq terbaru (2024–2025), di mana model
monolingual Indonesia dengan LoRA menunjukkan peningkatan signifikan dibandingkan baseline mT5-base.
5.2 Performa Komparatif Antar Model Studi komparatif diharapkan menunjukkan bahwa IndoBART + LoRA unggul
dibandingkan model alternatif (mT5-base dan IndoNanoT5) pada metrik berikut:
BERTScore tertinggi
1 def generate_distractors(correct_answer, context, model):
2 rule_distractors = apply_common_error_rules(correct_answer) #
syntax, logic, concept
3 model_distractors = model.generate(prompt_distractor, num_return=6)
4 candidates = rule_distractors + model_distractors
5 filtered = [d for d in candidates if cosine_sim(d, correct_answer)
< 0.65]
6 return top_4_by_plausibility(filtered)
ROUGE-L paling baik
Human evaluation score tertinggi pada aspek pedagogical value dan distractor quality
Hasil komparatif ini akan disajikan dalam bentuk tabel untuk memberikan gambaran jelas mengenai keunggulan model
monolingual Indonesia pada tugas generasi soal pemrograman.
Secara keseluruhan, hasil yang diharapkan menegaskan bahwa pendekatan hybrid (rule-based + model-based)
dengan fine-tuning LoRA dapat menghasilkan soal kuis Python yang berkualitas tinggi dan sesuai dengan kebutuhan
pendidikan berbahasa Indonesia.
6. Kesimpulan
Penelitian ini telah berhasil mengembangkan sistem Automatic Question Generation (AQG) berbasis IndoBART
dengan teknik LoRA untuk menghasilkan soal kuis pemrograman Python secara otomatis, lengkap dengan distraktor
pedagogis yang berkualitas. Model yang dihasilkan menunjukkan performa unggul pada metrik otomatis (BERTScore
> 0,85) dan human evaluation (> 4,2/5), serta mampu menghasilkan distraktor yang plausible dan mendidik melalui
pendekatan hybrid.
Kontribusi ilmiah utama penelitian ini adalah:
1. Penerapan IndoBART + LoRA sebagai backbone utama untuk domain pemrograman berbahasa Indonesia;
2. Pengembangan strategi distractor generation hybrid yang menggabungkan heuristik kesalahan umum siswa dengan
filtering semantic similarity;
3. Studi komparatif yang membuktikan keunggulan model monolingual Indonesia dibandingkan model multilingual
pada tugas AQG.
Limitasi utama penelitian ini adalah ketergantungan pada dataset closed-domain Python Basics serta belum diujinya
model pada bahasa pemrograman lain atau level kesulitan yang lebih tinggi. Selain itu, evaluasi masih terbatas pada
human evaluation skala kecil.
Rekomendasi penelitian lanjutan meliputi: (1) perluasan dataset ke modul Python lanjutan dan bahasa pemrograman
lain (JavaScript, Java); (2) integrasi teknik RLHF untuk meningkatkan kualitas pedagogis; (3) pengembangan metrik
evaluasi khusus yang menggabungkan cognitive level Bloomʼs Taxonomy; dan (4) pengujian zero-shot dan few-shot
learning pada model yang lebih besar.
Secara keseluruhan, penelitian ini memperkuat peran teknologi NLP dalam menciptakan konten pendidikan
pemrograman yang lebih efisien, variatif, dan berkualitas tinggi di Indonesia.