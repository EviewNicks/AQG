# Adapter Training Analysis Report - IndoNanoT5 AQG

**Date:** April 2026  
**Model:** LazarusNLP/IndoNanoT5-base  
**Library:** `adapters` (NEW, successor of `adapter-transformers`)  
**Status:** ⚠️ NEEDS INVESTIGATION

---

## 📊 Executive Summary

Model berhasil dimuat dan adapter berhasil ditambahkan, TAPI ada **perbedaan signifikan** antara hasil aktual dengan dokumentasi target.

| Metric                  | Target (Dokumentasi)   | Hasil Aktual                     | Gap             |                       |
| -------------------------| ------------------------| ----------------------------------| -----------------| -----------------------|
| **Trainab----arams**    | 8.9M (3.6%)            | 2.38M (0.95%)                    | **-73%**        |                       |
| -*Adapter Dimension**   | d=64                   | ???                              | Unknown         |                       |
| *---------Method**      | AutoAdapterModel       | transformers + adapters.init()   | Fallback used   |                       |
| ----------------------- | ---------------------- | -------------------------------- | --------------- | # ✅ Apa yang BERHASIL |
## 1. Library Migration Success
- ✅ Migrasi dari `adapter-transformers` → `adapters` berhasil
- ✅ Model berhasil dimuat (via fallback method)
- ✅ Adapter berhasil ditambahkan
- ✅ Model siap untuk training

### 2. Fallback Mechanism Works
```
⚠ AutoAdapterModel failed: The state dictionary of the model you are trying to load is corrupted...
  Trying alternative: Load with transformers + adapters.init()

✓ Base model loaded with transformers + adapters.init()
✓ Adapter added: pfeiffer config, d=64
✓ Adapter activated for training
✓ Model moved to GPU
```

**Kesimpulan:** Fallback mechanism bekerja dengan baik, ini BUKAN error fatal.

---

## ⚠️ Apa yang PERLU INVESTIGASI

### Issue 1: Trainable Parameters Mismatch

**Expected (dari dokumentasi):**
```
Parameters:
  Trainable: 8,900,000 (3.6%)
  Total:     248,000,000
  Frozen:    239,100,000
```

**Actual (dari hasil running):**
```
Parameters:
  Trainable: 2,379,264 (0.95%)
  Total:     249,957,120
  Frozen:    247,577,856
```

**Gap Analysis:**
- Trainable params: **-73% lebih sedikit** dari target
- Percentage: **0.95%** vs target **3.6%**
- Ini bisa berarti:
  1. Adapter dimension tidak sesuai (bukan d=64?)
  2. Adapter config berbeda antara library lama vs baru
  3. Ada layer yang tidak ter-adapt

---

## 🔬 RESEARCH NEEDED

### Research Topic 1: Adapters Library Configuration for T5

**Yang perlu dicari:**

1. **Official Documentation:**
   - URL: https://docs.adapterhub.ml/
   - Cari: "T5 adapter configuration"
   - Cari: "Pfeiffer adapter parameters"
   - Cari: "reduction_factor calculation"

2. **Specific Questions:**
   - Bagaimana cara menghitung trainable parameters untuk T5 dengan adapter?
   - Apakah reduction_factor=12 menghasilkan d=64 untuk T5-base (768 hidden)?
   - Berapa expected trainable params untuk T5-base (248M) dengan Pfeiffer adapter d=64?

3. **Keywords untuk Search:**
   - "adapters library T5 trainable parameters"
   - "Pfeiffer adapter reduction factor T5"
   - "adapters.init() vs AutoAdapterModel T5"
   - "T5 adapter configuration calculation"

---

### Research Topic 2: Perbedaan adapter-transformers vs adapters

**Yang perlu dicari:**

1. **Migration Guide:**
   - URL: https://docs.adapterhub.ml/transitioning.html
   - Fokus: Breaking changes in adapter configuration
   - Fokus: Parameter calculation differences

2. **Specific Questions:**
   - Apakah ada perbedaan cara adapter ditambahkan ke T5 model?
   - Apakah `adapters.init()` menambahkan adapter dengan cara berbeda dari `AutoAdapterModel`?
   - Apakah ada config parameter yang berubah antara library lama dan baru?

3. **Keywords untuk Search:**
   - "adapter-transformers vs adapters parameter count"
   - "adapters.init() trainable parameters"
   - "AutoAdapterModel vs adapters.init() difference"

---

### Research Topic 3: T5 Model Architecture & Adapter Placement

**Yang perlu dicari:**

1. **T5 Architecture:**
   - Berapa jumlah layer di T5-base? (expected: 12 encoder + 12 decoder = 24 layers)
   - Berapa hidden dimension? (expected: 768)
   - Di layer mana adapter ditambahkan?

2. **Adapter Placement:**
   - Apakah adapter ditambahkan di semua layer?
   - Apakah adapter ditambahkan di encoder saja, decoder saja, atau keduanya?
   - Berapa parameter per adapter layer?

3. **Calculation Formula:**
   ```
   Expected formula:
   trainable_params = num_layers × adapter_params_per_layer
   
   For Pfeiffer adapter:
   adapter_params_per_layer = 2 × hidden_dim × bottleneck_dim
                            = 2 × 768 × 64
                            = 98,304 params per layer
   
   If 24 layers:
   total = 24 × 98,304 = 2,359,296 params
   
   ⚠️ This is CLOSE to our actual result (2,379,264)!
   ```

4. **Keywords untuk Search:**
   - "T5-base architecture layers"
   - "Pfeiffer adapter parameter calculation"
   - "adapter placement T5 encoder decoder"
   - "T5 adapter trainable parameters formula"

---

### Research Topic 4: Verifikasi Adapter Configuration

**Yang perlu dicari:**

1. **Code Examples:**
   - Cari contoh code penggunaan `adapters` library dengan T5
   - Cari expected output trainable parameters
   - Bandingkan dengan hasil kita

2. **GitHub Issues:**
   - Search di: https://github.com/adapter-hub/adapters/issues
   - Keywords: "T5 trainable parameters"
   - Keywords: "reduction_factor 12"
   - Cari apakah ada user lain dengan issue serupa

3. **Community Discussions:**
   - HuggingFace Forums
   - AdapterHub Discussions
   - Cari: "T5 adapter 3.6% trainable parameters"

---

## 💻 Kode Implementasi Saat Ini

### File: `src/finetuned/utils/adapter_loader.py`

```python
def load_model_with_adapter(
    model_name: str = 'LazarusNLP/IndoNanoT5-base',
    adapter_name: str = 'mcq_generation',
    adapter_config: str = 'pfeiffer',
    reduction_factor: int = 12,  # ← Target: d=64
    non_linearity: str = 'relu',
    device: str = 'cuda'
):
    # Method 1: AutoAdapterModel (FAILED)
    try:
        model = AutoAdapterModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        # Method 2: transformers + adapters.init() (SUCCESS)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        adapters.init(model)  # ← Initialize adapter support
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure adapter
    config = AdapterConfig.load(
        adapter_config,  # 'pfeiffer'
        reduction_factor=reduction_factor,  # 12
        non_linearity=non_linearity  # 'relu'
    )
    
    # Add adapter
    model.add_adapter(adapter_name, config=config)
    model.train_adapter(adapter_name)
    
    return model, tokenizer
```

**Pertanyaan untuk Research:**
1. Apakah `adapters.init()` menambahkan adapter dengan benar?
2. Apakah `reduction_factor=12` diterapkan dengan benar?
3. Apakah perlu parameter tambahan untuk T5 model?

---

## 🎯 Hipotesis & Kemungkinan Penyebab

### Hipotesis 1: Adapter Hanya di Encoder ATAU Decoder (BUKAN Keduanya)

**Reasoning:**
- T5 punya 12 encoder layers + 12 decoder layers = 24 total
- Jika adapter hanya di encoder (12 layers):
  - 12 × 98,304 = 1,179,648 params
- Jika adapter di encoder + decoder (24 layers):
  - 24 × 98,304 = 2,359,296 params ← **MATCH dengan hasil kita!**

**Kesimpulan Sementara:** 
Adapter kemungkinan sudah ditambahkan di semua layer (encoder + decoder), dan hasil 2.38M params adalah **BENAR** untuk d=64!

**Yang perlu diverifikasi:**
- Apakah dokumentasi salah (target 8.9M untuk d=128, bukan d=64)?
- Apakah kita perlu ubah reduction_factor untuk dapat 8.9M params?

---

### Hipotesis 2: Dokumentasi Mengacu pada d=128 (Bukan d=64)

**Calculation:**
```
For d=128 (reduction_factor=6):
adapter_params_per_layer = 2 × 768 × 128 = 196,608
total (24 layers) = 24 × 196,608 = 4,718,592 params

For d=256 (reduction_factor=3):
adapter_params_per_layer = 2 × 768 × 256 = 393,216
total (24 layers) = 24 × 393,216 = 9,437,184 params ← CLOSE to 8.9M!
```

**Kesimpulan:** 
Target 8.9M (3.6%) kemungkinan untuk **d=256**, bukan d=64!

---

## 📋 Action Items untuk User

### 1. Research Documentation
- [ ] Baca https://docs.adapterhub.ml/ bagian T5 configuration
- [ ] Cari formula calculation trainable parameters
- [ ] Verifikasi apakah 2.38M params benar untuk d=64

### 2. Verify Configuration
- [ ] Cek apakah reduction_factor=12 → d=64 benar
- [ ] Cari contoh code dengan expected output
- [ ] Bandingkan dengan hasil kita

### 3. Check GitHub Issues
- [ ] Search "T5 trainable parameters" di adapter-hub/adapters
- [ ] Cari apakah ada user lain dengan hasil serupa
- [ ] Cek apakah ini expected behavior

### 4. Clarify Target
- [ ] Apakah target 3.6% (8.9M) untuk d=64 atau d=256?
- [ ] Apakah dokumentasi artikel salah?
- [ ] Apakah kita perlu adjust reduction_factor?

---

## 🔗 Useful Links untuk Research

1. **Adapters Library Docs:** https://docs.adapterhub.ml/
2. **Migration Guide:** https://docs.adapterhub.ml/transitioning.html
3. **T5 Model Card:** https://huggingface.co/LazarusNLP/IndoNanoT5-base
4. **GitHub Repo:** https://github.com/adapter-hub/adapters
5. **Houlsby et al. Paper:** https://arxiv.org/abs/1902.00751

---

## 📝 Notes

- Model **BISA digunakan** untuk training dengan konfigurasi saat ini
- Hasil 2.38M (0.95%) mungkin **BENAR** untuk d=64
- Perlu verifikasi apakah target 8.9M (3.6%) untuk d=128 atau d=256
- Fallback mechanism bekerja dengan baik, tidak perlu khawatir tentang warning

---

**Status:** Waiting for research results from user.
