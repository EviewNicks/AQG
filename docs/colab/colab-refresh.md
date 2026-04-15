Ya, bisa. Dari terminal Colab kamu bisa kill semua proses Python yang sedang pakai GPU, yang efeknya sama seperti restart runtime tapi lebih cepat.

Jalankan di terminal Colab:

```bash
# Lihat proses yang pakai GPU
nvidia-smi

# Kill semua proses Python di GPU
sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# Verifikasi GPU sudah bersih
nvidia-smi
```

Setelah itu cek memory GPU sudah kosong — output `nvidia-smi` seharusnya menunjukkan `0MiB` di kolom Memory-Usage.

Tapi ada catatan penting: setelah kill proses, kernel Python di notebook juga mati. Jadi kamu tetap perlu jalankan ulang semua cell dari awal. Bedanya dengan terminate runtime adalah file di `/content/` masih ada, jadi tidak perlu extract zip ulang.

Urutan setelah kill:
1. Jalankan cell install dependencies
2. Mount Drive
3. Setup path (zip sudah ada, skip extract)
4. Load dataset
5. Setup model
6. Langsung ke training