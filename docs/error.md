Generating 20 sample outputs...

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Matriks dapat digunakan untuk merepresentasikan berbagai data dalam kehidupan nyata, seperti gambar digital (pixel), tabel da...
Reference: question: Apa saja contoh penggunaan matriks dalam kehidupan nyata?
answer: Gambar digital, tabel data, graf, sistem persamaan linear
distractors: Han...
Prediction: question: bagaimana matriks dapat digunakan untuk merepresentasikan berbagai data dalam kehidupan nyata? answer: gambar digital (pixel), tabel data, g...
BLEU: 0.2405

--- Sample 2 ---
Input: buat_soal_pilihan_ganda: Dalam penerapan unit test, test case dapat menggunakan assertion untuk memverifikasi bahwa string tidak mengandung substring....
Reference: question: Apa yang dapat diverifikasi dengan assertion untuk substring tidak dalam string?
answer: Memastikan bahwa string tidak mengandung substring
...
Prediction: question: apa fungsi assertion dalam unit test case? answer: memastikan bahwa string tidak mengandung substring distractors: memastikan string berisi ...
BLEU: 0.1936

--- Sample 3 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
data = [3, 6, 9, 12, 15]
min_val = data[0]
for i in range(1, len(data)):
    if data[i] < ...
Reference: question: Perhatikan kode berikut:
```python
data = [3, 6, 9, 12, 15]
min_val = data[0]
for i in range(1, len(data)):
    if data[i] < min_val:
      ...
Prediction: question: perhatikan kode berikut: ```python data = [3, 6, 9, 12, 35] min_val = data[0] for i in range(1, len(data)): if data[i] < min-val: min_arri =...
BLEU: 0.6683

--- Sample 4 ---
Input: buat_soal_pilihan_ganda: Untuk membuat virtual environment, gunakan perintah python -m venv nama_env di terminal....
Reference: question: Bagaimana cara membuat virtual environment?
answer: python -m venv nama_env
distractors: create venv nama_env | new env nama_env | make venv...
Prediction: question: perintah apa yang digunakan untuk membuat virtual environment? answer: python -m venv nama_env di terminal distractors: perintah | perintah ...
BLEU: 0.3466

--- Sample 5 ---
Input: buat_soal_pilihan_ganda: Fungsi len() mengembalikan integer yang merepresentasikan jumlah elemen....
Reference: question: Perhatikan kode berikut:
```python
data = [10, 20, 30, 40, 50]
print(type(len(data)))
```
Apa output dari kode di atas?
answer: <class 'int'...
Prediction: question: perhatikan kode berikut: ```python data = [1, 2, 3, 4, 5] print(len(data)) *** data | error...
BLEU: 0.2181

--- Sample 6 ---
Input: buat_soal_pilihan_ganda: Multiple instance dari class yang sama dapat memiliki atribut yang berbeda-beda. Setiap instance memiliki copy sendiri dari a...
Reference: question: Perhatikan kode berikut:
```python
class Orang:
    def __init__(self, nama, umur):
        self.nama = nama
        self.umur = umur

orang...
Prediction: question: perhatikan kode berikut: ```python class = [1, 2, 3, 4, 5] class apa yang memiliki atribut berbeda? answer: 1 distractors: 2 | 3 | error...
BLEU: 0.0714

--- Sample 7 ---
Input: buat_soal_pilihan_ganda: Di mode interaktif, Anda dapat menggunakan dir() untuk melihat atribut dan method suatu objek....
Reference: question: Fungsi apa yang digunakan untuk melihat atribut objek di mode interaktif?
answer: dir()
distractors: list() | show() | view()...
Prediction: question: apa fungsi dir() dalam mode interaktif? answer: melihat atribut dan method suatu objek distractors: menghapus atribut | menghapus atribut...
BLEU: 0.1919

--- Sample 8 ---
Input: buat_soal_pilihan_ganda: Dalam penerapan unit test, test case dapat menggunakan parameter untuk menguji berbagai input. Ini memungkinkan satu test cas...
Reference: question: Apa manfaat menggunakan parameter dalam test case?
answer: Memungkinkan satu test case menguji banyak skenario tanpa duplikasi kode
distract...
Prediction: question: apa fungsi parameter dalam unit test case? answer: menguji berbagai input tanpa duplikasi kode distractors: membuat test case berjalan lebih...
BLEU: 0.3137

--- Sample 9 ---
Input: buat_soal_pilihan_ganda: Method rjust() bisa mengganti whitespace dengan karakter lain. Contoh: print('Dicoding'.rjust(20, '!')) akan menghasilkan '!!...
Reference: question: Perhatikan kode berikut:
```python
print('Dicoding'.rjust(20, '!'))
```
Apa output dari kode di atas?
answer: !!!!!!!!!!!!Dicoding
distracto...
Prediction: question: perhatikan kode berikut: ```python print('dicoding'.rjust(20, '!')) `'` apa output dari kode di atas? answer: true distractors: false | erro...
BLEU: 0.2495

--- Sample 10 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
data = [1, 2, 3, 4, 5, 6]
result = list(filter(lambda x: x % 2 == 0, data))
print(result)
...
Reference: question: Perhatikan kode berikut:
```python
data = [1, 2, 3, 4, 5, 6]
result = list(filter(lambda x: x % 2 == 0, data))
print(result)
```
Apa output ...
Prediction: question: perhatikan kode berikut: ```python data = [1, 2, 3, 4, 5, 6] result = list(filter(lambda x: x % 2 == 0, data)) print(result) '`` apa output ...
BLEU: 0.4714

--- Sample 11 ---
Input: buat_soal_pilihan_ganda: One-liner untuk pertukaran variabel lebih efisien dalam hal memori karena tidak memerlukan alokasi variabel bantuan temp. Pyt...
Reference: question: Mengapa one-liner lebih efisien dalam memori?
answer: Tidak memerlukan alokasi variabel bantuan
distractors: Menggunakan lebih banyak memori...
Prediction: question: mengapa one-liner lebih efisien dalam hal memori? answer: karena tidak memerlukan alokasi variabel bantuan temp distractors: karena lebih ce...
BLEU: 0.2717

--- Sample 12 ---
Input: buat_soal_pilihan_ganda: Library unittest menyediakan method assertRaises() untuk menguji bahwa fungsi melempar exception yang diharapkan. Ini penting...
Reference: question: Apa fungsi method assertRaises() dalam unittest?
answer: Menguji bahwa fungsi melempar exception yang diharapkan
distractors: Menguji bahwa ...
Prediction: question: apa yang dimaksud dengan method assertraises() dalam unittest? answer: menguji bahwa fungsi melempar exception yang diharapkan distractors: ...
BLEU: 0.3325

--- Sample 13 ---
Input: buat_soal_pilihan_ganda: Operasi matriks dengan konstanta bersifat komutatif untuk perkalian, artinya `A * k = k * A` dimana A adalah matriks dan k ad...
Reference: question: Apakah operasi `mat * 5` sama dengan `5 * mat` dalam NumPy?
answer: Ya, karena perkalian dengan konstanta bersifat komutatif
distractors: Ti...
Prediction: question: apa yang dimaksud dengan operasi matriks dengan konstanta? answer: `a * k = k * a` distractors: a = a | a * c | k...
BLEU: 0.0000

--- Sample 14 ---
Input: buat_soal_pilihan_ganda: Abstraksi data memungkinkan pemisahan antara apa yang dilakukan oleh data dan bagaimana data itu diimplementasikan....
Reference: question: Apa yang dipisahkan oleh konsep abstraksi data?
answer: Apa yang dilakukan data dan bagaimana data diimplementasikan
distractors: Keyboard d...
Prediction: question: apa yang dimaksud dengan abstraksi data? answer: pemisahan antara apa yang dilakukan oleh data dan bagaimana data diimplementasikan distract...
BLEU: 0.3190

--- Sample 15 ---
Input: buat_soal_pilihan_ganda: Dalam OOP, nama kelas menggunakan konvensi PascalCase dimana setiap kata dimulai dengan huruf kapital. Contohnya MobilSport, ...
Reference: question: Konvensi penamaan apa yang digunakan untuk nama kelas?
answer: PascalCase dimana setiap kata dimulai dengan huruf kapital
distractors: snake...
Prediction: question: apa yang dimaksud dengan pascalcase dalam oop? answer: setiap kata dimulai dengan huruf kapital distractors: hanya huruf kapital | hanya hur...
BLEU: 0.2380

--- Sample 16 ---
Input: buat_soal_pilihan_ganda: Method chaining bekerja dari kiri ke kanan....
Reference: question: Perhatikan kode berikut:
```python
print('hello'.replace('h', 'H').upper())
```
Apa output dari kode di atas?
answer: HELLO
distractors: Hel...
Prediction: question: perhatikan kode berikut: ```python data = [1, 2, 3, 4, 5, 6] print(data = data) print(list(data)))`` apa output dari kode di atas? answer: 1...
BLEU: 0.2945

--- Sample 17 ---
Input: buat_soal_pilihan_ganda: Google Colaboratory otomatis menyimpan perubahan notebook secara berkala....
Reference: question: Bagaimana Google Colaboratory menyimpan perubahan?
answer: Otomatis secara berkala
distractors: Harus manual Ctrl+S | Tidak menyimpan | Hany...
Prediction: question: perhatikan kode berikut: ```python data = [1, 2, 3, 4, 5] print(data = data) print(print(data)) *** data | error...
BLEU: 0.0000

--- Sample 18 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut:
```python
try:
    for i in range(3):
        print(f"Iterasi {i}")
        if i == 1:
            r...
Reference: question: Perhatikan kode berikut:
```python
try:
    for i in range(3):
        print(f"Iterasi {i}")
        if i == 1:
            raise StopIterat...
Prediction: question: perhatikan kode berikut: ```python try: for i in range(3): print(f"iterasi {i}") if i == 1: raise stopiteration("berhenti") except stopitera...
BLEU: 0.4189

--- Sample 19 ---
Input: buat_soal_pilihan_ganda: Perhatikan kode berikut yang menunjukkan fungsi dengan variabel lokal:
```python
def hitung():
    x = 10
    y = 20
    hasi...
Reference: question: Perhatikan kode berikut:
```python
def hitung():
    x = 10
    y = 20
    hasil = x + y
    return hasil

print(hitung())
```
Apa output da...
Prediction: question: perhatikan kode berikut: ```python def hitung(): x = 10 y = x + y return hasil print(hitung()) '`` apa output dari kode di atas? answer: 20 ...
BLEU: 0.6372

--- Sample 20 ---
Input: buat_soal_pilihan_ganda: Namespace adalah ruang nama yang memetakan nama ke objek. Setiap fungsi memiliki namespace lokalnya sendiri yang terpisah dar...
Reference: question: Apa yang dimaksud dengan namespace dalam fungsi?
answer: Ruang nama yang memetakan nama ke objek, terpisah dari namespace global
distractors...
Prediction: question: apa yang dimaksud dengan namespace? answer: ruang nama yang memetakan nama ke objek distractors: ruangan nama yang tidak memiliki namespace ...
BLEU: 0.2559

✓ Samples saved to /content/drive/MyDrive/dataset_aqg/evaluation_results_v3/sample_outputs.json
✓ 20 samples generated

=== Sample Outputs ===

--- Sample 1 ---
Input: buat_soal_pilihan_ganda: Matriks dapat digunakan untuk merepresentasikan berbagai data dalam kehidupan nyata, seperti gambar digital (pixel), tabel data, graf (adjacency matrix), atau sistem persamaan linear....