# Kasus B: Sentimen Ulasan Produk (Amazon Reviews 2023, Video Games)

Model klasifikasi biner yang menentukan apakah ulasan bernada **negatif** (rating 1-2) atau **positif** (rating 4-5). Rating 3 dibuang. Input berupa judul + teks ulasan. Metrik: **macro-F1** (rata-rata F1 kelas negatif dan positif).

## Metode

TF-IDF (unigram + bigram, maks. 300.000 fitur, `sublinear_tf`) + **Logistic Regression** dengan `class_weight="balanced"`. Setelah dilatih, *threshold* keputusan dicari di data validasi (rentang 0,20-0,80) untuk memaksimalkan macro-F1. Hasilnya threshold **0,33**, disimpan bersama model di `models/sentiment_model.joblib`.

## Alasan memilih metode ini

- Kelas timpang (negatif hanya ±17,5%). `class_weight="balanced"` dan tuning threshold menjaga model tidak hanya jago di kelas positif, sesuai tujuan macro-F1.
- Bigram menangkap frasa penting seperti "not good" atau "not bad" yang artinya berbeda dari kata tunggalnya.
- Ringan: baca data 18 detik, training 81 detik di CPU, ukuran model 13,3 MB. Mudah dijalankan ulang tanpa GPU.

## Cara membagi data

- Data dibaca per baris dari file `.gz` (tanpa dimuat utuh), hanya ulasan dengan timestamp sampai **31 Des 2019** (UTC, timestamp dalam milidetik).
- Dari 4.624.615 baris, diambil sampel acak 15% (`SAMPLE_RATE=0.15`, seed 42) lalu difilter dan diberi label: **422.211 ulasan** dipakai (82,5% positif, 17,5% negatif).
- **Split berdasarkan waktu**: latih = ulasan sebelum 2019 (359.082), validasi = ulasan tahun 2019 (63.129).
- Alasan: model dipakai untuk ulasan yang baru masuk, jadi validasi meniru kondisi itu (latih di masa lalu, uji di masa depan). Split acak membuat skor terlalu optimis karena gaya bahasa periode yang sama bocor ke data latih. Proporsi negatif di latih (0,175) dan validasi (0,174) hampir sama.

## Skor di data validasi (63.129 ulasan)

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Negatif | 0,9062 | 0,9156 | 0,9109 |
| Positif | 0,9821 | 0,9800 | 0,9811 |

**Macro-F1: 0,9460** (threshold 0,33). Dengan threshold default 0,50: 0,9402. Akurasi: 0,9688.

## Catatan dan keterbatasan

- Threshold dipilih di data validasi yang sama dengan tempat skor dilaporkan, jadi skor bisa sedikit optimis.
- Kata teratas model termasuk frasa seperti "one star" dan "five stars": sebagian penulis menyebut ratingnya di teks, sehingga model memanfaatkannya.
- Kesalahan umumnya pada ulasan campuran (memuji tapi mengeluh), ulasan berbahasa lain, dan label yang tidak cocok dengan isi teks. Kategori Video Games juga memuat aksesori (controller, headset, dll).
- Hanya 15% data yang dipakai. Menaikkan `SAMPLE_RATE` atau memakai model transformer berpotensi menaikkan skor.

## Cara menjalankan

1. `pip install -r requirements.txt`
2. Unduh `Video_Games.jsonl.gz` (review) dari [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io), taruh di `data/` (jangan di-extract).
3. Jalankan `notebook.ipynb` dari atas ke bawah. Model tersimpan di `models/sentiment_model.joblib`.
