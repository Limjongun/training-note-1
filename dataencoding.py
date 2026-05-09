# ============================================================
# DATA ENCODING
# ============================================================

# Data encoding adalah proses mengubah data kategorikal
# menjadi data numerik agar bisa diproses oleh machine learning.
#
# Contoh data kategorikal:
# color = red, blue, green
#
# Model machine learning biasanya tidak bisa membaca teks langsung,
# jadi teks tersebut perlu diubah menjadi bentuk angka.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns


# ============================================================
# 1. MEMBUAT DATAFRAME
# ============================================================

# Membuat dataframe sederhana yang berisi kolom color
# Kolom color adalah data kategorikal karena isinya berupa label/kategori
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue']
})

# Menampilkan 5 data pertama
print(df.head())


# ============================================================
# 2. MEMBUAT INSTANCE ONE HOT ENCODER
# ============================================================

# OneHotEncoder digunakan untuk mengubah data kategorikal
# menjadi beberapa kolom biner 0 dan 1.
#
# Contoh:
# red   -> [0, 0, 1]
# blue  -> [1, 0, 0]
# green -> [0, 1, 0]
#
# Setiap kategori akan dibuat menjadi kolom baru.
encoder = OneHotEncoder()


# ============================================================
# 3. FIT DAN TRANSFORM DATA
# ============================================================

# fit_transform() melakukan dua hal:
#
# fit:
# Mempelajari kategori unik yang ada di kolom color
# Misalnya: blue, green, red
#
# transform:
# Mengubah kategori tersebut menjadi angka 0 dan 1
#
# df[['color']] ditulis dengan double bracket agar bentuknya tetap DataFrame,
# bukan Series.
#
# toarray() digunakan karena hasil OneHotEncoder awalnya berbentuk sparse matrix.
# Sparse matrix adalah format hemat memori untuk data yang banyak angka 0.
encoded = encoder.fit_transform(df[['color']]).toarray()

# Menampilkan hasil encoding dalam bentuk array
print(encoded)


# ============================================================
# 4. MEMBUAT DATAFRAME HASIL ENCODING
# ============================================================

# get_feature_names_out() digunakan untuk mengambil nama kolom hasil encoding.
#
# Karena kolom awal bernama color,
# hasilnya biasanya:
# color_blue
# color_green
# color_red
encoder_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out()
)

# Menampilkan dataframe hasil one hot encoding
print(encoder_df)


# ============================================================
# 5. TRANSFORM DATA BARU
# ============================================================

# Setelah encoder belajar dari data awal,
# kita bisa mengubah data baru menggunakan transform().
#
# Contoh di sini:
# Mengubah kategori 'blue' menjadi bentuk one hot encoding.
#
# Jangan pakai fit_transform() lagi untuk data baru,
# karena fit_transform() akan belajar ulang dari data baru.
# Untuk data baru, cukup pakai transform().
blue = encoder.transform([['blue']]).toarray()

# Menampilkan hasil encoding untuk blue
print(blue)


# ============================================================
# 6. MENGGABUNGKAN DATA ASLI DENGAN DATA HASIL ENCODING
# ============================================================

# pd.concat() digunakan untuk menggabungkan dataframe asli
# dengan dataframe hasil encoding.
#
# axis=1 berarti digabungkan secara horizontal / berdasarkan kolom.
x = pd.concat([df, encoder_df], axis=1)

# Menampilkan dataframe akhir
print(x)


# ============================================================
# KESIMPULAN
# ============================================================

# One Hot Encoding cocok digunakan untuk data kategorikal nominal.
#
# Nominal berarti kategori yang tidak punya urutan/hirarki.
# Contoh:
# - warna
# - kota
# - jenis kelamin
# - nama produk
#
# Untuk data yang punya urutan/hirarki seperti:
# degree = SMA, S1, S2, S3
# level = low, medium, high
#
# Biasanya lebih cocok menggunakan Ordinal Encoding,
# bukan One Hot Encoding.