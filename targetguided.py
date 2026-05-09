# ============================================================
# TARGET GUIDED ORDINAL ENCODING
# ============================================================

# Target Guided Ordinal Encoding adalah teknik encoding
# untuk mengubah data kategorikal menjadi angka berdasarkan nilai target.
#
# Dalam contoh ini:
# - city adalah fitur kategorikal
# - price adalah target/nilai numerik
#
# Setiap kota akan diganti dengan rata-rata price dari kota tersebut.
#
# Contoh:
# jakarta punya price 200 dan 100
# rata-rata jakarta = (200 + 100) / 2 = 150
#
# Maka jakarta akan di-encode menjadi 150.

import pandas as pd


# ============================================================
# 1. MEMBUAT DATAFRAME
# ============================================================

# Membuat dataset sederhana
# city berisi data kategorikal
# price berisi nilai target/nilai numerik yang akan dijadikan panduan encoding
df = pd.DataFrame({
    'city': ['jakarta', 'surabaya', 'boston', 'sydney', 'paris', 'denver', 'jakarta'],
    'price': [200, 150, 300, 250, 180, 320, 100]
})

# Menampilkan dataframe awal
print(df)


# ============================================================
# 2. MENGHITUNG RATA-RATA TARGET BERDASARKAN KATEGORI
# ============================================================

# groupby('city') digunakan untuk mengelompokkan data berdasarkan kota
# ['price'].mean() digunakan untuk menghitung rata-rata price tiap kota
# to_dict() mengubah hasilnya menjadi dictionary agar mudah dipakai mapping
#
# Hasil contoh:
# {
#   'boston': 300,
#   'denver': 320,
#   'jakarta': 150,
#   'paris': 180,
#   'surabaya': 150,
#   'sydney': 250
# }

meann = df.groupby('city')['price'].mean().to_dict()

# Menampilkan hasil rata-rata price setiap city
print(meann)


# ============================================================
# 3. MELAKUKAN TARGET GUIDED ENCODING
# ============================================================

# map(meann) digunakan untuk mengganti setiap nama kota
# dengan rata-rata price berdasarkan dictionary meann.
#
# Jadi:
# jakarta  -> 150
# surabaya -> 150
# boston   -> 300
# sydney   -> 250
# paris    -> 180
# denver   -> 320

df['city_encode'] = df['city'].map(meann)

# Menampilkan dataframe setelah city di-encode
print(df)


# ============================================================
# 4. MENGAMBIL KOLOM PRICE DAN CITY_ENCODE
# ============================================================

# pc hanya mengambil kolom price dan city_encode
# Ini berguna untuk melihat hubungan antara target asli
# dan hasil encoding dari city.
pc = df[['price', 'city_encode']]

# Menampilkan hasil akhir
print(pc)


# ============================================================
# KESIMPULAN
# ============================================================

# Target Guided Ordinal Encoding mengubah kategori menjadi angka
# berdasarkan rata-rata nilai target.
#
# Teknik ini berguna ketika kategori punya hubungan dengan target.
#
# Contoh:
# city dengan rata-rata price tinggi akan mendapat nilai encoding tinggi.
# city dengan rata-rata price rendah akan mendapat nilai encoding rendah.
#
# Kelebihan:
# - Bisa menangkap pengaruh kategori terhadap target
# - Lebih ringkas daripada One Hot Encoding jika kategori sangat banyak
#
# Kekurangan:
# - Bisa menyebabkan data leakage jika dilakukan sebelum train-test split
# - Harus hati-hati jika kategori punya data sangat sedikit