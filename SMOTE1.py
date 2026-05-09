# ============================================================
# SMOTE: Synthetic Minority Over-sampling Technique
# ============================================================

# SMOTE adalah teknik untuk menangani imbalanced dataset.
# Imbalanced dataset terjadi ketika jumlah data antar class tidak seimbang.
#
# Contoh:
# target 0 = 90 data
# target 1 = 10 data
#
# Masalahnya:
# Model machine learning bisa terlalu bias ke class mayoritas.
#
# Konsep SMOTE:
# SMOTE tidak sekadar menduplikasi data minority.
# SMOTE membuat data sintetis baru dengan cara interpolasi,
# yaitu membuat titik data baru di antara titik-titik data minority yang sudah ada.

from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. MEMBUAT DATASET CLASSIFICATION YANG IMBALANCED
# ============================================================

# make_classification digunakan untuk membuat dataset buatan
# n_samples=100 artinya jumlah total data adalah 100
# n_features=2 artinya dataset punya 2 fitur, yaitu F1 dan F2
# n_redundant=0 artinya tidak ada fitur tambahan yang redundant
# n_clusters_per_class=1 artinya setiap class memiliki 1 cluster utama
# weights=[0.90] artinya class 0 sekitar 90%, class 1 sekitar 10%
# random_state=12 digunakan agar hasil random tetap sama saat dijalankan ulang

x, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.90],
    random_state=12
)

# Menampilkan hasil data fitur dan target
print(x)
print(y)


# ============================================================
# 2. MEMBUAT DATAFRAME DARI FITUR DAN TARGET
# ============================================================

# Membuat dataframe untuk fitur
# x berisi data numerik dengan 2 kolom fitur
df1 = pd.DataFrame(x, columns=['F1', 'F2'])

# Membuat dataframe untuk target/class
# y berisi label class 0 dan 1
df2 = pd.DataFrame(y, columns=['target'])

# Menggabungkan dataframe fitur dan target menjadi satu dataframe
# axis=1 artinya digabungkan secara horizontal berdasarkan kolom
final_df = pd.concat([df1, df2], axis=1)

# Menampilkan dataframe fitur
print(df1)

# Menampilkan dataframe target
print(df2)

# Menampilkan 5 data pertama dari dataframe final
print(final_df.head())


# ============================================================
# 3. MENGECEK JUMLAH DATA SETIAP CLASS
# ============================================================

# value_counts() digunakan untuk menghitung jumlah data pada setiap class
# Ini penting untuk melihat apakah dataset seimbang atau tidak
print(final_df['target'].value_counts())

# Dari hasil ini biasanya terlihat:
# class 0 jumlahnya jauh lebih banyak
# class 1 jumlahnya jauh lebih sedikit


# ============================================================
# 4. VISUALISASI DATA SEBELUM SMOTE
# ============================================================

# Scatter plot digunakan karena dataset hanya punya 2 fitur
# F1 sebagai sumbu X
# F2 sebagai sumbu Y
# c=final_df['target'] memberi warna berbeda berdasarkan class target

plt.scatter(final_df['F1'], final_df['F2'], c=final_df['target'])
plt.title("Data Sebelum SMOTE")
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()


# ============================================================
# 5. MEMISAHKAN FITUR DAN TARGET
# ============================================================

# X berisi kolom fitur yang akan digunakan model
X = final_df[['F1', 'F2']]

# y berisi kolom target/class
y = final_df['target']


# ============================================================
# 6. MENERAPKAN SMOTE
# ============================================================

# Membuat object SMOTE
# random_state=42 digunakan agar hasil SMOTE konsisten saat dijalankan ulang
oversample = SMOTE(random_state=42)

# fit_resample() digunakan untuk membuat data sintetis baru
# X_resampled adalah fitur setelah SMOTE
# y_resampled adalah target setelah SMOTE
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Mengecek bentuk data fitur setelah SMOTE
print(X_resampled.shape)

# Mengecek bentuk data target setelah SMOTE
print(y_resampled.shape)


# ============================================================
# 7. MEMBUAT DATAFRAME HASIL SMOTE
# ============================================================

# Mengubah hasil SMOTE menjadi dataframe agar lebih mudah dibaca
df_smote = pd.DataFrame(X_resampled, columns=['F1', 'F2'])

# Menambahkan kolom target ke dataframe hasil SMOTE
df_smote['target'] = y_resampled

# Menampilkan dataframe hasil SMOTE
print(df_smote)

# Mengecek jumlah data setiap class setelah SMOTE
# Setelah SMOTE, jumlah class 0 dan class 1 akan menjadi seimbang
print(df_smote['target'].value_counts())


# ============================================================
# 8. VISUALISASI DATA SETELAH SMOTE
# ============================================================

# Setelah SMOTE, class minority akan bertambah
# Titik tambahan tersebut adalah data sintetis hasil interpolasi

plt.scatter(df_smote['F1'], df_smote['F2'], c=df_smote['target'])
plt.title("Data Setelah SMOTE")
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()


# ============================================================
# 9. MENGHITUNG JUMLAH MASING-MASING CLASS SETELAH SMOTE
# ============================================================

# Menghitung jumlah target 0 setelah SMOTE
print("Jumlah class 0 setelah SMOTE:", len(y_resampled[y_resampled == 0]))

# Menghitung jumlah target 1 setelah SMOTE
print("Jumlah class 1 setelah SMOTE:", len(y_resampled[y_resampled == 1]))


# ============================================================
# KESIMPULAN
# ============================================================

# Sebelum SMOTE:
# Dataset tidak seimbang, class 0 jauh lebih banyak daripada class 1.
#
# Setelah SMOTE:
# Class minority dibuat bertambah menggunakan data sintetis.
#
# SMOTE cocok digunakan ketika:
# - Dataset klasifikasi tidak seimbang
# - Class minority terlalu sedikit
# - Kita ingin model lebih adil dalam mempelajari class kecil
#
# SMOTE berbeda dari upsampling biasa:
# - Upsampling biasa menduplikasi data lama
# - SMOTE membuat data baru hasil interpolasi antar data minority