# ============================================================
# EDA: EXPLORATORY DATA ANALYSIS
# Dataset: Wine Quality Red
# ============================================================

# Import library yang dibutuhkan
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Membaca file CSV
# Pastikan file winequality-red.csv berada di folder yang sama
# dengan file Python kamu, atau tulis path lengkapnya.
df = pd.read_csv('winequality-red.csv')

# Menampilkan seluruh dataset
print(df)


# ============================================================
# 2. SUMMARY DATASET
# ============================================================

# info() digunakan untuk melihat informasi struktur dataset:
# - jumlah baris
# - jumlah kolom
# - nama kolom
# - tipe data setiap kolom
# - jumlah data non-null
print(df.info())

# describe() digunakan untuk melihat statistik deskriptif:
# - count
# - mean
# - std
# - min
# - Q1 / 25%
# - median / 50%
# - Q3 / 75%
# - max
print(df.describe())


# ============================================================
# 3. CEK KOLOM DAN TARGET
# ============================================================

# Menampilkan semua nama kolom pada dataset
print(df.columns)

# Menampilkan isi kolom quality
# Pada dataset ini, quality biasanya dianggap sebagai target
print(df['quality'])

# Melihat nilai unik pada kolom quality
# Contoh: 3, 4, 5, 6, 7, 8
print(df['quality'].unique())

# Menghitung jumlah data untuk setiap nilai quality
# Ini berguna untuk melihat apakah target seimbang atau tidak
print(df['quality'].value_counts())


# ============================================================
# 4. CEK MISSING VALUE
# ============================================================

# Mengecek jumlah missing value pada setiap kolom
# Jika hasilnya 0 semua, berarti tidak ada data kosong
print(df.isnull().sum())


# ============================================================
# 5. CEK DATA DUPLIKAT
# ============================================================

# duplicated() menghasilkan True jika baris tersebut duplikat
print(df.duplicated())

# Mengambil semua baris yang duplikat
x = df[df.duplicated()]
print(x)

# Menghitung jumlah data duplikat
print("Jumlah data duplikat:", df.duplicated().sum())


# ============================================================
# 6. MENGHAPUS DATA DUPLIKAT
# ============================================================

# drop_duplicates() digunakan untuk menghapus data duplikat
# inplace=True artinya dataframe df langsung diubah
df.drop_duplicates(inplace=True)

# Jangan gunakan print(df.drop_duplicates(inplace=True))
# karena jika inplace=True, output-nya adalah None

# Mengecek ukuran dataset setelah data duplikat dihapus
print("Shape setelah drop duplicates:", df.shape)


# ============================================================
# 7. KORELASI ANTAR FITUR
# ============================================================

# corr() digunakan untuk menghitung korelasi antar kolom numerik
# Nilai korelasi berada di antara -1 sampai 1
#
#  1  = korelasi positif sangat kuat
#  0  = hampir tidak ada hubungan linear
# -1  = korelasi negatif sangat kuat
correlation = df.corr()

# Menampilkan tabel korelasi
print(correlation)


# ============================================================
# 8. HEATMAP KORELASI
# ============================================================

# Heatmap digunakan untuk memvisualisasikan korelasi antar fitur
# annot=True menampilkan angka korelasi
# fmt=".2f" membulatkan angka menjadi 2 angka di belakang koma
# figsize digunakan agar grafik lebih besar dan mudah dibaca
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Heatmap - Wine Quality")
plt.show()


# ============================================================
# 9. CEK DISTRIBUSI TARGET QUALITY
# ============================================================

# value_counts().plot(kind='bar') digunakan untuk melihat jumlah data
# pada setiap class quality
#
# Ini berguna untuk mengecek apakah dataset imbalanced atau tidak.
# Jika salah satu quality jauh lebih banyak, berarti target tidak seimbang.
df['quality'].value_counts().plot(kind='bar')

plt.title("Distribution of Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# Menampilkan 5 data pertama
print(df.head())


# ============================================================
# 10. HISTOGRAM SETIAP KOLOM
# ============================================================

# Histogram digunakan untuk melihat distribusi data setiap fitur.
# kde=True menambahkan garis distribusi halus.
#
# Dari histogram, kita bisa melihat:
# - apakah data normal
# - apakah data skewed/miring
# - apakah ada nilai ekstrem
# - bagaimana persebaran setiap fitur

for column in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# ============================================================
# 11. BOXPLOT QUALITY VS ALCOHOL
# ============================================================

# Boxplot digunakan untuk melihat hubungan antara quality dan alcohol.
#
# x='quality' berarti sumbu X adalah kualitas wine.
# y='alcohol' berarti sumbu Y adalah kadar alkohol.
#
# Dari boxplot ini, kita bisa melihat apakah wine dengan quality tinggi
# cenderung memiliki alcohol lebih tinggi atau tidak.

sns.catplot(
    x='quality',
    y='alcohol',
    data=df,
    kind='box',
    height=5,
    aspect=1.5
)

plt.title("Alcohol Distribution by Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Alcohol")
plt.show()


# ============================================================
# 12. BOXPLOT SEMUA FITUR TERHADAP QUALITY
# ============================================================

# Jika ingin melihat semua fitur terhadap quality,
# gunakan loop seperti ini.
#
# Setiap fitur akan dibandingkan dengan quality menggunakan boxplot.
# Ini lebih masuk akal daripada membuat boxplot alcohol berulang-ulang.

for column in df.columns:
    if column != 'quality':
        sns.catplot(
            x='quality',
            y=column,
            data=df,
            kind='box',
            height=5,
            aspect=1.5
        )

        plt.title(f"{column} Distribution by Quality")
        plt.xlabel("Quality")
        plt.ylabel(column)
        plt.show()


# ============================================================
# 13. SCATTERPLOT ALCOHOL VS PH
# ============================================================

# Scatterplot digunakan untuk melihat hubungan antara dua fitur numerik.
#
# x='alcohol' berarti sumbu X adalah alcohol.
# y='pH' berarti sumbu Y adalah pH.
# hue='quality' memberi warna berbeda berdasarkan quality.
#
# Grafik ini membantu melihat apakah kombinasi alcohol dan pH
# memiliki pola tertentu terhadap quality.

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x='alcohol',
    y='pH',
    hue='quality'
)

plt.title("Scatterplot Alcohol vs pH by Quality")
plt.xlabel("Alcohol")
plt.ylabel("pH")
plt.show()


# ============================================================
# 14. SCATTERPLOT SEMUA FITUR TERHADAP QUALITY
# ============================================================

# Jika ingin melihat hubungan setiap fitur numerik terhadap quality,
# gunakan loop seperti ini.
#
# Setiap fitur menjadi sumbu X,
# quality menjadi sumbu Y.

for column in df.columns:
    if column != 'quality':
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x=column,
            y='quality'
        )

        plt.title(f"{column} vs Quality")
        plt.xlabel(column)
        plt.ylabel("Quality")
        plt.show()


# ============================================================
# KESIMPULAN EDA
# ============================================================

# Dari EDA ini, kita bisa menganalisis:
#
# 1. Struktur dataset
#    Dilihat dari df.info() dan df.describe()
#
# 2. Missing value
#    Dilihat dari df.isnull().sum()
#
# 3. Data duplikat
#    Dilihat dari df.duplicated().sum()
#
# 4. Korelasi antar fitur
#    Dilihat dari df.corr() dan heatmap
#
# 5. Distribusi target quality
#    Dilihat dari bar chart value_counts()
#
# 6. Distribusi setiap fitur
#    Dilihat dari histogram
#
# 7. Hubungan fitur dengan quality
#    Dilihat dari boxplot dan scatterplot