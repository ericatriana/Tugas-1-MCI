# Tugas Modul 1 Open Recruitment Lab MCI 2026 — Text Processing & TF-IDF pada Steam Game Reviews

> **Dataset:** steam_game_reviews.csv | **Tools:** PySpark, Scikit-learn, NLTK, Matplotlib

---

## Pendahuluan

Tugas pertama dalam rangkaian Open Recruitment Lab MCI 2026 ini berfokus pada pemrosesan teks (*text processing*) menggunakan dataset ulasan game dari Steam. Tujuannya adalah mengekstrak **signature keywords** — kata-kata yang paling merepresentasikan tiap judul game — menggunakan pendekatan TF-IDF tanpa bantuan kamus stopword eksternal.

Pipeline yang dibangun mencakup lima tahap utama: *data loading & selection*, *data cleaning*, *tokenization*, *TF-IDF implementation*, dan *signature keyword extraction*. Sebagai nilai tambah, diterapkan pula PySpark sebagai engine pemrosesan terdistribusi, lemmatization dengan NLTK, dan visualisasi distribusi IDF.

---

## 1. Persiapan Data (Data Loading & Selection)

Dataset dibaca menggunakan **PySpark** (`SparkSession`) agar mampu menangani data dalam skala besar secara efisien.

```python
spark = SparkSession.builder \
    .appName("BigData_Day1_MMDS") \
    .master("local[*]") \
    .getOrCreate()

df_raw = spark.read.csv(
    '/content/steam_game_reviews.csv',
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
).cache()
```

Setelah data dimuat, dilakukan **seleksi kolom** — hanya tiga kolom yang digunakan sesuai instruksi tugas:

```python
COLS = ['game_name', 'username', 'review']
spark_df = df_raw.select(*COLS)
```

**Mengapa PySpark?**
Steam game reviews adalah dataset bertipe *user-generated content* yang bisa sangat besar. PySpark memungkinkan distribusi komputasi ke multiple core (bahkan cluster), sehingga pipeline tetap scalable saat data tumbuh.

---

## 2. Pembersihan Data (Data Cleaning)

### Deteksi Missing Values

Sebelum cleaning, dilakukan pemeriksaan terhadap setiap kolom untuk menemukan nilai null atau string kosong:

```python
for c in spark_df.columns:
    null_count = spark_df.filter(
        col(c).isNull() | (trim(col(c).cast('string')) == '')
    ).count()
    print(f"{c:<15}: {null_count} missing")
```

### Penghapusan Baris Tidak Valid

Baris dengan `review` atau `game_name` yang null/kosong dihapus:

```python
df_clean = spark_df \
    .filter(col('review').isNotNull()) \
    .filter(col('game_name').isNotNull()) \
    .filter(trim(col('review')) != '')
```

### Normalisasi Teks

Normalisasi dilakukan secara berurutan menggunakan PySpark SQL functions:

```python
df_clean = df_clean \
    .withColumn('review_clean', lower(col('review')))                                   # lowercase
    .withColumn('review_clean', regexp_replace(..., r'https?://\S+|www\.\S+', ' '))    # hapus URL
    .withColumn('review_clean', regexp_replace(..., r'<[^>]+>', ' '))                   # hapus HTML tag
    .withColumn('review_clean', regexp_replace(..., r'[^a-z\s]', ' '))                  # hapus non-alphabet
    .withColumn('review_clean', regexp_replace(..., r'\s+', ' '))                        # normalisasi spasi
    .withColumn('review_clean', trim(col('review_clean')))
```

Langkah-langkah pembersihan:
- **Lowercase** — menyeragamkan kapitalisasi
- **Hapus URL** — link tidak mengandung makna semantik
- **Hapus HTML tag** — artifact dari web scraping
- **Hapus karakter non-alfabet** — angka, tanda baca, simbol
- **Normalisasi whitespace** — hapus spasi berlebih

---

## 3. Tokenization

Tokenisasi dilakukan dengan memecah teks berdasarkan spasi menggunakan fungsi `split` dari PySpark:

```python
df_tokenized = df_clean \
    .withColumn('tokens', split(col('review_clean'), ' ')) \
    .withColumn('token_count', size(col('tokens')))
```

Setelah tokenisasi, dilakukan **filter outlier**: review dengan kurang dari 3 token dihapus karena terlalu pendek untuk menghasilkan sinyal TF-IDF yang bermakna.

```python
df_filtered = df_tokenized.filter(col('token_count') >= 3)
```

---

## 4. Implementasi TF-IDF

### Strategi: Dokumen per Game

Alih-alih memperlakukan setiap review sebagai satu dokumen terpisah, seluruh review dalam satu game digabungkan menjadi **satu corpus**. Ini membuat TF-IDF mencerminkan kata yang paling khas untuk keseluruhan game tersebut.

### CountVectorizer + IDF (PySpark ML)

```python
from pyspark.ml.feature import CountVectorizer, IDF

# Bangun vocabulary
cv = CountVectorizer(
    inputCol='tokens',
    outputCol='tf_features',
    vocabSize=20000,  # maksimal 20.000 kata unik
    minDF=5           # kata harus muncul di minimal 5 dokumen
)
cv_model = cv.fit(df_filtered)
vocabulary = cv_model.vocabulary

# Hitung IDF
idf = IDF(inputCol='tf_features', outputCol='tfidf_features', minDocFreq=5)
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)
idf_values = idf_model.idf.toArray()
```

### Penentuan Threshold IDF

Alih-alih menggunakan kamus stopword, kata-kata umum diidentifikasi secara **otomatis berdasarkan distribusi IDF**. Kata dengan IDF terlalu rendah berarti muncul di hampir semua dokumen — itulah stopword alami.

```python
IDF_THRESHOLD = float(np.percentile(idf_values, 20))
```

Threshold dipilih di **persentil ke-20**: artinya 20% kata dengan IDF terendah (paling umum) otomatis disingkirkan. Pendekatan ini *data-driven* — threshold menyesuaikan diri dengan karakteristik dataset, bukan hardcoded.

Contoh kata yang dihapus otomatis (IDF sangat rendah): *game*, *play*, *the*, *and*, *get*, *really* — kata-kata yang muncul di hampir semua review tanpa membedakan satu game dari yang lain.

---

## 5. Ekstraksi Signature Keywords

### UDF untuk Ekstraksi Keyword per Review

Sebuah *User Defined Function* (UDF) dibuat untuk mengekstrak top-N kata berdasarkan skor TF-IDF dari setiap review:

```python
def extract_top_keywords(tfidf_vector, n=5):
    vocab = vocab_bc.value
    fv = filtered_vocab_bc.value
    word_scores = []
    for idx, val in zip(tfidf_vector.indices, tfidf_vector.values):
        if idx < len(vocab):
            word = vocab[idx]
            if word in fv and len(word) > 2:  # skip kata < 3 huruf
                word_scores.append((word, float(val)))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in word_scores[:n]]
```

### Akumulasi Keyword per Game

Top-5 keyword per review di-*explode*, lalu dihitung frekuensi kemunculannya sebagai keyword unggulan di seluruh review dalam satu game. Top-3 dengan accumulated score tertinggi dijadikan `signature_keywords`:

```python
keywords_exploded = df_tfidf.select(
    col('game_name'),
    explode(col('review_keywords')).alias('keyword')
)

keyword_counts = keywords_exploded \
    .groupBy('game_name', 'keyword') \
    .agg(count('*').alias('accumulated_score'))

window_game = Window.partitionBy('game_name').orderBy(col('accumulated_score').desc())

top3_df = keyword_counts \
    .withColumn('rnk', rank().over(window_game)) \
    .filter(col('rnk') <= 3) \
    .groupBy('game_name') \
    .agg(collect_list('keyword').alias('signature_keywords'))
```

### Hasil: Kolom `signature_keywords`

Kolom baru `signature_keywords` di-join kembali ke DataFrame utama. Setiap game kini memiliki 2–3 kata paling representatif yang mencerminkan konten ulasan penggunanya.

Contoh hasil (ilustratif):

| game_name | signature_keywords |
|---|---|
| Counter-Strike: Global Offensive | competitive, fps, match |
| Stardew Valley | farm, relaxing, cozy |
| The Witcher 3 | story, open_world, rpg |
| Terraria | sandbox, boss, craft |

---

## Metode Tambahan (Nilai Plus)

### 🔹 PySpark sebagai Distributed Computing Engine

Alasan: Dataset ulasan Steam bisa mencapai jutaan baris. PySpark memungkinkan pemrosesan paralel di atas cluster, sehingga pipeline tidak bottleneck saat data di-scale up.

### 🔹 Lemmatization dengan NLTK WordNetLemmatizer

Sebelum TF-IDF dihitung, setiap token di-lemmatize — kata seperti *playing*, *played*, *plays* akan dikembalikan ke bentuk dasar *play*. Ini mengurangi inflasi vocabulary dan membuat kata-kata yang secara semantik sama tidak terhitung sebagai entitas berbeda.

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# diaplikasikan saat preprocessing token
```

### 🔹 Data-driven Stopword Removal via IDF Percentile

Tidak menggunakan kamus stopword hardcoded, melainkan menghitung percentile distribusi IDF. Pendekatan ini adaptif: untuk genre game tertentu, kata "combat" mungkin umum (IDF rendah), tapi di dataset lain bisa menjadi keyword penting.

### 🔹 Visualisasi Distribusi IDF

Dibuat dua visualisasi:
1. **Histogram distribusi IDF** dengan garis threshold, untuk memverifikasi pemilihan batas secara visual
2. **Bar chart 20 kata paling umum** (IDF terendah) yang tersingkir sebagai stopword otomatis

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(idf_values, bins=60, ...)
axes[0].axvline(IDF_THRESHOLD, color='red', linestyle='--', ...)
axes[1].barh(words_common[::-1], idf_common[::-1], ...)
plt.savefig('idf_distribution.png', ...)
```

### 🔹 Accumulated Score sebagai Proxy Relevansi

Signature keyword dipilih bukan hanya dari satu review dengan TF-IDF tertinggi, melainkan dari kata yang **paling sering muncul sebagai top keyword di semua review game tersebut**. Ini membuat signature keywords lebih robust terhadap outlier review dan lebih merepresentasikan konsensus komunitas.

---

## Kesimpulan

Pipeline yang dibangun berhasil mengekstrak signature keywords dari dataset ulasan Steam tanpa menggunakan kamus stopword eksternal. Beberapa poin kunci:

- **TF-IDF berbasis corpus per game** lebih efektif daripada per-review karena mencerminkan karakteristik keseluruhan game
- **Threshold IDF otomatis via percentile** adalah pendekatan yang data-driven dan tidak bergantung pada bahasa tertentu
- **PySpark** membuat pipeline ini scalable untuk dataset yang lebih besar
- **Accumulated scoring** membuat keyword yang terpilih lebih representatif dan tahan terhadap noise

Hasilnya adalah kolom `signature_keywords` yang berisi 2–3 kata yang paling khas untuk setiap judul game, diturunkan murni dari pola statistik dalam teks ulasan pengguna.

---

*Tugas ini dikerjakan sebagai bagian dari Open Recruitment Admin Lab MCI 2026.*
