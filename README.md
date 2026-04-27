# Text Processing & TF-IDF pada Steam Game Reviews: Mengekstrak Signature Keywords dengan Apache Spark

**Laporan Tugas Modul 1 — Open Recruitment Lab MCI 2026**

---

> *"Setiap game punya ceritanya sendiri — dan ulasan para pemain menyimpan jejak kata-kata yang paling merepresentasikannya. Tugas kita adalah menemukannya."*

---

## Pendahuluan

Bayangkan kamu memiliki jutaan ulasan game dari Steam. Bukan satu-dua, tapi ribuan review dari ratusan judul berbeda. Bagaimana cara menemukan kata-kata yang paling "mewakili" sebuah game tanpa membacanya satu per satu?

Inilah tantangan yang diselesaikan dalam tugas ini: membangun pipeline **Text Processing + TF-IDF** menggunakan **Apache Spark** untuk mengekstrak *signature keywords* — kata-kata yang paling khas dan representatif untuk setiap game di dataset Steam.

### Dataset

Dataset yang digunakan adalah **`steam_game_reviews.csv`**, berisi kolom-kolom seperti `game_name`, `username`, `review`, dan berbagai metadata lainnya. Dari keseluruhan kolom tersebut, hanya tiga yang relevan untuk analisis ini:

| Kolom | Deskripsi |
|---|---|
| `game_name` | Nama judul game |
| `username` | Nama pengguna yang memberikan review |
| `review` | Teks ulasan dari pemain |

### Tech Stack

| Teknologi | Kegunaan |
|---|---|
| **Apache Spark (PySpark)** | Distributed computing untuk memproses data besar |
| **Spark ML (CountVectorizer + IDF)** | Implementasi TF-IDF skala besar |
| **NLTK (WordNetLemmatizer)** | Lemmatization sebagai metode tambahan |
| **Matplotlib & Seaborn** | Visualisasi distribusi IDF |
| **Pandas** | Inspeksi dan presentasi hasil akhir |

---

## Mengapa Apache Spark?

Sebelum masuk ke pipeline, ada pertanyaan mendasar: *kenapa tidak pakai Pandas saja?*

Dataset Steam game reviews bisa sangat besar — dengan ratusan ribu hingga jutaan baris review. Pandas memuat semua data ke RAM (in-memory), yang akan menjadi bottleneck pada dataset besar. Apache Spark menggunakan paradigma **distribusi komputasi**, artinya pemrosesan dibagi ke beberapa worker (atau core CPU) secara paralel, sehingga jauh lebih skalabel.

Selain itu, Spark ML menyediakan implementasi **CountVectorizer** dan **IDF** yang natively bekerja pada DataFrame terdistribusi — sangat cocok untuk kasus ini.

```python
spark = SparkSession.builder \
    .appName("BigData_Day1_MMDS") \
    .master("local[*]") \
    .getOrCreate()
```

Dengan `local[*]`, Spark akan menggunakan semua core CPU yang tersedia pada mesin lokal — ini sudah jauh lebih cepat dari single-threaded Pandas untuk data besar.

---

## Step 1: Persiapan Data (Data Loading & Selection)

### Loading Dataset

Dataset CSV dimuat menggunakan Spark dengan opsi `multiLine=True` dan `escape='"'` untuk menangani review yang mengandung karakter khusus atau newline di dalam satu cell:

```python
df_raw = spark.read.csv(
    '/content/steam_game_reviews.csv',
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
).cache()
```

Opsi `.cache()` penting di sini — ini memberitahu Spark untuk menyimpan DataFrame di memori setelah pertama kali dihitung, sehingga operasi berikutnya tidak perlu membaca ulang dari disk.

### Seleksi Kolom

Dari seluruh kolom yang tersedia, hanya tiga kolom yang dipertahankan:

```python
COLS = ['game_name', 'username', 'review']
spark_df = df_raw.select(*COLS)
```

Melakukan seleksi kolom di awal bukan hanya soal kerapian kode — ini juga menghemat memori dan mempercepat operasi downstream karena Spark tidak perlu membawa kolom yang tidak diperlukan melalui seluruh pipeline.

---

## Step 2: Pembersihan Data (Data Cleaning)

### Deteksi Missing Values

Sebelum melakukan pembersihan, dilakukan audit missing values untuk setiap kolom:

```python
for c in spark_df.columns:
    null_count = spark_df.filter(
        col(c).isNull() | (trim(col(c).cast('string')) == '')
    ).count()
    print(f"  {c:<15}: {null_count} missing")
```

Pemeriksaan ini menggabungkan dua kondisi: nilai `null` (benar-benar kosong di level SQL) dan string kosong setelah di-trim (nilai `''` atau `'   '`). Keduanya perlu ditangani agar tidak ada review "palsu" yang masuk ke pipeline.

### Menghapus Data Kotor

Baris yang memiliki `review` atau `game_name` yang null/kosong dihapus:

```python
df_clean = spark_df \
    .filter(col('review').isNotNull()) \
    .filter(col('game_name').isNotNull()) \
    .filter(trim(col('review')) != '')
```

### Normalisasi Teks

Inilah inti dari tahap cleaning: mengubah teks mentah menjadi format yang konsisten dan bersih. Dilakukan serangkaian transformasi secara berurutan (method chaining):

```python
df_clean = df_clean \
    .withColumn('review_clean', lower(col('review'))) \
    .withColumn('review_clean', regexp_replace(col('review_clean'), r'https?://\S+|www\.\S+', ' ')) \
    .withColumn('review_clean', regexp_replace(col('review_clean'), r'<[^>]+>', ' ')) \
    .withColumn('review_clean', regexp_replace(col('review_clean'), r'[^a-z\s]', ' ')) \
    .withColumn('review_clean', regexp_replace(col('review_clean'), r'\s+', ' ')) \
    .withColumn('review_clean', trim(col('review_clean')))
```

Setiap langkah memiliki fungsi spesifik:

| Langkah | Regex | Tujuan |
|---|---|---|
| Lowercase | — | Menyamakan kapital (`Game` = `game`) |
| URL removal | `https?://\S+\|www\.\S+` | Hapus tautan web |
| HTML tag removal | `<[^>]+>` | Hapus tag HTML seperti `<br>`, `<b>` |
| Non-alpha removal | `[^a-z\s]` | Hapus angka, tanda baca, emoji |
| Whitespace normalization | `\s+` | Satukan multiple spasi menjadi satu |

**Contoh transformasi:**

```
BEFORE: "This game is AMAZING!! 10/10 would recommend. Check https://store.steampowered.com"
AFTER : "this game is amazing would recommend check"
```

---

## Step 3: Tokenization

Tokenisasi adalah proses memecah teks menjadi unit-unit kata (token). Di sini digunakan fungsi `split` dari PySpark yang memotong string berdasarkan spasi:

```python
df_tokenized = df_clean \
    .withColumn('tokens', split(col('review_clean'), ' ')) \
    .withColumn('token_count', size(col('tokens')))
```

### Filter Review Pendek

Setelah tokenisasi, review dengan kurang dari 3 token dibuang. Review yang terlalu pendek (misalnya hanya "good" atau "ok") tidak memberikan sinyal kata yang bermakna untuk analisis TF-IDF:

```python
df_filtered = df_tokenized.filter(col('token_count') >= 3)
```

---

## Step 4: Implementasi TF-IDF

Ini adalah inti dari seluruh pipeline. Sebelum membahas implementasinya, penting untuk memahami konsep dasarnya.

### Apa itu TF-IDF?

**TF-IDF (Term Frequency–Inverse Document Frequency)** adalah metrik yang mengukur seberapa penting sebuah kata dalam sebuah dokumen relatif terhadap kumpulan dokumen lainnya (corpus).

- **TF (Term Frequency)**: Seberapa sering sebuah kata muncul dalam satu dokumen.
- **IDF (Inverse Document Frequency)**: Kebalikan dari seberapa sering kata tersebut muncul di *semua* dokumen. Kata yang ada di semua dokumen (seperti "the", "is", "and") mendapat IDF rendah, sehingga bobotnya kecil meskipun sering muncul.

Rumus IDF:

```
IDF(t) = log((N + 1) / (df(t) + 1))
```

Di mana `N` adalah jumlah total dokumen dan `df(t)` adalah jumlah dokumen yang mengandung kata `t`. Nilai `+1` pada penyebut digunakan untuk menghindari pembagian nol (smoothing).

**TF-IDF = TF × IDF**

Kata yang sering muncul dalam satu dokumen spesifik tapi jarang di dokumen lain akan memiliki skor TF-IDF tinggi — itulah yang kita cari sebagai *signature keyword*.

### 4a: CountVectorizer

`CountVectorizer` dari Spark ML membangun vocabulary dari semua token dan menghitung frekuensi setiap kata per dokumen (Term Frequency):

```python
cv = CountVectorizer(
    inputCol='tokens',
    outputCol='tf_features',
    vocabSize=20000,   # maksimal 20.000 kata unik
    minDF=5            # kata harus muncul di minimal 5 dokumen
)

cv_model = cv.fit(df_filtered)
vocabulary = cv_model.vocabulary
df_tf = cv_model.transform(df_filtered)
```

Parameter `minDF=5` berarti kata yang hanya muncul di kurang dari 5 dokumen tidak masuk vocabulary. Ini adalah cara sederhana untuk membuang kata-kata sangat langka yang mungkin hanya typo atau nama unik.

**Mengapa CountVectorizer, bukan HashingTF?**

CountVectorizer menyimpan vocabulary eksplisit (berupa list kata), sehingga kita bisa tahu persis kata mana yang dipilih. HashingTF lebih cepat tapi tidak menyimpan mapping kata → indeks, sehingga hasilnya tidak interpretable secara langsung.

### 4b: IDF Computation

Setelah mendapat TF, langkah berikutnya menghitung IDF:

```python
idf = IDF(
    inputCol='tf_features',
    outputCol='tfidf_features',
    minDocFreq=5
)

idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)
idf_values = idf_model.idf.toArray()
```

`idf_model.idf` menghasilkan vektor IDF untuk setiap kata dalam vocabulary. Nilai ini kemudian digunakan untuk menentukan threshold.

### 4c: IDF Threshold — Stopword Removal Otomatis

Inilah keunikan pendekatan ini: **tidak menggunakan kamus stopword eksternal**. Sebaliknya, kata-kata umum diidentifikasi secara otomatis menggunakan distribusi IDF itu sendiri.

Threshold ditentukan menggunakan **persentil ke-20** dari seluruh nilai IDF:

```python
IDF_THRESHOLD = float(np.percentile(idf_values, 20))
```

Kata-kata dengan IDF di bawah threshold ini dianggap "terlalu umum" dan dibuang. Secara statistik, ini berarti 20% kata dengan IDF terendah (paling sering muncul di banyak dokumen) dikeluarkan dari analisis.

**Mengapa percentile ke-20?**

Ini adalah pilihan yang cukup konservatif — hanya membuang kata-kata yang benar-benar paling umum. Nilai yang lebih tinggi (misalnya 30 atau 40) akan membuang lebih banyak kata, berpotensi kehilangan kata bermakna. Nilai yang terlalu rendah (5-10) mungkin tidak cukup memfilter stopword alami.

### 4d: Visualisasi Distribusi IDF

Visualisasi membantu kita memvalidasi bahwa threshold yang dipilih masuk akal:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram distribusi IDF
axes[0].hist(idf_values, bins=60, color='steelblue', edgecolor='white')
axes[0].axvline(IDF_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {IDF_THRESHOLD:.2f}')
axes[0].set_title('Distribusi IDF Values')

# Bar chart 20 kata paling umum
axes[1].barh(words_common[::-1], idf_common[::-1], color='salmon')
axes[1].set_title('20 Kata Paling Umum (IDF Terendah)')
```

Dari histogram, kita bisa melihat distribusi IDF: kata-kata dengan IDF mendekati nol adalah kata-kata sangat umum seperti "game", "play", "the". Garis merah vertikal menunjukkan di mana threshold berada — semua kata di sebelah kiri threshold dibuang.

---

## Step 5: Ekstraksi Signature Keywords

### 5a: UDF untuk Ekstraksi Keywords per Review

Dibuat sebuah **User Defined Function (UDF)** yang mengekstrak top-5 kata dengan skor TF-IDF tertinggi dari setiap review, dengan syarat:
1. Kata ada dalam `filtered_vocab` (IDF ≥ threshold)
2. Panjang kata lebih dari 2 karakter (skip kata sangat pendek seperti "ok", "no")

```python
vocab_bc = spark.sparkContext.broadcast(vocabulary)
filtered_vocab_bc = spark.sparkContext.broadcast(filtered_vocab)

def extract_top_keywords(tfidf_vector, n=5):
    if tfidf_vector is None:
        return []
    vocab = vocab_bc.value
    fv = filtered_vocab_bc.value
    word_scores = []
    for idx, val in zip(tfidf_vector.indices, tfidf_vector.values):
        if idx < len(vocab):
            word = vocab[idx]
            if word in fv and len(word) > 2:
                word_scores.append((word, float(val)))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in word_scores[:n]]
```

**Catatan tentang Broadcast Variables:**

`spark.sparkContext.broadcast()` digunakan untuk mendistribusikan vocabulary dan filtered_vocab ke semua worker Spark secara efisien. Tanpa broadcast, Spark akan mengirimkan data ini berulang kali ke setiap task — sangat tidak efisien untuk vocabulary berukuran ribuan kata.

### 5b: Akumulasi Signature Keywords per Game

Setelah mendapat top keywords per review, dilakukan akumulasi untuk mendapatkan kata yang paling "representatif" untuk seluruh game, bukan hanya satu review:

```python
# Explode: satu baris per (game_name, keyword)
keywords_exploded = df_tfidf.select(
    col('game_name'),
    explode(col('review_keywords')).alias('keyword')
)

# Hitung berapa kali kata muncul sebagai top keyword di review game tersebut
keyword_counts = keywords_exploded \
    .groupBy('game_name', 'keyword') \
    .agg(count('*').alias('accumulated_score'))

# Ambil top-3 per game
window_game = Window.partitionBy('game_name').orderBy(col('accumulated_score').desc())

top3_df = keyword_counts \
    .withColumn('rnk', rank().over(window_game)) \
    .filter(col('rnk') <= 3) \
    .groupBy('game_name') \
    .agg(collect_list('keyword').alias('signature_keywords'))
```

**Logika di balik akumulasi:**

Daripada mengambil rata-rata skor TF-IDF (yang bisa menyesatkan karena skala berbeda per review), pendekatan ini menghitung **berapa kali** sebuah kata muncul sebagai top keyword di semua review game tersebut. Kata yang konsisten muncul sebagai top keyword di banyak review kemungkinan besar memang menjadi kata khas game itu.

### 5c: Join ke DataFrame Utama

Hasil signature keywords digabungkan kembali ke DataFrame utama:

```python
df_final = df_tfidf.join(top3_df, on='game_name', how='left')
```

Dengan `left join`, semua baris dari `df_tfidf` dipertahankan — game yang tidak memiliki keywords (misalnya karena terlalu sedikit data) tetap ada dengan nilai null di kolom `signature_keywords`.

### 5d: Tabel Hasil Akhir

```python
summary_pd = top3_df.orderBy('game_name').toPandas()
summary_pd['signature_keywords'] = summary_pd['signature_keywords'].apply(
    lambda x: ', '.join(x) if isinstance(x, list) else x
)
```

Hasil akhir berupa tabel yang menunjukkan 2-3 kata paling khas untuk setiap game dalam dataset.

---

## Metode Tambahan: Teknik-Teknik yang Memperkuat Pipeline

Selain langkah-langkah wajib, terdapat beberapa teknik tambahan yang diterapkan untuk meningkatkan kualitas hasil.

### 1. Broadcast Variables untuk Efisiensi Distribusi

Seperti dijelaskan sebelumnya, vocabulary dan filtered_vocab di-broadcast ke semua worker Spark. Ini mencegah Spark mengirim data berulang kali dan signifikan mempercepat eksekusi UDF.

```python
vocab_bc = spark.sparkContext.broadcast(vocabulary)
filtered_vocab_bc = spark.sparkContext.broadcast(filtered_vocab)
```

**Mengapa ini penting:** Tanpa broadcast, setiap task Spark (bisa ratusan) akan menerima salinan vocabulary tersendiri melalui jaringan. Dengan broadcast, data dikirim sekali ke setiap executor, lalu di-cache lokal.

### 2. DataFrame Caching (`.cache()`)

Beberapa DataFrame di-cache untuk menghindari re-computation:

```python
df_raw = spark.read.csv(...).cache()
```

Spark secara default mengevaluasi transformasi secara *lazy* — artinya tidak dieksekusi sampai ada action (seperti `count()` atau `show()`). Tanpa `.cache()`, jika DataFrame diakses dua kali, Spark akan membaca ulang file dari disk dua kali. Caching menyimpan hasilnya di memori.

### 3. Window Function untuk Ranking

Penggunaan `Window.partitionBy('game_name').orderBy(...)` memungkinkan ranking per game dilakukan dalam satu pass, tanpa perlu loop atau groupby yang terpisah:

```python
window_game = Window.partitionBy('game_name').orderBy(col('accumulated_score').desc())
top3_df = keyword_counts \
    .withColumn('rnk', rank().over(window_game)) \
    .filter(col('rnk') <= 3)
```

Ini adalah pola SQL-style yang sangat efisien di Spark dan jauh lebih ekspresif dibanding alternatif imperatif.

### 4. Integrasi NLTK WordNetLemmatizer (Nilai Plus)

Library NLTK diimpor untuk lemmatization — proses mengubah kata ke bentuk dasarnya (misalnya "playing" → "play", "games" → "game"). Ini dapat meningkatkan kualitas TF-IDF karena variasi kata yang berbeda (but bermakna sama) akan dikelompokkan menjadi satu token.

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
```

Lemmatization berbeda dari stemming: stemming memotong akhiran secara mekanis (bisa menghasilkan kata tidak valid seperti "happi"), sedangkan lemmatization menggunakan kamus linguistik untuk menghasilkan bentuk kata yang valid.

**Kapan ini berguna?** Misalnya, dalam review gaming, kata "bought", "buy", "buying" semuanya merujuk ke konsep yang sama. Dengan lemmatization, ketiganya jadi "buy" — memperkuat sinyal TF-IDF untuk kata tersebut.

### 5. Multi-layer Text Cleaning

Pipeline cleaning tidak hanya lowercase, tapi juga mencakup:
- **URL removal**: Review gamer sering mencantumkan link (YouTube, Steam Store, wiki)
- **HTML tag removal**: Dataset Steam kadang mengandung HTML dari sistem review
- **Character normalization**: Angka, emoji, dan tanda baca dihapus untuk fokus pada kata bermakna

---

## Alur Pipeline Lengkap

```
Raw CSV
   │
   ▼
[1] Load & Select Columns
   │  (game_name, username, review)
   ▼
[2] Data Cleaning
   │  - Remove nulls/empty
   │  - Lowercase, remove URL, HTML, non-alpha
   │  - Normalize whitespace
   ▼
[3] Tokenization
   │  - split(' ') → array of tokens
   │  - Filter: token_count >= 3
   ▼
[4] TF-IDF Pipeline (Spark ML)
   │  ┌─ CountVectorizer (vocab: 20K, minDF: 5)
   │  └─ IDF (minDocFreq: 5)
   │  - Determine IDF threshold (percentile 20)
   │  - Auto-identify stopwords via IDF
   ▼
[5] Signature Keyword Extraction
   │  - UDF: top-5 keywords per review
   │  - Explode + GroupBy: accumulated score per game
   │  - Window rank: top-3 per game
   ▼
df_final dengan kolom signature_keywords
```

---

## Refleksi dan Insight

### Apa yang Menarik dari Pendekatan Ini?

**TF-IDF tanpa kamus stopword** adalah pendekatan yang elegan. Alih-alih bergantung pada daftar kata buatan manusia yang bersifat statis dan bahasa-spesifik, IDF secara alami mengidentifikasi kata-kata umum dari data itu sendiri. Kata seperti "game", "play", "fun" yang mungkin tidak ada di stopword list standar akan mendapat IDF rendah karena muncul di hampir semua review — dan otomatis ter-downweight.

### Keterbatasan

Beberapa hal yang bisa ditingkatkan ke depannya:

1. **Context-blind**: TF-IDF tidak memahami semantik. Kata "dark" dan "darkness" dianggap berbeda, dan relasi antar kata tidak ditangkap.
2. **Review length bias**: Review panjang cenderung mengandung lebih banyak kata, yang bisa mendominasi akumulasi keyword jika tidak dinormalisasi.
3. **Multibahasa**: Dataset Steam mengandung review dalam berbagai bahasa. Pipeline ini berasumsi review berbahasa Inggris; review dalam bahasa lain akan menghasilkan token yang tidak bermakna.

### Pengembangan Lebih Lanjut

- **Word2Vec / BERT Embeddings**: Untuk menangkap semantik dan sinonim
- **BM25**: Alternatif TF-IDF yang memiliki normalisasi panjang dokumen lebih baik
- **LDA (Latent Dirichlet Allocation)**: Untuk topic modeling yang lebih kaya

---

## Kesimpulan

Pipeline yang dibangun dalam tugas ini berhasil mengekstrak *signature keywords* dari dataset Steam game reviews menggunakan pendekatan berbasis **TF-IDF murni** (tanpa kamus stopword eksternal). Dengan memanfaatkan kekuatan **Apache Spark** untuk distributed computing, pipeline ini skalabel dan dapat dijalankan pada dataset jauh lebih besar.

Kata-kata dengan akumulasi skor TF-IDF tertinggi per game berhasil diidentifikasi sebagai representasi paling khas dari setiap judul — mencerminkan apa yang paling sering dibicarakan pemain tentang game tersebut.

---

## Referensi

- [Apache Spark Documentation — MLlib Feature Extraction](https://spark.apache.org/docs/latest/ml-features.html)
- [Scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK WordNet Lemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html)
- Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management.
- [Steam Game Reviews Dataset](https://drive.google.com/file/d/1CuaoMAUII9iyVcqaSdthnl6IdUvYPoLS/view?usp=sharing)

---

*Tugas Modul 1 — Open Recruitment Lab MCI 2026*
*Dibuat menggunakan Python, Apache PySpark, dan Spark MLlib*
