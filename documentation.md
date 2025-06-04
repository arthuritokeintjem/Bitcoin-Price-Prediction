# Laporan Proyek Machine Learning - Prediksi Harga Bitcoin

## Domain Proyek

Bitcoin (BTC), sebagai pionir mata uang kripto, telah merevolusi pandangan terhadap aset digital dan sistem keuangan. Sejak kemunculannya, Bitcoin tidak hanya menarik minat investor individu tetapi juga institusi besar, menjadikannya salah satu aset dengan kapitalisasi pasar terbesar di dunia. Namun, salah satu karakteristik paling menonjol dari Bitcoin adalah volatilitas harganya yang ekstrem. Pergerakan harga yang tajam dan seringkali tidak terduga dalam waktu singkat menawarkan potensi keuntungan yang besar, tetapi di sisi lain juga membawa risiko kerugian yang signifikan. Volatilitas ini dipengaruhi oleh berbagai faktor, termasuk sentimen pasar, berita regulasi, perkembangan teknologi blockchain, adopsi oleh institusi, hingga faktor makroekonomi global.

Kemampuan untuk memprediksi arah dan besaran pergerakan harga Bitcoin menjadi sangat krusial, tidak hanya bagi para _trader_ harian yang mencari profit jangka pendek, tetapi juga bagi investor jangka panjang yang ingin mengoptimalkan strategi investasi mereka dan mengelola risiko portofolio. Model prediksi yang akurat dapat memberikan _insight_ berharga untuk menentukan waktu masuk atau keluar pasar, melakukan alokasi aset, dan mengembangkan produk keuangan berbasis Bitcoin. Oleh karena itu, proyek ini berfokus pada eksplorasi dan implementasi berbagai model _time series forecasting_, mulai dari metode statistik klasik hingga pendekatan _deep learning_ yang lebih canggih, untuk memprediksi harga penutupan harian Bitcoin berdasarkan data historisnya. Tujuannya adalah untuk menemukan model yang paling efektif dan memberikan pemahaman lebih dalam mengenai dinamika harga Bitcoin.

## Business Understanding

### Problem Statements
1.  **Kesulitan Prediksi Akibat Volatilitas Tinggi:** Harga Bitcoin menunjukkan fluktuasi yang sangat tinggi dan seringkali tidak dapat diprediksi dengan metode konvensional, sehingga menyulitkan investor dan _trader_ dalam membuat keputusan investasi yang tepat waktu dan menguntungkan, serta meningkatkan eksposur risiko.
2.  **Perbandingan Efektivitas Model:** Terdapat berbagai pendekatan untuk _time series forecasting_, mulai dari model statistik tradisional hingga model _deep learning_ yang kompleks. Belum ada konsensus mengenai model mana yang secara konsisten paling unggul untuk prediksi harga Bitcoin, mengingat karakteristik data yang unik (non-linear, non-stasioner, dan dipengaruhi banyak faktor eksternal). Oleh karena itu, perlu dilakukan perbandingan kinerja antara model statistik klasik (ARIMA) dengan model _deep learning_ (GRU dan LSTM) pada dataset harga Bitcoin yang spesifik.
3.  **Optimalisasi Akurasi Prediksi:** Untuk memaksimalkan kegunaan praktis dari model prediksi, diperlukan upaya untuk mencapai tingkat akurasi setinggi mungkin. Hal ini melibatkan pemilihan model yang tepat, persiapan data yang cermat, dan evaluasi yang komprehensif untuk mengidentifikasi model mana di antara ARIMA, GRU, dan LSTM yang mampu memberikan prediksi harga penutupan harian Bitcoin dengan error terkecil berdasarkan metrik RMSE dan MAPE.

### Goals
1.  **Mengembangkan dan Mengevaluasi Model Prediksi:** Mengimplementasikan serangkaian model _time series forecasting_ yang mencakup ARIMA sebagai representasi model statistik, serta GRU dan LSTM sebagai representasi model _deep learning_, untuk memprediksi harga penutupan harian Bitcoin, dan mengevaluasi performanya secara kuantitatif.
2.  **Menganalisis Karakteristik dan Kinerja Model:** Memahami cara kerja dasar masing-masing model (ARIMA, GRU, LSTM), parameter-parameter kunci yang memengaruhinya, serta menganalisis bagaimana setiap model menangkap pola dalam data harga Bitcoin. Perbandingan ini akan menyoroti kelebihan dan kekurangan masing-masing pendekatan pada kasus prediksi harga Bitcoin.
3.  **Mengidentifikasi Model Terbaik untuk Prediksi Harga Bitcoin:** Berdasarkan evaluasi menggunakan metrik Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE) pada data uji, menentukan model mana di antara ARIMA, GRU, dan LSTM yang paling akurat dan andal untuk digunakan dalam prediksi harga penutupan harian Bitcoin pada dataset yang digunakan dalam proyek ini.

### Solution statements
Untuk mencapai tujuan-tujuan di atas, proyek ini akan menerapkan solusi metodologis sebagai berikut:
1.  **Akuisisi dan Pra-pemrosesan Data:** Mengunduh data harga historis Bitcoin (BTC-USD) harian dari Yahoo Finance (`yfinance`), melakukan pembersihan data jika diperlukan, dan melakukan analisis data eksploratif (EDA) untuk memahami karakteristik data.
2.  **Pengembangan Model ARIMA:**
    * Melakukan uji stasioneritas pada data harga penutupan.
    * Menggunakan fungsi `auto_arima` dari library `pmdarima` untuk secara otomatis mengidentifikasi orde (p,d,q) yang optimal berdasarkan _Akaike Information Criterion_ (AIC).
    * Melatih model ARIMA pada data latih dan menghasilkan prediksi pada data uji.
3.  **Pengembangan Model GRU (Gated Recurrent Unit):**
    * Mempersiapkan data dengan normalisasi (`MinMaxScaler`) dan pembuatan sekuens (menggunakan _time step_ 60 hari).
    * Membangun arsitektur model GRU dengan dua layer GRU (masing-masing 50 unit) dan layer Dropout untuk regularisasi, diakhiri dengan layer Dense sebagai output.
    * Melatih model GRU selama 50 epoch menggunakan optimizer 'adam' dan loss 'mean_squared_error'.
4.  **Pengembangan Model LSTM (Long Short-Term Memory):**
    * Mempersiapkan data dengan normalisasi dan pembuatan sekuens (menggunakan _sequence length_ 30 hari).
    * Membangun arsitektur model LSTM dengan satu layer LSTM (50 unit) dan layer Dense sebagai output.
    * Melatih model LSTM selama 50 epoch, juga dengan optimizer 'adam', loss 'mean_squared_error', dan menyertakan callback `EarlyStopping` untuk mencegah _overfitting_.
5.  **Evaluasi dan Pemilihan Model:**
    * Mengevaluasi ketiga model (ARIMA, GRU, LSTM) pada data uji menggunakan metrik RMSE dan MAPE.
    * Membandingkan hasil evaluasi untuk menentukan model mana yang memberikan prediksi paling akurat dan dapat diandalkan untuk kasus penggunaan ini.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data historis harga Bitcoin dengan ticker `BTC-USD`, yang diunduh secara harian dari Yahoo Finance menggunakan library `yfinance` pada Python.
* **Sumber Data:** Yahoo Finance (diakses melalui `yfinance`).
* **Ticker:** `BTC-USD`.
* **Rentang Waktu Pengambilan Data:** Data untuk proyek ini diambil mulai tanggal 1 Januari 2020 hingga tanggal eksekusi notebook. Berdasarkan eksekusi notebook (Cell 9), periode ini menghasilkan 1972 sampel data harian, yang sudah lebih dari cukup untuk memenuhi syarat minimal 500 sampel data.
* **Fitur Utama yang Digunakan dari `yfinance`:**
    * `Date`: Tanggal pencatatan data. Dalam notebook, kolom ini dijadikan sebagai index DataFrame.
    * `Open`: Harga pembukaan Bitcoin pada hari perdagangan tersebut.
    * `High`: Harga tertinggi yang dicapai Bitcoin pada hari perdagangan tersebut.
    * `Low`: Harga terendah yang dicapai Bitcoin pada hari perdagangan tersebut.
    * `Close`: Harga penutupan Bitcoin pada hari perdagangan tersebut. **Ini adalah variabel target utama yang akan diprediksi oleh model-model kita.**
    * `Volume`: Volume transaksi Bitcoin yang terjadi pada hari perdagangan tersebut.
    *(Catatan: Kolom 'Adj Close' yang juga disediakan oleh yfinance tidak secara eksplisit digunakan lebih lanjut dalam analisis ini karena untuk aset kripto seperti Bitcoin, nilainya seringkali identik dengan harga 'Close').*

**Tahapan _Exploratory Data Analysis_ (EDA) yang Dilakukan (mengacu pada Notebook):**
EDA merupakan langkah krusial untuk memahami karakteristik data sebelum melakukan pemodelan.
1.  **Visualisasi Harga Penutupan Historis (Cell 13):** Plot harga penutupan Bitcoin dari waktu ke waktu dibuat untuk mengamati tren jangka panjang, pola musiman (jika ada secara visual), dan tingkat volatilitas secara umum. Dari plot ini, terlihat adanya tren naik yang signifikan dalam periode tertentu, diikuti oleh periode koreksi atau konsolidasi, serta fluktuasi harga harian yang cukup besar.
2.  **Uji Stasioneritas Data (Cell 14, 15):** Stasioneritas adalah asumsi penting untuk beberapa model _time series_ seperti ARIMA. _Augmented Dickey-Fuller (ADF) Test_ digunakan untuk menguji apakah data harga penutupan bersifat stasioner atau non-stasioner.
    * **Hasil Uji ADF pada Data Asli:** Output pada Cell 14 menunjukkan nilai p-value yang lebih besar dari 0.05. Ini mengindikasikan bahwa kita gagal menolak hipotesis nol (data memiliki unit root), sehingga data harga penutupan asli bersifat non-stasioner.
    * **Hasil Uji ADF pada Data Setelah _1st Differencing_:** Setelah melakukan diferensiasi pertama (mengambil selisih antara nilai saat ini dan nilai sebelumnya), uji ADF kembali dilakukan (Cell 15). Kali ini, p-value yang dihasilkan jauh lebih kecil dari 0.05, yang berarti kita dapat menolak hipotesis nol. Ini menunjukkan bahwa data harga penutupan Bitcoin menjadi stasioner setelah satu kali diferensiasi. _Insight_ ini penting untuk menentukan parameter `d` pada model ARIMA.
3.  **Analisis Fungsi Autokorelasi (ACF) dan Autokorelasi Parsial (PACF) (Cell 16):** Plot ACF dan PACF dari data harga yang telah di-differencing pertama digunakan untuk membantu mengidentifikasi orde autoregresif (p) dan _moving average_ (q) untuk model ARIMA.
    * Plot ACF menunjukkan bagaimana korelasi antara _time series_ dengan lag-nya menurun.
    * Plot PACF menunjukkan korelasi antara _time series_ dengan lag tertentu setelah menghilangkan efek dari lag-lag yang lebih pendek.
    Pola spesifik pada kedua plot ini (misalnya, _cut-off_ tajam atau penurunan bertahap) memberikan petunjuk visual untuk pemilihan orde p dan q.
4.  **Pengecekan Nilai Hilang (Cell 10):** Dilakukan pemeriksaan terhadap _missing values_ pada dataset awal yang diunduh. Berdasarkan output notebook, tidak ditemukan adanya nilai yang hilang pada kolom-kolom utama (`Open`, `High`, `Low`, `Close`, `Volume`).

## Data Preparation
Tahap persiapan data adalah fondasi penting untuk membangun model machine learning yang baik. Urutan langkah yang dilakukan dalam notebook adalah sebagai berikut:

1.  **Penanganan Nilai Hilang Awal (Cell 11):** Meskipun hasil pengecekan pada Cell 10 menunjukkan tidak ada nilai null pada data yang baru diunduh, sebagai praktik terbaik dan untuk mengantisipasi jika data dari sumber lain atau pada waktu lain mungkin memiliki _missing values_, langkah `btc_data.ffill(inplace=True)` tetap dijalankan. Metode _forward fill_ (`ffill`) mengisi nilai yang hilang dengan nilai valid terakhir sebelumnya. Ini adalah pendekatan yang umum dan logis untuk data _time series_ karena mempertahankan informasi dari observasi terakhir yang diketahui dan menghindari memasukkan bias dari data masa depan.
2.  **Seleksi Fitur Target (Cell 18):** Dari berbagai kolom yang tersedia, kolom 'Close' (harga penutupan) dipilih sebagai variabel dependen atau target yang akan diprediksi oleh semua model. Ini adalah praktik standar dalam prediksi harga aset finansial. Variabel `data_for_arima` dan `data_for_dl` dibuat berdasarkan kolom ini.
3.  **Pembagian Data menjadi Set Latih dan Set Uji (Cell 19):** Dataset dibagi menjadi dua bagian utama: data latih (_training set_) dan data uji (_test set_). Pembagian dilakukan secara kronologis, di mana 80% data awal digunakan sebagai data latih, dan 20% data terakhir digunakan sebagai data uji. Pemisahan kronologis ini sangat krusial untuk model _time series_ untuk mencegah _data leakage_, yaitu situasi di mana model "melihat" data dari masa depan selama proses pelatihan, yang akan menghasilkan evaluasi kinerja yang tidak realistis dan terlalu optimis.
4.  **Normalisasi Data (Khusus untuk Model Deep Learning GRU dan LSTM) (Cell 25 untuk GRU, Cell 30 untuk LSTM):**
    * **Alasan Normalisasi:** Model _deep learning_ seperti GRU dan LSTM seringkali lebih sensitif terhadap skala data input. Fitur dengan rentang nilai yang besar dapat mendominasi proses pembelajaran dan menyebabkan konvergensi yang lambat atau bahkan hasil yang tidak optimal. Normalisasi membantu menstandarkan rentang nilai fitur, biasanya ke rentang [0, 1] atau [-1, 1].
    * **Proses:** `MinMaxScaler` dari `sklearn.preprocessing` digunakan untuk mengubah skala data harga 'Close'. Penting untuk dicatat bahwa _scaler_ ini di-_fit_ **hanya** pada data latih (`train_data['Close']`). Menggunakan informasi dari data uji untuk _fitting scaler_ akan menjadi bentuk _data leakage_ lain. Setelah di-_fit_ pada data latih, _scaler_ yang sama kemudian digunakan untuk mentransformasi (mengubah skala) baik data latih maupun data uji. Ini memastikan bahwa transformasi pada data uji konsisten dengan apa yang telah dipelajari model dari data latih.
5.  **Pembuatan Sekuens Data (Khusus untuk Model Deep Learning GRU dan LSTM) (Cell 26 untuk GRU, Cell 30 untuk LSTM):**
    * **Alasan Pembuatan Sekuens:** Model RNN (termasuk GRU dan LSTM) dirancang untuk memproses data sekuensial. Mereka tidak melihat data sebagai titik-titik individual, melainkan sebagai urutan di mana konteks dari observasi sebelumnya penting. Oleh karena itu, data _time series_ perlu diubah menjadi format sekuens input-output.
    * **Proses untuk Model GRU (Cell 26, fungsi `prepare_data`):** Data yang telah dinormalisasi diubah menjadi sekuens-sekuens di mana setiap sampel input terdiri dari data harga penutupan dari 60 hari sebelumnya (`time_step = 60`), dan targetnya adalah harga penutupan pada hari ke-61.
    * **Proses untuk Model LSTM (Cell 30, fungsi `create_sequences`):** Serupa dengan GRU, data yang telah dinormalisasi diubah menjadi sekuens. Untuk model LSTM ini, panjang sekuens input yang digunakan adalah 30 hari (`SEQ_LENGTH = 30`) untuk memprediksi harga pada hari ke-31. Penggunaan panjang sekuens yang berbeda memungkinkan eksplorasi bagaimana _look-back window_ yang berbeda memengaruhi performa model.

## Modeling
Tiga model _time series forecasting_ dikembangkan: ARIMA, GRU, dan LSTM.

1.  **ARIMA (Autoregressive Integrated Moving Average):**
    * **Cara Kerja Dasar:** ARIMA adalah model statistik linear yang populer untuk menganalisis dan memprediksi data _time series_ yang stasioner atau dapat dibuat stasioner melalui _differencing_. Model ini menggabungkan tiga komponen utama:
        * **AR (Autoregressive) - Komponen `p`:** Bagian ini mengasumsikan bahwa nilai variabel pada suatu waktu tertentu dipengaruhi secara linear oleh nilai-nilai sebelumnya (lag observasi). Orde `p` menentukan berapa banyak periode waktu sebelumnya yang dimasukkan ke dalam model untuk prediksi.
        * **I (Integrated) - Komponen `d`:** Komponen ini merujuk pada proses _differencing_ yang dilakukan untuk membuat _time series_ menjadi stasioner. Orde `d` adalah jumlah berapa kali _differencing_ diterapkan.
        * **MA (Moving Average) - Komponen `q`:** Bagian ini memodelkan error prediksi saat ini sebagai kombinasi linear dari error-error prediksi pada periode waktu sebelumnya. Orde `q` menentukan jumlah error _lag_ yang dipertimbangkan.
    * **Parameter Penting & Implementasi di Notebook:**
        * **Pemilihan Orde (p,d,q):** Fungsi `auto_arima` dari library `pmdarima` digunakan untuk secara otomatis mencari kombinasi parameter (p,d,q) yang optimal.
        * **`auto_arima` Parameter yang Digunakan (sesuai kode di notebook):**
            * `train_data['Close']`: Data latih yang digunakan.
            * `start_p=1`, `start_q=1`: Nilai awal untuk orde p dan q.
            * `max_p=5`, `max_q=5`: Batas maksimum pencarian untuk orde p dan q.
            * `start_P=0`: Parameter awal untuk bagian musiman (meskipun `seasonal=False`).
            * `seasonal=False`: Model non-musiman yang dicari.
            * `d=None`: Membiarkan `auto_arima` menentukan orde `d` yang optimal berdasarkan uji stasioneritas internal (seperti ADF test).
            * `trace=True`: Menampilkan proses pencarian model.
            * `error_action='ignore'`: Mengabaikan error jika beberapa kombinasi parameter tidak valid.
            * `suppress_warnings=True`: Menyembunyikan peringatan selama pencarian.
            * `stepwise=True`: Menggunakan algoritma pencarian _stepwise_ yang efisien.
        * **Hasil `auto_arima`:** Model terbaik yang diidentifikasi adalah **ARIMA(1,1,0)**. Ini berarti:
            * `p=1`: Menggunakan satu nilai observasi sebelumnya.
            * `d=1`: Satu kali _differencing_ diterapkan (ditemukan secara otomatis oleh `auto_arima`).
            * `q=0`: Tidak ada komponen _moving average_ dari error yang digunakan.
        Model ini kemudian dilatih pada `train_data['Close']` dan digunakan untuk prediksi pada `test_data`.

2.  **GRU (Gated Recurrent Unit):**
    * **Cara Kerja Dasar:** GRU adalah jenis _Recurrent Neural Network_ (RNN) yang dirancang untuk mengatasi masalah _vanishing gradient_. GRU memiliki mekanisme gerbang (_gates_) yang secara selektif mengatur aliran informasi, memungkinkan model untuk mengingat informasi relevan dari masa lalu dan melupakan yang tidak penting. GRU lebih sederhana dari LSTM dengan dua gerbang utama:
        * **Reset Gate ($r_t$):** Menentukan seberapa banyak informasi dari _hidden state_ sebelumnya ($h_{t-1}$) yang akan diabaikan.
        * **Update Gate ($z_t$):** Mengontrol seberapa banyak informasi dari _hidden state_ sebelumnya ($h_{t-1}$) yang akan diteruskan dan seberapa banyak informasi dari kandidat _hidden state_ baru (${\tilde{h}}_t$) yang akan digunakan.
    * **Parameter Penting & Implementasi di Notebook (Fungsi `create_gru_model` dan `gru_model.fit`):**
        * **Arsitektur Model (`create_gru_model`):**
            * `GRU(50, return_sequences=True, input_shape=input_shape)`: Layer GRU pertama dengan 50 unit. Fungsi aktivasi default untuk GRU di Keras adalah **'tanh'** dan `recurrent_activation` adalah **'sigmoid'**. `return_sequences=True` karena outputnya akan menjadi input layer GRU berikutnya. `input_shape` adalah (60, 1) karena `time_step=60` dan 1 fitur.
            * `Dropout(0.2)`: Layer Dropout pertama.
            * `GRU(50, return_sequences=False)`: Layer GRU kedua dengan 50 unit. Fungsi aktivasi default **'tanh'** dan `recurrent_activation` **'sigmoid'**. `return_sequences=False` karena ini adalah layer rekuren terakhir sebelum layer Dense output.
            * `Dropout(0.2)`: Layer Dropout kedua.
            * `Dense(1)`: Layer output dengan 1 unit (aktivasi linear default).
        * **Kompilasi Model:** `optimizer='adam'`, `loss='mean_squared_error'`.
        * **Pelatihan Model (`gru_model.fit`):** `epochs=50`, `batch_size=32`. `validation_data=(X_test, y_test)` digunakan.

3.  **LSTM (Long Short-Term Memory):**
    * **Cara Kerja Dasar:** LSTM juga merupakan jenis RNN yang sangat efektif dalam mempelajari dependensi jangka panjang. LSTM memiliki struktur internal yang lebih kompleks dengan tiga gerbang utama dan sebuah _cell state_ ($C_t$).
        * **Forget Gate ($f_t$):** Memutuskan informasi mana dari _cell state_ sebelumnya ($C_{t-1}$) yang harus dibuang.
        * **Input Gate ($i_t$):** Memutuskan nilai baru mana yang akan disimpan dalam _cell state_.
        * **Cell State Update:** Memperbarui _cell state_ lama ($C_{t-1}$) menjadi _cell state_ baru ($C_t$).
        * **Output Gate ($o_t$):** Mengontrol bagian mana dari _cell state_ saat ini ($C_t$) yang akan dikeluarkan sebagai _hidden state_ berikutnya ($h_t$).
    * **Parameter Penting & Implementasi di Notebook (Definisi `lstm_model` dan `history = lstm_model.fit(...)`):**
        * **Arsitektur Model:**
            * `LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1))`: Satu layer LSTM dengan 50 unit. Fungsi aktivasi default untuk LSTM di Keras adalah **'tanh'** dan `recurrent_activation` adalah **'sigmoid'**. `return_sequences=False` karena ini adalah layer LSTM terakhir sebelum layer Dense. `input_shape` adalah (30, 1) karena `SEQ_LENGTH=30` dan 1 fitur.
            * `Dense(1)`: Layer output dengan 1 unit (aktivasi linear default).
        * **Kompilasi Model:** `optimizer='adam'`, `loss='mean_squared_error'`.
        * **Pelatihan Model (`lstm_model.fit`):** `epochs=50`, `batch_size=32`. `validation_split=0.1`. `callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]`.
        
**Kelebihan dan Kekurangan Model serta Pertimbangan Pemilihan:**

* **ARIMA**:
    * **Kelebihan**: Relatif sederhana untuk diimplementasikan dan diinterpretasikan, terutama untuk data time series univariat dengan pola linear yang jelas dan stasioner (atau dapat distasionerkan). Efisien secara komputasional.
    * **Kekurangan**: Kurang efektif dalam menangkap pola non-linear yang kompleks yang sering ditemukan dalam data keuangan seperti harga Bitcoin. Membutuhkan data stasioner, sehingga perlu _differencing_. Pemilihan order (p,d,q) bisa subjektif atau memerlukan proses iteratif.

* **GRU**:
    * **Kelebihan**: Lebih sederhana dan lebih cepat dilatih daripada LSTM karena memiliki lebih sedikit parameter (dua _gates_ vs tiga). Dapat memberikan performa yang sebanding dengan LSTM pada banyak tugas, terutama jika dataset tidak terlalu besar atau ketergantungan jangka panjang tidak terlalu kompleks.
    * **Kekurangan**: Mungkin tidak seefektif LSTM dalam menangkap ketergantungan jangka panjang yang sangat kompleks atau halus dalam data.

* **LSTM**:
    * **Kelebihan**: Sangat efektif dalam menangani ketergantungan jangka panjang dan pola non-linear yang kompleks dalam data sekuensial, menjadikannya cocok untuk data harga Bitcoin yang volatil. Lebih robust terhadap masalah _vanishing/exploding gradient_ dibandingkan RNN standar.
    * **Kekurangan**: Lebih kompleks secara arsitektur dan komputasional dibandingkan GRU dan ARIMA, sehingga memerlukan waktu pelatihan yang lebih lama dan potensi kebutuhan data yang lebih besar untuk performa optimal. Interpretasi model bisa lebih sulit.

## Evaluation
Kinerja ketiga model (ARIMA, GRU, dan LSTM) dievaluasi pada data uji menggunakan dua metrik utama yang umum digunakan untuk tugas regresi dan _time series forecasting_: Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE). Fungsi `evaluate_model` (Cell 33) dibuat untuk menghitung kedua metrik ini.

1.  **RMSE (Root Mean Squared Error):**
    * **Formula:** $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
        Di mana $n$ adalah jumlah sampel, $y_i$ adalah nilai aktual, dan $\hat{y}_i$ adalah nilai prediksi.
    * **Cara Kerja dan Interpretasi:** RMSE mengukur standar deviasi dari residual (error prediksi). Karena error dikuadratkan sebelum dirata-ratakan, RMSE memberikan bobot yang lebih besar pada error yang besar. Ini berarti model akan mendapatkan "hukuman" yang lebih berat jika membuat prediksi yang sangat jauh dari nilai aktual. Nilai RMSE dinyatakan dalam unit yang sama dengan variabel target (dalam kasus ini, USD untuk harga Bitcoin). Semakin rendah nilai RMSE, semakin baik kinerja model dalam memprediksi nilai aktual, karena menunjukkan error prediksi yang lebih kecil secara rata-rata.
2.  **MAPE (Mean Absolute Percentage Error):**
    * **Formula:** $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$
    * **Cara Kerja dan Interpretasi:** MAPE mengukur rata-rata persentase error absolut antara nilai aktual dan prediksi, relatif terhadap nilai aktual. Keuntungan utama MAPE adalah ia tidak bersatuan (dinyatakan dalam persen), sehingga mudah diinterpretasikan dan dapat digunakan untuk membandingkan akurasi prediksi antar dataset dengan skala yang berbeda (meskipun dalam proyek ini hanya satu dataset). Misalnya, MAPE 5% berarti prediksi model rata-rata menyimpang 5% dari nilai aktual. Semakin rendah nilai MAPE, semakin akurat modelnya. Namun, perlu diperhatikan bahwa MAPE bisa menjadi tidak terdefinisi atau sangat besar jika nilai aktual ($y_i$) ada yang nol atau sangat mendekati nol (tidak menjadi masalah signifikan untuk harga Bitcoin).

**Hasil Evaluasi (berdasarkan output Cell 118 di notebook):**
Setelah melatih ketiga model dan melakukan prediksi pada data uji, hasil evaluasi metrik adalah sebagai berikut:

| Model                      | RMSE (USD)    | MAPE (%) |
| :------------------------- | :------------ | :------- |
| ARIMA                      | 21937.81      | 18.60    |
| GRU                        | 15195.96      | 15.99    |
| **LSTM (Bagian 5.3)** | **2168.12** | **2.02** |

Berdasarkan tabel di atas, dapat disimpulkan:
* **Model LSTM (yang didefinisikan di bagian 5.3 notebook)** menunjukkan kinerja yang jauh lebih unggul dibandingkan dua model lainnya. Model ini berhasil mencapai **RMSE sebesar 2168.12 USD** dan **MAPE sebesar 2.02%**. Nilai error yang rendah ini mengindikasikan bahwa prediksi model LSTM sangat dekat dengan harga aktual Bitcoin pada periode uji.
* Model GRU menempati posisi kedua dengan RMSE 15195.96 USD dan MAPE 15.99%. Meskipun lebih baik dari ARIMA, performanya masih jauh di bawah model LSTM.
* Model ARIMA menunjukkan performa terendah dengan RMSE 21937.81 USD dan MAPE 18.60%. Ini mungkin mengindikasikan bahwa model linier seperti ARIMA kurang mampu menangkap pola non-linear yang kompleks dalam data harga Bitcoin dibandingkan model _deep learning_.

Dengan demikian, untuk dataset dan konfigurasi yang digunakan dalam proyek ini, **model LSTM (Bagian 5.3)** adalah model terbaik untuk memprediksi harga penutupan harian Bitcoin.

*(Kesimpulan akhir, termasuk diskusi mengenai keterbatasan model GRU akibat penggunaan data uji sebagai validasi, serta potensi pengembangan seperti hyperparameter tuning lebih lanjut untuk model deep learning, penggunaan fitur tambahan, atau eksplorasi arsitektur model yang lebih canggih, dapat ditambahkan di sini oleh Anda berdasarkan Cell 35).*