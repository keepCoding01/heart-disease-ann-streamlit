# 🫀 Prediksi Risiko Penyakit Jantung Menggunakan Artificial Neural Network (ANN)

Aplikasi berbasis web untuk memprediksi probabilitas risiko penyakit jantung pada pasien menggunakan model **Artificial Neural Network (ANN)** yang dioptimasi dengan teknik *Class Weighting* dan *Confidence Threshold Tuning*.

---

## 📌 Fitur Utama Aplikasi Streamlit

1. **Preset Sample Data Pasien Nyata:** Memuat data instan dari sampel riil dataset UCI Cleveland untuk efisiensi pengujian tanpa menginput 13 indikator medis secara manual saat demonstrasi.
2. **Visualisasi Gauge Chart & Export Laporan CSV:** Menampilkan indikator visual tingkat risiko secara interaktif menggunakan Plotly serta menyediakan fitur unduhan laporan rekam medis pasien dalam format CSV.
3. **Optimasi Confidence Threshold:** Menggunakan batas ambang probabilitas optimal ($0.42$) hasil tuning validasi untuk mengklasifikasikan diagnosis secara presisi.

---

## 🏗️ Arsitektur Model & Metodologi

* **Input Layer:** 13 Fitur Medis (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`).
* **Preprocessing:** Standardisasi skala fitur menggunakan `StandardScaler`.
* **Optimasi Eksperimen:**
  * Penanganan *imbalanced data* menggunakan strategi *Class Weighting* pada *loss function* (`binary_crossentropy`).
  * Optimasi batas keputusan klasifikasi via *Confidence Threshold Search* (rentang uji $0.30$–$0.70$).
* **Output Layer:** Single Neuron dengan fungsi aktivasi Sigmoid.

---

## 📁 Struktur Repositori

```text
.
├── app.py                # Kode utama antarmuka Streamlit (UI & Prediksi)
├── model_ann.h5          # File model tersimpan (Keras/TensorFlow)
├── scaler.pkl            # Objek StandardScaler hasil fitting data latih
├── requirements.txt      # Daftar dependensi library Python
└── README.md             # Dokumentasi utama proyek
```

---

## 🚀 Cara Menjalankan di Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/keepCoding01/heart-disease-ann-streamlit.git
cd heart-disease-ann-streamlit
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

---

## 🌐 Live Demo Application

Aplikasi ini dapat diakses secara publik melalui tautan Streamlit Cloud berikut:  
👉 **[Sistem Prediksi Risiko Jantung ANN](https://heart-disease-ann-app-puvumksktm4pshyouv2phy.streamlit.app/)**