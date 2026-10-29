import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Diagnosis Penyakit Jantung", page_icon="❤️", layout="wide"
)


# Load Artefak Model, Imputer, dan Scaler
@st.cache_resource
def load_artifacts():
    # Ganti format .keras jika kau menyimpan menggunakan .keras
    model = load_model("heart_disease_model.h5")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, imputer, scaler


model, imputer, scaler = load_artifacts()

# Tampilan Header App
st.title("❤️ Sistem Prediksi & Diagnosis Penyakit Jantung (ANN)")
st.write(
    "Masukkan data klinis pasien di bawah ini untuk menguji probabilitas risiko penyakit jantung."
)
st.markdown("---")

# Form Input Data Pasien (Dibagi 2 Kolom biar Rapi)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Demografi & Klinis Dasar")
    age = st.number_input(
        "Umur (Age)", min_value=1, max_value=120, value=50, step=1
    )
    sex = st.selectbox(
        "Jenis Kelamin (Sex)",
        options=[1, 0],
        format_func=lambda x: "Laki-Laki" if x == 1 else "Perempuan",
    )
    cp = st.selectbox(
        "Tipe Nyeri Dada (Chest Pain Type - cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "0: Typical Angina",
            "1: Atypical Angina",
            "2: Non-anginal Pain",
            "3: Asymptomatic",
        ][x],
    )
    trestbps = st.number_input(
        "Tekanan Darah Istirahat / mm Hg (trestbps)",
        min_value=50,
        max_value=250,
        value=120,
    )
    chol = st.number_input(
        "Kolesterol Serum / mg/dl (chol)",
        min_value=100,
        max_value=600,
        value=200,
    )
    fbs = st.selectbox(
        "Gula Darah Puasa > 120 mg/dl (fbs)",
        options=[1, 0],
        format_func=lambda x: "Ya (>120 mg/dl)" if x == 1 else "Tidak",
    )

with col2:
    st.subheader("Hasil Pemeriksaan Medis Tambahan")
    restecg = st.selectbox(
        "Hasil EKG Istirahat (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: [
            "0: Normal",
            "1: ST-T Wave Abnormality",
            "2: Left Ventricular Hypertrophy",
        ][x],
    )
    thalach = st.number_input(
        "Detak Jantung Maksimum (thalach)",
        min_value=50,
        max_value=250,
        value=150,
    )
    exang = st.selectbox(
        "Angina Induksi Olahraga (exang)",
        options=[1, 0],
        format_func=lambda x: "Ya" if x == 1 else "Tidak",
    )
    oldpeak = st.number_input(
        "Depresi ST akibat Olahraga (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    slope = st.selectbox(
        "Kemiringan Segmen ST (slope)",
        options=[0, 1, 2],
        format_func=lambda x: [
            "0: Upsloping",
            "1: Flat",
            "2: Downsloping",
        ][x],
    )
    ca = st.selectbox(
        "Jumlah Pembuluh Darah Utama Mewarna (ca)", options=[0, 1, 2, 3, 4]
    )
    thal = st.selectbox(
        "Hasil Thalium Heart Scan (thal)",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "0: Unknown",
            "1: Normal",
            "2: Fixed Defect",
            "3: Reversable Defect",
        ][x],
    )

st.markdown("---")

# Tombol Prediksi
if st.button("Jalankan Diagnosis Model", type="primary", use_container_width=True):
    # 1. Gabungkan Input Menjadi DataFrame
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ],
    )

    # 2. Preprocessing Data Baru (Imputasi & Scaling pakai transform)
    data_imputed = imputer.transform(input_data)
    data_scaled = scaler.transform(data_imputed)

    # 3. Prediksi Model
    probability = model.predict(data_scaled)[0][0]
    prediction = 1 if probability >= 0.5 else 0

    # 4. Tampilkan Hasil
    st.subheader("Hasil Diagnosis Model:")
    prob_percent = probability * 100

    if prediction == 1:
        st.error(
            f"⚠️ **TERDETEKSI POSITIF RISIKO PENYAKIT JANTUNG**\n\nProbabilitas: **{prob_percent:.2f}%**"
        )
    else:
        st.success(
            f"✅ **NEGATIF / KONDISI JANTUNG SEHAT**\n\nProbabilitas Risiko: **{prob_percent:.2f}%**"
        )

    st.progress(float(probability))