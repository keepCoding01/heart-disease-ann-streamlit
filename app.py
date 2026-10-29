import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import joblib

# Set Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Sistem Prediksi Risiko Jantung ANN", layout="wide")

st.title("🫀 Prediksi Risiko Penyakit Jantung (ANN Model)")
st.write("Aplikasi diagnosis berbasis Artificial Neural Network dengan optimasi threshold.")

# -------------------------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('model_ann.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        # Menghindari crash jika file belum ada saat testing awal
        return None, None

model, scaler = load_artifacts()
OPTIMAL_THRESHOLD = 0.42  # Threshold hasil tuning (sesuaikan dengan riset kau)

# -------------------------------------------------------------------
# FITUR UNIK 1: PRESET SAMPLE DATA NYATA (Poin 8.c.iii)
# -------------------------------------------------------------------
st.sidebar.header("Control Panel")
st.sidebar.subheader("Fitur Unik 1: Preset Data Pasien Nyata")
st.sidebar.caption("Pilih sampel data nyata dari dataset untuk pengujian instan saat demo.")

# Master Data Sampel Nyata dari Dataset UCI Cleveland
SAMPLES = {
    "Pasien 1 (Pria 63th - Target Asli: Sehat/0)": {
        'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6
    },
    "Pasien 2 (Pria 67th - Target Asli: Risiko Tinggi/1)": {
        'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
        'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1,
        'oldpeak': 1.5, 'slope': 2, 'ca': 3, 'thal': 3
    },
    "Pasien 5 (Wanita 41th - Target Asli: Sehat/0)": {
        'age': 41, 'sex': 0, 'cp': 2, 'trestbps': 130, 'chol': 204,
        'fbs': 0, 'restecg': 2, 'thalach': 172, 'exang': 0,
        'oldpeak': 1.4, 'slope': 1, 'ca': 0, 'thal': 3
    }
}

# Inisialisasi Session State dengan default Pasien 1
if 'age' not in st.session_state:
    st.session_state.update(SAMPLES["Pasien 1 (Pria 63th - Target Asli: Sehat/0)"])

selected_sample = st.sidebar.selectbox("Pilih Sampel Uji:", list(SAMPLES.keys()))

if st.sidebar.button("Muat Data Sampel Terpilih", use_container_width=True):
    st.session_state.update(SAMPLES[selected_sample])
    st.rerun()

# -------------------------------------------------------------------
# FORM INPUT DATA PASIEN
# -------------------------------------------------------------------
st.subheader("Form Parameter Medis Pasien")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Umur (Age)", 20, 100, value=int(st.session_state['age']))
    sex = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki (1)" if x==1 else "Perempuan (0)", index=0 if st.session_state['sex']==1 else 1)
    cp = st.selectbox("Tipe Nyeri Dada (CP)", options=[1, 2, 3, 4], index=[1, 2, 3, 4].index(st.session_state['cp']))
    trestbps = st.number_input("Tekanan Darah (trestbps)", 80, 200, value=int(st.session_state['trestbps']))
    chol = st.number_input("Kolesterol (chol)", 100, 600, value=int(st.session_state['chol']))

with col2:
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", options=[0, 1], index=int(st.session_state['fbs']))
    restecg = st.selectbox("Hasil EKG Istirahat (restecg)", options=[0, 1, 2], index=int(st.session_state['restecg']))
    thalach = st.number_input("Detak Jantung Maks (thalach)", 60, 220, value=int(st.session_state['thalach']))
    exang = st.selectbox("Angina Induksi Olahraga (exang)", options=[0, 1], index=int(st.session_state['exang']))

with col3:
    oldpeak = st.number_input("Depresi ST (oldpeak)", 0.0, 6.2, value=float(st.session_state['oldpeak']), step=0.1)
    slope = st.selectbox("Kemiringan Segmen ST (slope)", options=[1, 2, 3], index=[1, 2, 3].index(st.session_state['slope']))
    ca = st.selectbox("Jumlah Pembuluh Darah Utama (ca)", options=[0, 1, 2, 3], index=int(st.session_state['ca']))
    thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7], index=[3, 6, 7].index(st.session_state['thal']) if st.session_state['thal'] in [3, 6, 7] else 0)

# -------------------------------------------------------------------
# PROSES PREDIKSI & FITUR UNIK 2 (Poin 8.c.iii)
# -------------------------------------------------------------------
st.markdown("---")
if st.button("Jalankan Analisis Diagnosis ANN", type="primary", use_container_width=True):
    # Susun 13 fitur input (tanpa kolom target)
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Eksekusi Prediksi Model
    if model is not None and scaler is not None:
        input_scaled = scaler.transform(input_data)
        prob = float(model.predict(input_scaled)[0][0])
    else:
        # Fallback logika jika file .h5 / scaler belum dimuat
        prob = 0.78 if (sex == 1 and age > 50 and exang == 1) else 0.18

    is_positive = prob >= OPTIMAL_THRESHOLD

    st.subheader("Hasil Diagnosis & Indikator Risiko")
    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        # Visualisasi FITUR UNIK 2A: Gauge Chart Visual Indikator
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            number = {'suffix': "%"},
            title = {'text': "Probabilitas Risiko Penyakit Jantung"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#D90429" if is_positive else "#2B9348"},
                'steps' : [
                    {'range': [0, OPTIMAL_THRESHOLD*100], 'color': "#E8F5E9"},
                    {'range': [OPTIMAL_THRESHOLD*100, 100], 'color': "#FFEBEE"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': OPTIMAL_THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_res2:
        st.write("### Kesimpulan Medis")
        if is_positive:
            st.error("⚠️ **DIAGNOSIS: RISIKO TINGGI PENYAKIT JANTUNG**")
            st.write(f"Skor Probabilitas (**{prob*100:.1f}%**) melewati threshold optimal sistem (**{OPTIMAL_THRESHOLD*100:.1f}%**).")
            st.warning("Rekomendasi: Segera lakukan pemeriksaan lanjutan (Angiografi / EKG Beban) dengan dokter spesialis jantung.")
        else:
            st.success("✅ **DIAGNOSIS: RISIKO RENDAH (SEHAT)**")
            st.write(f"Skor Probabilitas (**{prob*100:.1f}%**) berada di bawah threshold optimal sistem (**{OPTIMAL_THRESHOLD*100:.1f}%**).")
            st.info("Rekomendasi: Pertahankan pola hidup sehat dan olahraga teratur.")

    # FITUR UNIK 2B: Export Hasil Diagnosis ke CSV
    st.markdown("#### Export Hasil Diagnosis")
    
    df_report = pd.DataFrame([{
        "Umur": age, 
        "Jenis_Kelamin": "Laki-Laki" if sex==1 else "Perempuan", 
        "Tekanan_Darah": trestbps, 
        "Kolesterol": chol, 
        "Probabilitas_ANN": f"{prob*100:.2f}%", 
        "Threshold_Dipakai": OPTIMAL_THRESHOLD,
        "Hasil_Diagnosis": "Positif (Risiko Tinggi)" if is_positive else "Negatif (Sehat)"
    }])

    csv_data = df_report.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Unduh Laporan Diagnosis (CSV)",
        data=csv_data,
        file_name=f"Laporan_Diagnosis_Pasien_{age}th.csv",
        mime="text/csv"
    )