import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Toyota Bekas",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model dan scaler
model = load_model('ann_model.h5')
scaler = joblib.load('scaler.pkl')

# Gambar header/logo
st.image("https://alurnews.com/wp-content/uploads/2023/01/toyota.jpg", width=200)

# Judul aplikasi
st.markdown("<h1 style='text-align: center; color: #d32f2f;'>Prediksi Harga Mobil Toyota Bekas</h1>", unsafe_allow_html=True)
st.markdown("---")

# Upload gambar (opsional)
uploaded_file = st.file_uploader("📸 Upload Foto Mobil (opsional)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Foto Mobil yang Diunggah", use_column_width=True)
    st.markdown("---")

# Form input data mobil
with st.form("input_form"):
    st.subheader("🔧 Masukkan Data Mobil:")
    year = st.number_input('Tahun Mobil', min_value=1990, max_value=2025, value=2015)
    mileage = st.number_input('Jarak Tempuh (km)', min_value=0, max_value=300000, value=50000)
    tax = st.number_input('Pajak (£)', min_value=0, max_value=600, value=150)
    mpg = st.number_input('MPG (miles per gallon)', min_value=10.0, max_value=100.0, value=50.0)
    engineSize = st.number_input('Ukuran Mesin (L)', min_value=0.5, max_value=6.0, value=1.6, step=0.1)
    transmission = st.selectbox('Transmisi', ['Manual', 'Automatic', 'Semi-Auto'])
    fuel = st.selectbox('Tipe Bahan Bakar', ['Petrol', 'Diesel', 'Hybrid', 'Other'])

    submitted = st.form_submit_button("🔍 Prediksi Harga")

# Proses prediksi
if submitted:
    input_dict = {
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize,
        'transmission_Manual': 1 if transmission == 'Manual' else 0,
        'transmission_Semi-Auto': 1 if transmission == 'Semi-Auto' else 0,
        'fuelType_Diesel': 1 if fuel == 'Diesel' else 0,
        'fuelType_Hybrid': 1 if fuel == 'Hybrid' else 0,
        'fuelType_Other': 1 if fuel == 'Other' else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.markdown("### 💰 Perkiraan Harga Mobil:")
    st.success(f"£ {prediction[0][0]:,.2f}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Prediksi Toyota Bekas | Dibuat dengan ❤️ oleh [Nama Kamu]</p>", unsafe_allow_html=True)
