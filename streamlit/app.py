import streamlit as st
import importlib.util
import os

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Aplikasi Skripsi",
    layout="wide"
)

# ================================
# SIDEBAR NAVIGASI
# ================================
st.sidebar.title("📂 Navigasi Halaman")

page = st.sidebar.radio(
    "Pilih Halaman:",
    (
        "🔍 Prediksi Sentimen",
        "📊 Hasil Penelitian"
    )
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_page(file_name):
    file_path = os.path.join(BASE_DIR, file_name)
    spec = importlib.util.spec_from_file_location("page_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

# ================================
# ROUTING
# ================================
if page == "🔍 Prediksi Sentimen":
    load_page("pages/app_prediksi.py")

elif page == "📊 Hasil Penelitian":
    load_page("pages/hasil_penelitian.py")
