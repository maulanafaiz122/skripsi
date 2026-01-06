import streamlit as st
import pandas as pd
import os
from PIL import Image

# =============================
# PATH ABSOLUTE (ANTI ERROR)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artefak")

PATHS = {
    "confusion_matrix": os.path.join(ARTIFACT_DIR, "confusion_matrix_final.png"),
    "distribusi_kelas": os.path.join(ARTIFACT_DIR, "distribusi_kelas_powerset.png"),
    "classification_csv": os.path.join(ARTIFACT_DIR, "classification_report_final.csv"),
    "classification_xlsx": os.path.join(ARTIFACT_DIR, "classification_report_final.xlsx"),
    "metrics_summary": os.path.join(ARTIFACT_DIR, "final_metrics_summary.txt"),
    "perbandingan": os.path.join(ARTIFACT_DIR, "perbandingan_skenario_skripsi.xlsx"),
    "log_harian": os.path.join(ARTIFACT_DIR, "log_harian.csv"),
    "hasil_powerset": os.path.join(ARTIFACT_DIR, "hasil_powerset.xlsx"),
}

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Hasil Penelitian – Skripsi",
    layout="wide"
)

st.title("📊 Hasil Penelitian Analisis Sentimen Multi-Label")
st.markdown("""
Aplikasi ini menampilkan **hasil akhir penelitian skripsi**, meliputi:
- Evaluasi model terbaik
- Distribusi kelas Powerset
- Classification Report
- Ringkasan metrik performa
- Perbandingan skenario eksperimen
""")

def file_exists(path):
    return os.path.isfile(path)

# =============================
# CONFUSION MATRIX
# =============================
st.header("🧩 Confusion Matrix Model Terbaik")
if file_exists(PATHS["confusion_matrix"]):
    st.image(Image.open(PATHS["confusion_matrix"]), use_container_width=True)
else:
    st.warning("❌ confusion_matrix_final.png tidak ditemukan")

# =============================
# DISTRIBUSI KELAS
# =============================
st.header("📈 Distribusi Kelas Powerset")
if file_exists(PATHS["distribusi_kelas"]):
    st.image(Image.open(PATHS["distribusi_kelas"]), use_container_width=True)
else:
    st.warning("❌ distribusi_kelas_powerset.png tidak ditemukan")

# =============================
# CLASSIFICATION REPORT
# =============================
st.header("📋 Classification Report")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Versi CSV")
    if file_exists(PATHS["classification_csv"]):
        df = pd.read_csv(PATHS["classification_csv"])
        st.dataframe(df, hide_index=True)
    else:
        st.warning("❌ classification_report_final.csv tidak ditemukan")

with col2:
    st.subheader("Versi Excel")
    if file_exists(PATHS["classification_xlsx"]):
        df = pd.read_excel(PATHS["classification_xlsx"])
        st.dataframe(df, hide_index=True)
    else:
        st.warning("❌ classification_report_final.xlsx tidak ditemukan")

# =============================
# METRICS SUMMARY
# =============================
st.header("🧮 Ringkasan Metrik Performa")
if file_exists(PATHS["metrics_summary"]):
    with open(PATHS["metrics_summary"], "r", encoding="utf-8") as f:
        st.code(f.read())
else:
    st.warning("❌ final_metrics_summary.txt tidak ditemukan")

# =============================
# PERBANDINGAN SKENARIO
# =============================
st.header("⚖️ Perbandingan Skenario Eksperimen")
if file_exists(PATHS["perbandingan"]):
    df = pd.read_excel(PATHS["perbandingan"])
    st.dataframe(df, use_container_width=True)
else:
    st.warning("❌ perbandingan_skenario_skripsi.xlsx tidak ditemukan")

# =============================
# DETAIL KELAS POWERSET (ANTI ERROR)
# =============================
st.header("🧠 Detail Kelas Powerset")

if file_exists(PATHS["hasil_powerset"]):
    df_kelas = pd.read_excel(PATHS["hasil_powerset"])

    if "Kelas_ID" in df_kelas.columns and "Detail" in df_kelas.columns:
        st.dataframe(df_kelas, use_container_width=True)
    else:
        st.warning("⚠️ Kolom Kelas_ID / Detail tidak ditemukan di hasil_powerset.xlsx")
else:
    st.info("ℹ️ Detail aspek tidak ditemukan (hasil_powerset.xlsx)")

# =============================
# LOG EKSPERIMEN
# =============================
st.header("🗂️ Log Proses Eksperimen")
if file_exists(PATHS["log_harian"]):
    df_log = pd.read_csv(PATHS["log_harian"], engine="python", on_bad_lines="skip")
    st.dataframe(df_log, use_container_width=True)
else:
    st.info("ℹ️ Log harian tidak tersedia")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Skripsi Analisis Sentimen BBM Pertamina | "
    "Word2Vec (Skip-gram) + LSTM | Multi-Label Powerset"
)
