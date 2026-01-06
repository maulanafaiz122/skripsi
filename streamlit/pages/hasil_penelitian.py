import streamlit as st
import pandas as pd
import os
from PIL import Image

# =================================================
# KONFIGURASI PATH (ABSOLUTE – ANTI ERROR DEPLOY)
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artefak"))

PATHS = {
    "confusion_matrix": os.path.join(ARTIFACT_DIR, "confusion_matrix_final.png"),
    "distribusi_kelas": os.path.join(ARTIFACT_DIR, "distribusi_kelas_powerset.png"),
    "classification_csv": os.path.join(ARTIFACT_DIR, "classification_report_final.csv"),
    "classification_xlsx": os.path.join(ARTIFACT_DIR, "classification_report_final.xlsx"),
    "metrics_summary": os.path.join(ARTIFACT_DIR, "final_metrics_summary.txt"),
    "perbandingan": os.path.join(ARTIFACT_DIR, "perbandingan_skenario_skripsi.xlsx"),
    "log_harian": os.path.join(ARTIFACT_DIR, "log_harian.csv"),
}

# =================================================
# KONFIGURASI HALAMAN
# =================================================
st.set_page_config(
    page_title="Hasil Penelitian – Skripsi",
    layout="wide"
)

st.title("📊 Hasil Penelitian Analisis Sentimen Multi-Label")
st.markdown(
    """
    Halaman ini menyajikan **hasil akhir penelitian skripsi** berupa:
    - Confusion Matrix model terbaik  
    - Distribusi kelas Powerset  
    - Classification Report  
    - Ringkasan metrik performa  
    - Perbandingan skenario eksperimen  
    """
)

# =================================================
# FUNGSI UTIL
# =================================================
def file_exists(path):
    return os.path.isfile(path)

# =================================================
# CONFUSION MATRIX
# =================================================
st.header("🧩 Confusion Matrix Model Terbaik")
if file_exists(PATHS["confusion_matrix"]):
    try:
        st.image(
            Image.open(PATHS["confusion_matrix"]),
            caption="Confusion Matrix Final (Model Terbaik)",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Gagal membuka gambar Confusion Matrix: {e}")
else:
    st.info("📌 Confusion Matrix belum tersedia di folder artefak.")

# =================================================
# DISTRIBUSI KELAS
# =================================================
st.header("📈 Distribusi Kelas Powerset")
if file_exists(PATHS["distribusi_kelas"]):
    try:
        st.image(
            Image.open(PATHS["distribusi_kelas"]),
            caption="Distribusi 15 Kelas Powerset",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Gagal membuka gambar distribusi kelas: {e}")
else:
    st.info("📌 Gambar distribusi kelas belum tersedia.")

# =================================================
# CLASSIFICATION REPORT
# =================================================
st.header("📋 Classification Report")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Versi CSV")
    if file_exists(PATHS["classification_csv"]):
        df_csv = pd.read_csv(PATHS["classification_csv"])
        st.dataframe(df_csv, hide_index=True)
    else:
        st.info("📌 File classification_report_final.csv tidak tersedia.")

with col2:
    st.subheader("Versi Excel")
    if file_exists(PATHS["classification_xlsx"]):
        df_xlsx = pd.read_excel(PATHS["classification_xlsx"])
        st.dataframe(df_xlsx, hide_index=True)
    else:
        st.info("📌 File classification_report_final.xlsx tidak tersedia.")

# =================================================
# METRICS SUMMARY
# =================================================
st.header("🧮 Ringkasan Metrik Performa")
if file_exists(PATHS["metrics_summary"]):
    with open(PATHS["metrics_summary"], "r", encoding="utf-8") as f:
        st.code(f.read(), language="text")
else:
    st.info("📌 Ringkasan metrik performa belum disertakan.")

# =================================================
# PERBANDINGAN SKENARIO
# =================================================
st.header("⚖️ Perbandingan Skenario Eksperimen")
if file_exists(PATHS["perbandingan"]):
    df_compare = pd.read_excel(PATHS["perbandingan"])
    st.dataframe(df_compare, use_container_width=True)
else:
    st.info("📌 File perbandingan skenario belum tersedia.")

# =================================================
# LOG EKSPERIMEN
# =================================================
st.header("🗂️ Log Proses Eksperimen")
if file_exists(PATHS["log_harian"]):
    df_log = pd.read_csv(
        PATHS["log_harian"],
        engine="python",
        on_bad_lines="skip"
    )
    st.dataframe(df_log, use_container_width=True)
else:
    st.info("📌 Log eksperimen tidak disertakan.")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption(
    "Model Terbaik: Word2Vec (Skip-gram) + LSTM | Multi-Label Powerset | "
    "Skripsi Analisis Sentimen BBM Pertamina"
)
