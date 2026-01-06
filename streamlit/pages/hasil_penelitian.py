import streamlit as st
import pandas as pd
import os
from PIL import Image

# =============================
# KONFIGURASI PATH ARTEFAK
# =============================
ARTIFACT_DIR = "artefak"

PATHS = {
    "confusion_matrix": os.path.join(ARTIFACT_DIR, "confusion_matrix_final.png"),
    "distribusi_kelas": os.path.join(ARTIFACT_DIR, "distribusi_kelas_powerset.png"),
    "classification_csv": os.path.join(ARTIFACT_DIR, "classification_report_final.csv"),
    "classification_xlsx": os.path.join(ARTIFACT_DIR, "classification_report_final.xlsx"),
    "metrics_summary": os.path.join(ARTIFACT_DIR, "final_metrics_summary.txt"),
    "perbandingan": os.path.join(ARTIFACT_DIR, "perbandingan_skenario_skripsi.xlsx"),
    "log_harian": os.path.join(ARTIFACT_DIR, "log_harian.csv"),
}

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Hasil Penelitian – Skripsi",
    layout="wide"
)

st.title("📊 Hasil Penelitian Analisis Sentimen Multi-Label")
st.markdown(
    """
    Halaman ini menyajikan **hasil akhir penelitian skripsi** berupa:
    - Evaluasi model (Confusion Matrix & Classification Report)
    - Distribusi kelas Powerset
    - Ringkasan metrik performa
    - Perbandingan skenario eksperimen
    """
)

# =============================
# FUNGSI UTILITAS
# =============================
def file_exists(path):
    return os.path.exists(path)

# =============================
# SECTION 1: CONFUSION MATRIX
# =============================
st.header("🧩 Confusion Matrix Model Terbaik")

if file_exists(PATHS["confusion_matrix"]):
    img_cm = Image.open(PATHS["confusion_matrix"])
    st.image(img_cm, caption="Confusion Matrix Final (Model Terbaik)", use_container_width=True)
else:
    st.warning("File confusion_matrix_final.png tidak ditemukan.")

# =============================
# SECTION 2: DISTRIBUSI KELAS
# =============================
st.header("📈 Distribusi Kelas Powerset")

if file_exists(PATHS["distribusi_kelas"]):
    img_dist = Image.open(PATHS["distribusi_kelas"])
    st.image(img_dist, caption="Distribusi Data pada 15 Kelas Powerset", use_container_width=True)
else:
    st.warning("File distribusi_kelas_powerset.png tidak ditemukan.")

# =============================
# SECTION 3: CLASSIFICATION REPORT
# =============================
st.header("📋 Classification Report")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Versi CSV")
    if file_exists(PATHS["classification_csv"]):
        df_report_csv = pd.read_csv(PATHS["classification_csv"])
        st.dataframe(df_report_csv, hide_index=True)
        st.download_button(
            "⬇️ Unduh CSV",
            data=df_report_csv.to_csv(index=False).encode("utf-8"),
            file_name="classification_report_final.csv",
            mime="text/csv"
        )
    else:
        st.warning("File classification_report_final.csv tidak ditemukan.")

with col2:
    st.subheader("Versi Excel")
    if file_exists(PATHS["classification_xlsx"]):
        df_report_xlsx = pd.read_excel(PATHS["classification_xlsx"])
        st.dataframe(df_report_xlsx, hide_index=True)
    else:
        st.warning("File classification_report_final.xlsx tidak ditemukan.")

# =============================
# SECTION 4: RINGKASAN METRIK
# =============================
st.header("🧮 Ringkasan Metrik Performa")

if file_exists(PATHS["metrics_summary"]):
    with open(PATHS["metrics_summary"], "r", encoding="utf-8") as f:
        metrics_text = f.read()
    st.code(metrics_text, language="text")
else:
    st.warning("File final_metrics_summary.txt tidak ditemukan.")

# =============================
# SECTION 5: PERBANDINGAN SKENARIO
# =============================
st.header("⚖️ Perbandingan Skenario Eksperimen")

if file_exists(PATHS["perbandingan"]):
    df_compare = pd.read_excel(PATHS["perbandingan"])
    st.dataframe(df_compare, use_container_width=True)

    st.download_button(
        "⬇️ Unduh Perbandingan (Excel)",
        data=open(PATHS["perbandingan"], "rb"),
        file_name="perbandingan_skenario_skripsi.xlsx"
    )
else:
    st.warning("File perbandingan_skenario_skripsi.xlsx tidak ditemukan.")

# =============================
# SECTION 6: LOG EKSPERIMEN (OPSIONAL)
# =============================
st.header("🗂️ Log Proses Eksperimen")

if file_exists(PATHS["log_harian"]):
    df_log = pd.read_csv(
    PATHS["log_harian"],
    engine="python",
    sep=",",
    on_bad_lines="skip"
)
    st.dataframe(df_log, use_container_width=True)
else:
    st.info("Log harian tidak tersedia atau tidak disertakan.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Model Terbaik: Word2Vec (Skip-gram) + LSTM | Multi-Label Powerset | Skripsi Analisis Sentimen BBM Pertamina"
)