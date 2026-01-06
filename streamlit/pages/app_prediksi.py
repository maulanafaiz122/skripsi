import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# ==================================================
# PAGE CONFIG (WAJIB SATU KALI & PALING ATAS)
# ==================================================
st.set_page_config(
    page_title="Prediksi Sentimen Multi-Label W2V-LSTM",
    layout="wide"
)

# ==================================================
# KONFIGURASI PATH ARTEFAK
# ==================================================
ARTIFACT_DIR = "artefak"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_lstm_model.keras")
LE_POWERSET_PATH = os.path.join(ARTIFACT_DIR, "le_powerset.pkl")
WORD_TO_INDEX_PATH = os.path.join(ARTIFACT_DIR, "word_to_index.pkl")
KETERANGAN_KELAS_PATH = os.path.join(ARTIFACT_DIR, "keterangan_tiap_kelas.xlsx")
MAX_LEN = 60

# ==================================================
# LOAD ARTEFAK (LAZY + CACHED)
# ==================================================
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    with open(LE_POWERSET_PATH, "rb") as f:
        le_powerset = pickle.load(f)
    with open(WORD_TO_INDEX_PATH, "rb") as f:
        word_to_index = pickle.load(f)
    df_keterangan = pd.read_excel(KETERANGAN_KELAS_PATH)
    return model, le_powerset, word_to_index, df_keterangan

model = None
le_powerset = None
word_to_index = None
df_keterangan = None

# ==================================================
# PREPROCESS & PREDICT
# ==================================================
def preprocess_and_predict(text, word_to_index, model, le_powerset, df_keterangan, max_len):

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text_to_word_sequence(text)
    encoded = [word_to_index.get(w, 0) for w in tokens]
    X = pad_sequences([encoded], maxlen=max_len, padding="post")

    probs = model.predict(X, verbose=0)[0]
    pred_id = np.argmax(probs)

    kode = le_powerset.inverse_transform([pred_id])[0]
    detail = df_keterangan.loc[df_keterangan["Kelas_ID"] == pred_id, "Detail"].iloc[0]

    top_idx = np.argsort(probs)[::-1][:5]
    top_data = []

    for i in top_idx:
        top_data.append({
            "Kelas ID": i,
            "Probabilitas": f"{probs[i]*100:.2f}%",
            "Detail Aspek": df_keterangan.loc[df_keterangan["Kelas_ID"] == i, "Detail"].iloc[0]
        })

    return {
        "pred_class_id": pred_id,
        "confidence": probs[pred_id],
        "kode": kode,
        "detail": detail,
        "top": top_data
    }

# ==================================================
# UI
# ==================================================
st.title("Analisis Sentimen Multi-Label BBM Pertamina")
st.markdown(
    "Model **Word2Vec + LSTM** untuk memprediksi **15 kelas Powerset** "
    "berdasarkan 4 aspek sentimen."
)

input_text = st.text_area(
    "Masukkan Ulasan Pelanggan:",
    "isi pertamax oplosan bikin mesin jadi kasar"
)

if st.button("Prediksi Sentimen"):
    if not input_text.strip():
        st.warning("Teks ulasan tidak boleh kosong.")
        st.stop()

    with st.spinner("Memuat model dan memprediksi..."):
        if model is None:
            model, le_powerset, word_to_index, df_keterangan = load_artifacts()

        result = preprocess_and_predict(
            input_text,
            word_to_index,
            model,
            le_powerset,
            df_keterangan,
            MAX_LEN
        )

    st.subheader("Hasil Prediksi Utama")
    st.metric(
        "Kelas Powerset",
        result["pred_class_id"],
        f"{result['confidence']*100:.2f}%"
    )

    st.info(f"""
    **Kode Biner:** {result['kode']}  
    **Detail Aspek:** {result['detail']}
    """)

    st.subheader("Top 5 Probabilitas Kelas")
    st.dataframe(pd.DataFrame(result["top"]), hide_index=True)

# ==================================================
# SIDEBAR / INFO TAMBAHAN
# ==================================================
if df_keterangan is not None:
    st.subheader("Keterangan 15 Kelas Powerset")
    st.dataframe(df_keterangan, height=400, hide_index=True)
