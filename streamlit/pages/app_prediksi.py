import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# ======================================================
# KONFIGURASI HALAMAN (HANYA SEKALI!)
# ======================================================
st.set_page_config(
    page_title="Prediksi Sentimen Multi-Label",
    layout="wide"
)

st.title("📊 Analisis Sentimen Multi-Label BBM Pertamina")
st.markdown(
    "Model **Word2Vec + LSTM** untuk klasifikasi sentimen "
    "berbasis **Label Powerset (15 kelas)**."
)

# ======================================================
# PATH ARTEFAK
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artefak")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_lstm_model.keras")
LE_POWERSET_PATH = os.path.join(ARTIFACT_DIR, "le_powerset.pkl")
WORD_TO_INDEX_PATH = os.path.join(ARTIFACT_DIR, "word_to_index.pkl")
KETERANGAN_KELAS_PATH = os.path.join(ARTIFACT_DIR, "keterangan_tiap_kelas.xlsx")

MAX_LEN = 60


# ======================================================
# LOAD ARTEFAK (AMAN + JELAS ERROR-NYA)
# ======================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(f"Gagal load model LSTM: {e}")

    try:
        with open(LE_POWERSET_PATH, "rb") as f:
            le_powerset = pickle.load(f)

        with open(WORD_TO_INDEX_PATH, "rb") as f:
            word_to_index = pickle.load(f)

        df_keterangan = pd.read_excel(KETERANGAN_KELAS_PATH)

    except Exception as e:
        raise RuntimeError(f"Gagal load artefak pendukung: {e}")

    return model, le_powerset, word_to_index, df_keterangan


# ======================================================
# FUNGSI PREDIKSI
# ======================================================
def preprocess_and_predict(
    text,
    model,
    word_to_index,
    le_powerset,
    df_keterangan,
    max_len
):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text_to_word_sequence(text)
    encoded = [word_to_index.get(tok, 0) for tok in tokens]

    X = pad_sequences([encoded], maxlen=max_len, padding="post")

    probs = model.predict(X, verbose=0)[0]
    pred_id = int(np.argmax(probs))

    powerset_label = le_powerset.inverse_transform([pred_id])[0]
    detail = df_keterangan.loc[
        df_keterangan["Kelas_ID"] == pred_id, "Detail"
    ].values[0]

    top_idx = np.argsort(probs)[::-1][:5]

    top_preds = []
    for i in top_idx:
        top_preds.append({
            "Kelas ID": int(i),
            "Probabilitas (%)": round(float(probs[i]) * 100, 2),
            "Detail": df_keterangan.loc[
                df_keterangan["Kelas_ID"] == i, "Detail"
            ].values[0]
        })

    return {
        "pred_id": pred_id,
        "confidence": probs[pred_id],
        "powerset": powerset_label,
        "detail": detail,
        "top": top_preds
    }


# ======================================================
# UI INPUT
# ======================================================
input_text = st.text_area(
    "Masukkan ulasan pelanggan:",
    "isi pertamax oplosan bikin mesin jadi kasar padahal beli non subsidi",
    height=120
)

# ======================================================
# TOMBOL PREDIKSI (MODEL BARU LOAD DI SINI)
# ======================================================
if st.button("🔍 Prediksi Sentimen"):
    if not input_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
        st.stop()

    with st.spinner("Memuat model & memproses prediksi..."):
        try:
            model, le_powerset, word_to_index, df_keterangan = load_artifacts()
        except Exception as e:
            st.error(str(e))
            st.stop()

        result = preprocess_and_predict(
            input_text,
            model,
            word_to_index,
            le_powerset,
            df_keterangan,
            MAX_LEN
        )

    st.success("Prediksi berhasil")

    st.subheader("📌 Hasil Utama")
    st.metric(
        "Kelas Powerset Terprediksi",
        result["pred_id"],
        f"{result['confidence']*100:.2f}% confidence"
    )

    st.info(
        f"**Kode Biner:** {result['powerset']}\n\n"
        f"**Detail Aspek:** {result['detail']}"
    )

    st.subheader("📈 Top 5 Probabilitas Kelas")
    st.dataframe(pd.DataFrame(result["top"]), use_container_width=True)


# ======================================================
# INFO KELAS
# ======================================================
st.markdown("---")
st.subheader("📚 Keterangan 15 Kelas Powerset")

try:
    _, _, _, df_keterangan = load_artifacts()
    st.dataframe(df_keterangan, height=400, use_container_width=True)
except Exception:
    st.info("Keterangan kelas akan muncul setelah artefak berhasil dimuat.")
