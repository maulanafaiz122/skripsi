import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import os

# ================================
# SIDEBAR NAVIGASI HALAMAN
# ================================
st.set_page_config(
    page_title="Hasil Penelitian",
    layout="wide"
)



# --- KONFIGURASI PATH ARTEFAK ---
ARTIFACT_DIR = "artefak"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_lstm_model.keras")
LE_POWERSET_PATH = os.path.join(ARTIFACT_DIR, "le_powerset.pkl")
WORD_TO_INDEX_PATH = os.path.join(ARTIFACT_DIR, "word_to_index.pkl")
KETERANGAN_KELAS_PATH = os.path.join(ARTIFACT_DIR, "keterangan_tiap_kelas.xlsx")
MAX_LEN = 60 # Sesuai dengan panjang padding saat training

# --- FUNGSI UTAMA LOAD ARTEFAK ---
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)  # PENTING
    with open(LE_POWERSET_PATH, 'rb') as f:
        le_powerset = pickle.load(f)
    with open(WORD_TO_INDEX_PATH, 'rb') as f:
        word_to_index = pickle.load(f)
    df_keterangan = pd.read_excel(KETERANGAN_KELAS_PATH)
    return model, le_powerset, word_to_index, df_keterangan

model = None
le_powerset = None
word_to_index = None
df_keterangan = None

# --- FUNGSI PREPROCESSING DAN PREDIKSI ---
def preprocess_and_predict(text, word_to_index, model, le_powerset, df_keterangan, max_len):
    
    # PERBAIKAN: Mengganti all([df]) dengan pemeriksaan is None/empty secara eksplisit
    if model is None or le_powerset is None or word_to_index is None or df_keterangan is None:
        return {"error": "Artefak model tidak berhasil dimuat (hasil load adalah None)."}
    
    if df_keterangan.empty:
        return {"error": "DataFrame keterangan kelas kosong."}

    # Asumsi: Stemming yang diterapkan saat training adalah Simple Tokenization
    # Dalam skenario riil, Anda harus MENGGUNAKAN FUNGSI STEMMING YANG SAMA PERSIS 
    # (misalnya Sastrawi) seperti pada tahap preprocessing data.
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    # Menggunakan text_to_word_sequence dari Keras untuk tokenisasi sederhana
    tokens = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    
    # Sequencing
    encoded_seq = [word_to_index.get(word, 0) for word in tokens]
    
    # Padding
    X_pred = pad_sequences([encoded_seq], maxlen=max_len, padding='post')
    
    # Prediksi
    y_pred_prob = model.predict(X_pred, verbose=0)[0]
    pred_class_id = np.argmax(y_pred_prob)
    
    # Inverse Transform dan Detail Aspek
    # Menggunakan .iloc[0] aman karena kita mencari berdasarkan Kelas_ID yang unik
    pred_powerset_label_str = le_powerset.inverse_transform([pred_class_id])[0]
    detail_aspek = df_keterangan[df_keterangan['Kelas_ID'] == pred_class_id]['Detail'].iloc[0]
    
    # Dapatkan probabilitas 5 kelas teratas
    top_5_indices = np.argsort(y_pred_prob)[::-1][:5]
    top_5_probs = y_pred_prob[top_5_indices]
    
    top_10_details = []
    for idx, prob in zip(top_5_indices, top_5_probs):
        detail = df_keterangan[df_keterangan['Kelas_ID'] == idx]['Detail'].iloc[0]
        top_10_details.append({
            'Kelas ID': idx,
            'Probabilitas': f"{prob*100:.2f}%",
            'Detail Aspek': detail
        })
        
    result = {
        "pred_class_id": pred_class_id,
        "kode_biner_powerset": pred_powerset_label_str,
        "detail_aspek": detail_aspek,
        "confidence_score": y_pred_prob[pred_class_id],
        "top_10_predictions": top_10_details
    }
    
    return result

# --- INTERFACE STREAMLIT ---
st.set_page_config(
    page_title="Prediksi Sentimen Multi-Label W2V-LSTM",
    layout="wide"
)
st.set_page_config(page_title="Prediksi Sentimen Multi-Label W2V-LSTM")
st.title(" Analisis Sentimen Multi-Label BBM Pertamina (Word2Vec + LSTM)")
st.markdown("Aplikasi ini memprediksi kelas sentimen Powerset (15 kelas) dari ulasan pelanggan berdasarkan 4 aspek utama: Kualitas, Ketidaksesuaian, Kesalahan Pengisian, dan Kepercayaan.")

input_text = st.text_area("Masukkan Ulasan Pelanggan (Sudah Distemming, jika bisa):", 
                          "isi pertamax oplosan bikin mesin jadi kasar padahal beli non subsidi")

if st.button("Prediksi Sentimen"):
    if not input_text:
        st.warning("Mohon masukkan teks ulasan untuk diprediksi.")
        st.stop()

    with st.spinner("Memuat model & memproses prediksi..."):
        if model is None:
            try:
                model, le_powerset, word_to_index, df_keterangan = load_artifacts()
            except Exception as e:
                st.error(f"Gagal memuat model: {e}")
                st.stop()

        prediction_result = preprocess_and_predict(
            input_text,
            word_to_index,
            model,
            le_powerset,
            df_keterangan,
            MAX_LEN
        )
                
                # Menampilkan Hasil Utama
                st.subheader(" Hasil Prediksi Utama")
                st.metric(label="Kelas Powerset Terprediksi (ID)", 
                          value=prediction_result['pred_class_id'], 
                          delta=f"{prediction_result['confidence_score']*100:.2f}% Confidence")
                
                st.info(f"""
                **Detail Aspek (Biner {prediction_result['kode_biner_powerset']})**:
                **{prediction_result['detail_aspek']}**
                """)

                # Menampilkan 5 Kelas Teratas
                st.subheader("Top 5 Probabilitas Kelas")
                df_top_10 = pd.DataFrame(prediction_result['top_10_predictions'])
                st.dataframe(df_top_10, hide_index=True)
                
                st.markdown("---")
                st.caption(f"Max Sequence Length: {MAX_LEN} | Model: LSTM | Vektor: Word2Vec")

# Optional: Tampilkan detail kelas di sidebar
st.subheader(" Keterangan 15 Kelas Powerset")
if df_keterangan is not None:
    st.dataframe(df_keterangan, height=400, hide_index=True)

