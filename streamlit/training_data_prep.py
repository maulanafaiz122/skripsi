import pandas as pd
import numpy as np
import re
import os
import pickle
import itertools
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from skmultilearn.problem_transform import LabelPowerset

# --- KONFIGURASI PATH ---
# Path input data Anda (Ganti dengan path lokal yang benar!)
# Asumsi Anda sudah memiliki file 07_stemming.xlsx
DATA_INPUT_PATH = "07_stemming.xlsx - Sheet1.csv" # GANTI jika nama file berbeda
ARTIFACT_DIR = "artefak"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MAX_LEN = 60 # Panjang sequence untuk padding
W2V_DIM = 100 # Dimensi Word2Vec terbaik

# --- LOAD DATA DAN PREP FUNGSI ---
try:
    # Memuat data. Jika file Anda besar, gunakan chunking.
    data = pd.read_csv(DATA_INPUT_PATH) 
except FileNotFoundError:
    print(f"ERROR: File data input tidak ditemukan di: {DATA_INPUT_PATH}")
    exit()

# 1. Pastikan kolom 'stemming' berupa list of strings/tokens
if isinstance(data['stemming'].iloc[0], str):
    # Membersihkan string list dan mengubahnya menjadi list
    data['stemming'] = data['stemming'].apply(lambda x: re.sub(r"[[\]']", "", x).split(', '))
    data['stemming'] = data['stemming'].apply(lambda x: [item.strip() for item in x])


# 2. ONE-HOT ENCODING (4 ASPEK)
aspect_to_index = {
    "kualitas bahan bakar": 0,
    "ketidaksesuaian produk": 1,
    "kesalahan pengisian bbm": 2,
    "kepercayaan terhadap pertamina": 3
}

def get_onehot_multilabel(text, mapping):
    onehot = [0] * len(mapping)
    if pd.isna(text):
        return onehot
    text = str(text).lower()
    for label, idx in mapping.items():
        if label in text:
            onehot[idx] = 1
    return onehot

data['aspek_onehot'] = data['aspek'].apply(lambda x: get_onehot_multilabel(x, aspect_to_index))


# 3. LABEL POWERSET TRANSFORMATION (15 KELAS)
data['powerset_label_str'] = data['aspek_onehot'].apply(lambda x: "".join(map(str, x)))
le_powerset = LabelEncoder()
data['powerset_class_id'] = le_powerset.fit_transform(data['powerset_label_str'])
num_classes = len(le_powerset.classes_)

# Simpan Keterangan Kelas Powerset
aspek_list = data['aspek'].str.split(', ').explode().unique()
nama_aspek = sorted(list([a.strip() for a in aspek_list if pd.notna(a)]))

df_detail = pd.DataFrame([
    {'Kelas_ID': i, 'Kode_Biner': c, 'Detail': ", ".join([nama_aspek[idx] for idx, char in enumerate(c) if char == '1'])}
    for i, c in enumerate(le_powerset.classes_)
])
df_detail.to_excel(os.path.join(ARTIFACT_DIR, "keterangan_tiap_kelas.xlsx"), index=False)
print(f"Keterangan kelas Powerset disimpan di {ARTIFACT_DIR}/keterangan_tiap_kelas.xlsx")


# 4. SPLITTING DATA (80:20 Stratified)
X = data['stemming']
y = data['powerset_class_id']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Data split: Train {len(X_train)}, Test {len(X_test)}")


# 5. WORD2VEC MODELING (Menggunakan dimensi yang ditentukan)
sentences = X_train.tolist()
print(f"Melatih model Word2Vec Skip-gram (Dimensi: {W2V_DIM})...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=W2V_DIM, 
    window=5,
    min_count=1,
    sg=1, # Skip-gram
    workers=4,
    seed=42
)
w2v_model.save(os.path.join(ARTIFACT_DIR, "word2vec_final.model"))


# 6. PEMBUATAN EMBEDDING MATRIX & PADDING
words = list(w2v_model.wv.index_to_key)
word_to_index = {word: i + 1 for i, word in enumerate(words)} # Index 0 untuk padding

# Simpan Word to Index
with open(os.path.join(ARTIFACT_DIR, 'word_to_index.pkl'), 'wb') as f:
    pickle.dump(word_to_index, f)

# Buat Embedding Matrix
vocab_size = len(word_to_index) + 1
embedding_matrix = np.zeros((vocab_size, W2V_DIM))
for word, i in word_to_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(f"Embedding Matrix bentuk: {embedding_matrix.shape}")

# Padding Data Training
X_train_pad = pad_sequences([[word_to_index.get(w, 0) for w in s] for s in X_train], maxlen=MAX_LEN, padding='post')
y_train_cat = to_categorical(y_train, num_classes=num_classes)


# 7. TRAINING MODEL LSTM (Menggunakan setelan terbaik dari Grid Search Anda)
# Kami akan menggunakan salah satu kombinasi terbaik dari log Anda (misalnya: Dim 100, unit 64, Adam 0.001)
BEST_LSTM_UNIT = 64
BEST_OPTIMIZER = Adam(learning_rate=0.001)
EPOCHS = 15
BATCH_SIZE = 128

# Hitung Class Weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
dict_weights = dict(enumerate(weights))

print(f"\nMemulai Training Model LSTM...")


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=W2V_DIM, weights=[embedding_matrix],
              input_length=MAX_LEN, trainable=True),
    SpatialDropout1D(0.2),
    LSTM(BEST_LSTM_UNIT, dropout=0.5), 
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=BEST_OPTIMIZER, metrics=['accuracy'])

model.fit(X_train_pad, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
          class_weight=dict_weights, verbose=1)

# 8. SIMPAN MODEL TERBAIK & ARTEFAK LAIN
model.save(os.path.join(ARTIFACT_DIR, "best_lstm_model.keras"))
with open(os.path.join(ARTIFACT_DIR, 'le_powerset.pkl'), 'wb') as f:
    pickle.dump(le_powerset, f)
    
print("\n--- PROSES PREPARASI SELESAI ---")
print(f"Artefak model berhasil disimpan di folder: {ARTIFACT_DIR}/")
print("Anda sekarang dapat menjalankan `app.py` menggunakan Streamlit.")
