import re
import pickle
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===============================
# LOAD STEMMER
# ===============================
stemmer = StemmerFactory().create_stemmer()

# ===============================
# LOAD MODEL & TF-IDF
# ===============================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ===============================
# PREPROCESSING
# ===============================
def preprocess(text):
    # Antisipasi NaN / bukan string
    if not isinstance(text, str):
        return ""

    # Hapus emoji & stiker
    text = emoji.replace_emoji(text, replace="")

    # Lowercase
    text = text.lower()

    # Hapus simbol & angka
    text = re.sub(r"[^a-z\s]", " ", text)

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # Stemming Bahasa Indonesia
    return stemmer.stem(text)

# ===============================
# PREDIKSI SENTIMEN
# ===============================
def predict_sentiment(text):
    clean_text = preprocess(text)

    # Jika kosong setelah preprocessing
    if clean_text == "":
        return "Netral"

    vector = tfidf.transform([clean_text])
    pred = model.predict(vector)[0]

    # Mapping label (WAJIB JELAS)
    if pred == 1:
        return "Positif"
    elif pred == 0:
        return "Negatif"
    else:
        return "Netral"
