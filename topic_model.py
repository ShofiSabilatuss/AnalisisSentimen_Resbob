import re
import pickle

# ===============================
# LOAD MODEL LDA
# ===============================
lda_model = pickle.load(open("lda_model.pkl", "rb"))
vectorizer = pickle.load(open("lda_vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_topic(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Tidak diketahui"

    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    topic_index = lda_model.transform(vector).argmax()

    return f"Topik {topic_index + 1}"
