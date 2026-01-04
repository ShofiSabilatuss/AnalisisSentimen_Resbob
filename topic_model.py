import re
import pickle

lda_model = pickle.load(open("lda_model.pkl", "rb"))
vectorizer = pickle.load(open("lda_vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def predict_topic(text):
    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    topic = lda_model.transform(vector).argmax()
    return topic + 1
