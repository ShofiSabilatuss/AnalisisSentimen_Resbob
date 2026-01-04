import pickle
import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

try:
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
except Exception as e:
    model = None
    tfidf = None

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return stemmer.stem(text)

def predict_sentiment(text):
    if model is None or tfidf is None:
        return "Model tidak tersedia"

    clean = preprocess(text)
    if clean == "":
        return "Netral"

    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    return pred.capitalize()
