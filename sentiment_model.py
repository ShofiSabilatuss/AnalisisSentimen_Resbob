import re
import pickle
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return stemmer.stem(text)

def predict_sentiment(text):
    clean_text = preprocess(text)

    if clean_text == "":
        return "Netral"

    vector = tfidf.transform([clean_text])
    pred = model.predict(vector)[0]

    if pred == 1:
        return "Positif"
    elif pred == 0:
        return "Negatif"
    else:
        return "Netral"
