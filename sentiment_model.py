import re
import pickle
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


stemmer = StemmerFactory().create_stemmer()

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def preprocess(text):