import requests
from bs4 import BeautifulSoup
import re

def get_text_from_url(url):
    """
    Mengambil teks artikel dari sebuah URL
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Ambil semua paragraf
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        # Cleaning dasar
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    except Exception as e:
        return ""
