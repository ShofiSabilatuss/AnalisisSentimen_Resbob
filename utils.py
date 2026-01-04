import requests
from bs4 import BeautifulSoup
import re

def get_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

        return text

    except:
        return ""
