import streamlit as st
import pandas as pd
from sentiment_model import predict_sentiment
from topic_model import predict_topic
from utils import get_text_from_url

st.set_page_config(
    page_title="Analisis Teks",
    layout="wide"
)

# ===============================
# SIDEBAR NAVIGASI
# ===============================
menu = st.sidebar.radio(
    "Navigasi",
    [
        "Prediksi Topik",
        "Analisis Sentimen",
        "Akurasi Model",
        "Informasi Model"
    ]
)

# ===============================
# HALAMAN 1: PREDIKSI TOPIK
# ===============================
if menu == "Prediksi Topik":
    st.title("📌 Prediksi Topik Artikel")

    input_type = st.radio(
        "Pilih metode input",
        ["Input Teks", "Input URL"]
    )

    text = ""

    if input_type == "Input Teks":
        text = st.text_area("Masukkan teks artikel")

    elif input_type == "Input URL":
        url = st.text_input("Masukkan URL artikel")
        if url:
            text = get_text_from_url(url)
            st.text_area("Isi artikel:", text, height=250)

    if st.button("Prediksi Topik"):
        if text.strip():
            topic = predict_topic(text)
            st.success(f"Hasil Prediksi Topik: **{topic}**")
        else:
            st.warning("Teks masih kosong")

# ===============================
# HALAMAN 2: ANALISIS SENTIMEN
# ===============================
elif menu == "Analisis Sentimen":
    st.title("😊😡 Analisis Sentimen Komentar")

    input_type = st.radio(
        "Pilih metode input",
        ["Input Teks", "Unggah CSV"]
    )

    if input_type == "Input Teks":
        text = st.text_area("Masukkan komentar")

        if st.button("Analisis Sentimen"):
            result = predict_sentiment(text)
            st.success(f"Sentimen Komentar: **{result}**")

    elif input_type == "Unggah CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            col = st.selectbox("Pilih kolom komentar", df.columns)

            if st.button("Analisis"):
                df = df.dropna(subset=[col])
                df[col] = df[col].astype(str)
                df["Sentimen"] = df[col].apply(predict_sentiment)
                st.dataframe(df)

# ===============================
# HALAMAN 3: AKURASI MODEL
# ===============================
elif menu == "Akurasi Model":
    st.title("📊 Akurasi Model")

    try:
        df_acc = pd.read_csv("hasil_akurasi_model.csv")
        df_acc["Akurasi (%)"] = df_acc["Akurasi"] * 100

        st.dataframe(df_acc)
        st.bar_chart(df_acc.set_index("Model")["Akurasi (%)"])
    except:
        st.warning("File hasil_akurasi_model.csv belum tersedia.")

# ===============================
# HALAMAN 4: INFORMASI MODEL
# ===============================
elif menu == "Informasi Model":
    st.title("ℹ️ Informasi Model")

    st.markdown("""
    ### 📌 Analisis Sentimen
    - Output: **Positif / Negatif**
    - Menggunakan model Machine Learning
    - Preprocessing: cleaning, stemming, emoji removal

    ### 📌 Prediksi Topik
    - Algoritma: **Latent Dirichlet Allocation (LDA)**
    - Output: Topik artikel

    ### 📌 Input Data
    - Teks langsung
    - URL artikel
    - File CSV

    ### 📌 Framework
    - Streamlit
    - Scikit-learn
    - Sastrawi
    """)
