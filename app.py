import streamlit as st
import pandas as pd
from sentiment_model import predict_sentiment

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis Sentimen",
    layout="wide"
)

st.title("📊 Analisis Sentimen Komentar")
st.write("Upload file CSV dan pilih kolom komentar untuk dianalisis.")

# UPLOAD FILE CSV
file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if file is not None:

    df = pd.read_csv(file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    kolom_teks = st.selectbox(
        "Pilih kolom yang berisi komentar:",
        df.columns
    )

    if st.button("🔍 Analisis Sentimen"):

        # Cleaning dasar (anti error)
        df = df.dropna(subset=[kolom_teks])
        df[kolom_teks] = df[kolom_teks].astype(str)
        df[kolom_teks] = df[kolom_teks].str.strip()
        df = df[df[kolom_teks] != ""]

        # Prediksi sentimen
        df["sentimen"] = df[kolom_teks].apply(predict_sentiment)

        st.success("✅ Analisis sentimen selesai!")

        st.subheader("📊 Hasil Analisis")
        st.dataframe(df)

# AKURASI MODEL
st.subheader("📈 Akurasi Model")

try:
    df_acc = pd.read_csv("hasil_akurasi_model.csv")

    # ubah ke persen
    df_acc["Akurasi (%)"] = df_acc["Akurasi"] * 100

    st.dataframe(df_acc[["Model", "Akurasi (%)"]])

    # Grafik akurasi
    st.bar_chart(
        data=df_acc.set_index("Model")["Akurasi (%)"]
    )

except:
    st.warning("File hasil_akurasi_model.csv belum tersedia.")

# CONFUSION MATRIX
st.subheader("📊 Confusion Matrix Tiap Model")

models_cm = {
    "Naive Bayes": "cm_naive_bayes.png",
    "SVM": "cm_svm.png",
    "KNN": "cm_knn.png",
    "Decision Tree": "cm_dt.png",
    "Random Forest": "cm_rf.png",
    "Neural Network": "cm_nn.png"
}

for model, img in models_cm.items():
    st.markdown(f"**{model}**")
    try:
        st.image(img)
    except:
        st.warning(f"Confusion matrix untuk {model} belum tersedia.")

