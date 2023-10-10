import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Judul aplikasi
st.title("Deteksi Sentimen")

# Input teks dari pengguna
input_text = st.text_area("Masukkan teks:", "")

# Load model dari file pickle
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load TF-IDF Vectorizer dari file pickle
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Fungsi untuk menghitung skor sentimen
def predict_sentiment(text):
    # Transformasi teks input ke dalam representasi TF-IDF
    tfidf_text = tfidf_vectorizer.transform([text])

    # Prediksi skor sentimen
    sentiment_score = model.predict_proba(tfidf_text)[0]

    return sentiment_score

# Tampilkan skor sentimen jika ada input teks dari pengguna
if input_text:
    sentiment_score = predict_sentiment(input_text)

    # Tampilkan skor sentimen
    st.write("Skor Sentimen:")
    st.write(f"Positif: {sentiment_score[1]:.2f}")
    st.write(f"Negatif: {sentiment_score[0]:.2f}")
