import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Judul aplikasi
st.title("Deteksi Kalimat Abusive")

# Input teks dari pengguna
input_text = st.text_area("Masukkan teks:", "")

# Tombol "Proses" untuk menghitung skor abusiveness
if st.button("Proses"):
    # Load model dari file pickle
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Load TF-IDF Vectorizer dari file pickle
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)

    # Fungsi untuk menghitung skor abusiveness
    def predict_abusiveness(text):
        # Transformasi teks input ke dalam representasi TF-IDF
        tfidf_text = tfidf_vectorizer.transform([text])

        # Prediksi skor abusiveness
        abusiveness_score = model.predict_proba(tfidf_text)[0]

        return abusiveness_score

    # Tampilkan skor abusiveness jika ada input teks dari pengguna
    if input_text:
        abusiveness_score = predict_abusiveness(input_text)

        # Tampilkan skor sentimen
        st.write("Skor abusiveness:")
        st.write(f"Positif: {abusiveness_score[0]:.2f}")
        st.write(f"Negatif: {abusiveness_score[1]:.2f}")

        # Tambahkan peringatan berdasarkan skor abusiveness
        if abusiveness_score[0] > abusiveness_score[1]:
            st.success("Kalimat ini TIDAK ABUSIVE.")
        elif abusiveness_score[0] < abusiveness_score[1]:
            st.warning("Kalimat ini ABUSIVE.")
        else:
            st.info("Kalimat ini NETRAL.")
