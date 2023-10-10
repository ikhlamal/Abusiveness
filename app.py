import streamlit as st
import pickle

# Load vectorizer and model from pickle files
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open('logistic_regression_model.pkl', 'rb') as model_file:
    logistic_regression = pickle.load(model_file)

# Streamlit UI
st.title('Abusiveness Detection')

# Input text from the user
user_input = st.text_input('Input a sentence:')
if user_input:
    # Preprocess and transform user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Predict the abusiveness level
    prediction = logistic_regression.predict(user_input_tfidf)[0]

    # Output the result
    st.write('Input Sentence:', user_input)
    if prediction == 0:
        st.write('Prediction: Non-Abusive')
    else:
        st.write('Prediction: Abusive')

st.write('Example:')
example_text = "Lu ganteng tapi mukanya kek anjing."
st.write('Input Sentence:', example_text)
example_text_tfidf = tfidf_vectorizer.transform([example_text])
example_prediction = logistic_regression.predict(example_text_tfidf)[0]
st.write('Prediction:', 'Abusive' if example_prediction == 1 else 'Non-Abusive')
