import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, "vectorizer.pkl")
model_path = os.path.join(base_path, "model.pkl")

tfidf = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title(" SMS Spam Classifier")
st.write("Detect whether a message is Spam or Not Spam using Machine Learning")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
