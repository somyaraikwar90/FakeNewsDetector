import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PIL import Image

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Function: Preprocess Input
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Page Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# Add Banner or Logo
st.image("C:/Users/SOMYA/Downloads/ChatGPT Image May 10, 2025, 05_41_23 PM.png", use_container_width=True)  

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #0072B2;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .stButton button:hover {
        background-color: #005f8a;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>🧠 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Classify news as Real or Fake using NLP and ML</h4>", unsafe_allow_html=True)

# Text Input
st.markdown("### 📌 Paste or type the news content below:")
user_input = st.text_area("News Content", height=200)

# Predict
if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned_input = preprocess(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        st.markdown("---")
        if prediction == 1:
            st.success("✅ This appears to be **REAL news**.")
            st.balloons()
        else:
            st.error("❌ This appears to be **FAKE news**.")
            st.snow()

# Footer
st.markdown("""
<hr>
<small style='color: gray;'>📚 Developed by Somya • A College Major Project</small>
""", unsafe_allow_html=True)
