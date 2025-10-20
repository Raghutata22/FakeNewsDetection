import streamlit as st
import nltk
from nltk.corpus import stopwords     
import re
import pickle
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove punctuation   
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords (optional but recommended)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Fake News Detection")
st.write("Enter news text to check if it's fake or real.")

user_input = st.text_area("News Text")

if st.button("Check"):
    if user_input.strip() == '':
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        prediction = model.predict([cleaned_text])[0]
        if prediction == 1:
            st.success("The news is **REAL**.")
        else:
            st.error("The news is **FAKE**.")


