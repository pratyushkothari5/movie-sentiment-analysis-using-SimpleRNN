from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model_path = Path(__file__).resolve().parent / 'simple_rnn_model.h5'
model = load_model(model_path)

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
def preprocess_review(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding='pre')
    return padded_review

def predict_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below and click 'Predict' to see the sentiment.")
user_input = st.text_area("Enter a movie review:")
if st.button('classify'):
    preprocessed_review = preprocess_review(user_input)
    prediction = model.predict(preprocessed_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]}")
else:
    st.write("Please enter a review and click 'Predict' to see the sentiment.")
