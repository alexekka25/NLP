import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib  # If needed for tokenizer

# Load the tokenizer and models
tokenizer = joblib.load("tokenizer.pkl")  # Make sure to save your tokenizer earlier

models = {
    "LSTM": load_model("LSTM_model.h5"),
    "GRU": load_model("GRU_model.h5"),
    "Bidirectional LSTM": load_model("Bidirectional_model.h5")
}

# Streamlit app
st.title("Sentiment Analysis Web App")

st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Choose a model", ["LSTM", "GRU", "Bidirectional LSTM"])

st.header("Enter Your Text")
user_input = st.text_area("Type your review here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess the user input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        
        # Get the selected model
        model = models[model_choice]
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        # Display result
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {prediction[0][0]:.2f}")
