import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import tokenizer, encoder

# Load Model
model = load_model("models/emotion_model.keras")

# Page Title
st.title("😊 Emotion Detection App")

st.write("Enter a sentence and detect the emotion.")

# Text Input
user_text = st.text_area(
    "Enter Text",
    height=150
)

# Predict Button
if st.button("Predict Emotion"):

    if user_text.strip() != "":

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_text])

        # Padding
        padded = pad_sequences(
            sequence,
            maxlen=100,
            padding="post"
        )

        # Prediction
        prediction = model.predict(
            padded,
            verbose=0
        )

        predicted_class = np.argmax(prediction)

        emotion = encoder.inverse_transform(
            [predicted_class]
        )

        st.success(
            f"Predicted Emotion: {emotion[0]}"
        )

    else:
        st.warning("Please enter some text.")