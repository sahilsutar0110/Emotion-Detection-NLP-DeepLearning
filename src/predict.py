from tensorflow.keras.models import load_model
from preprocessing import tokenizer, encoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load trained model
model = load_model("models/emotion_model.keras")

print("Emotion Detection System")
print("Type 'exit' to quit\n")

while True:

    text = input("Enter Text: ")

    if text.lower() == "exit":
        break

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding="post"
    )

    # Predict
    prediction = model.predict(padded, verbose=0)

    predicted_class = np.argmax(prediction)

    emotion = encoder.inverse_transform([predicted_class])

    print("Predicted Emotion:", emotion[0])
    print()