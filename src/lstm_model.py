from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

def build_lstm_model():

    model = Sequential([
        Input(shape=(100,)),

        Embedding(
            input_dim=10000,
            output_dim=128
        ),

        GlobalAveragePooling1D(),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model