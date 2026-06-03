from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

def build_gru_model():

    model = Sequential()

    model.add(
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_shape=(100,)
    )
)

    model.add(
        GRU(128)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(6, activation="softmax")
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.build(input_shape=(None, 100))

    return model
print("\nTraining GRU Model...")

