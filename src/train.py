from preprocessing import get_data
from lstm_model import build_lstm_model
from visualize import plot_history
import numpy as np

# Load Data
X_train, X_test, X_val, y_train, y_test, y_val, tokenizer, encoder = get_data()

print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)

print("\nClass Distribution:")
print(np.bincount(y_train))

# Build Model
model = build_lstm_model()

model.summary()

# Train Model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Plot Graphs
plot_history(history)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")

# Save Model
model.save("models/emotion_model.keras")

print("\nModel Saved Successfully!")