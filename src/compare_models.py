from preprocessing import get_data
from tensorflow.keras.models import load_model

# Load Data
X_train, X_test, X_val, y_train, y_test, y_val, tokenizer, encoder = get_data()

# Load Models
dense_model = load_model("models/emotion_model.keras")
lstm_model = load_model("models/lstm_model.keras")
gru_model = load_model("models/gru_model.keras")

# Evaluate Models
dense_loss, dense_acc = dense_model.evaluate(X_test, y_test, verbose=0)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
gru_loss, gru_acc = gru_model.evaluate(X_test, y_test, verbose=0)

print("\nMODEL COMPARISON")
print("-" * 40)

print(f"Dense Accuracy : {dense_acc:.4f}")
print(f"LSTM Accuracy  : {lstm_acc:.4f}")
print(f"GRU Accuracy   : {gru_acc:.4f}")

best_acc = max(dense_acc, lstm_acc, gru_acc)

if best_acc == dense_acc:
    print("\nBest Model : Dense")
elif best_acc == lstm_acc:
    print("\nBest Model : LSTM")
else:
    print("\nBest Model : GRU")