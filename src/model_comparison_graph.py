import matplotlib.pyplot as plt

models = ["Dense", "LSTM", "GRU"]
accuracies = [86.35, 29.05, 34.75]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha="center")

plt.savefig("graphs/model_comparison.png")
plt.show()