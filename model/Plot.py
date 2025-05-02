import matplotlib.pyplot as plt
import numpy as np
import mnist as data
from KNN import predict_knn, accuracy  

k_values = [k for k in range(1,11)]
accuracies = []

for k in k_values:
    print(f"Evaluating k = {k}")
    Y_pred = predict_knn(data.X_train, data.Y_train, data.X_validation, k=k)
    acc = accuracy(data.Y_validation, Y_pred)
    accuracies.append(acc)
    print(f"Accuracy: {acc * 100:.2f}%\n")

plt.figure(figsize=(8, 5))
plt.bar([str(k) for k in k_values], [a * 100 for a in accuracies], color='skyblue')
plt.title('KNN Accuracy for Different k Values')
plt.xlabel('k')
plt.ylabel('Validation Accuracy (%)')
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
