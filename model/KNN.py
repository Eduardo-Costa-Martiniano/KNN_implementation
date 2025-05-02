import numpy as np
import mnist as data
import sys

def predict_knn(X_train, Y_train, X_validation, Y_validation=None, k=3):
    predictions = []
    for i, x in enumerate(X_validation):
        distances = np.linalg.norm(X_train - x, axis=1)
        k_indices = distances.argsort()[:k]
        k_nearest_labels = Y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
        if Y_validation is not None:
            print(f"Sample {i:03}: Predicted = {most_common}, Ground Truth", 
            f"= {Y_validation[i]}, Correct = {most_common == Y_validation[i]}")    
    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    if len(sys.argv)>1:
        k = int(sys.argv[1])
        Y_val_pred = predict_knn(data.X_train, data.Y_train, \
        data.X_validation, data.Y_validation, k)
    else:
        Y_val_pred = predict_knn(data.X_train, data.Y_train, \
        data.X_validation, data.Y_validation)
    val_accuracy = accuracy(data.Y_validation, Y_val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
