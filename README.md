## 🧠 Project Summary

This project implements the K-Nearest Neighbors (KNN) algorithm from scratch to recognize handwritten digits using the MNIST dataset.

- `mnist.py` loads and normalizes the image and label data so that each digit image is represented as a 784-length vector with pixel values scaled from 0 to 1.
- `KNN.py` classifies each validation image by calculating the distance to all training images, selecting the `k` nearest neighbors, and predicting the most common label among them.
- The distance calculation is vectorized using NumPy for faster performance.
- `Plot.py` visualizes how accuracy changes for different values of `k`.

The model reaches over 95% accuracy across most tested values of `k`.
