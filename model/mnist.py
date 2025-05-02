import numpy as np
import gzip
import struct

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))      
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows) / 255.0

X_train = load_images("../data/mnist/train-images-idx3-ubyte.gz")
X_test_all = load_images("../data/mnist/t10k-images-idx3-ubyte.gz")
X_validation, X_test = np.split(X_test_all, 2)

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8)
    
Y_train = load_labels("../data/mnist/train-labels-idx1-ubyte.gz")                                  
Y_test_all = load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz")
Y_validation, Y_test = np.split(Y_test_all, 2)
