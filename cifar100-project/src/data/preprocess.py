
import numpy as np

def normalize(X):
    return X / 255.0


def reshape_images(X):
    # (N, 3072) → (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32)
    X = np.transpose(X, (0, 2, 3, 1))
    return X