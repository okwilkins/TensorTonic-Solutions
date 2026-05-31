import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    return np.maximum(np.array(x), 0.0)