import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    return  1 / (1 + np.exp(-np.array(x)))