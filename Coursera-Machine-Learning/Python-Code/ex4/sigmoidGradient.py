import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    
    g = sigmoid(z) * (1 - sigmoid(z))
    
    return g