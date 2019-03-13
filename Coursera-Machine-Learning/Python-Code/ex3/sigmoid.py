import numpy as np

def sigmoid(z):
    # compute the sigmoid of z
    g = np.zeros((z.shape))
    
    # Compute the sigmoid of each value of z (z can be a matrix,
    # vector of scalar)
    denominator = 1 + np.exp(-1 * z)
    g = denominator**(-1)
    
    return g