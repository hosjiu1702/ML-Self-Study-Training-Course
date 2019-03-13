import numpy as np
from sigmoid import sigmoid

def gradient(theta, X, y, lam_bda):
    m = len(y)
    grad_J = np.zeros((theta.shape))

    h = sigmoid(X.dot(theta))

    grad_J = (1 / m) * X.T.dot(h-y) + (lam_bda / m) * theta

    grad_J[0] = grad_J[0] - (lam_bda / m) * theta[0]

    return grad_J