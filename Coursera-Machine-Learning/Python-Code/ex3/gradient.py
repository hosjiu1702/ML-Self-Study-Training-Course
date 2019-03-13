import numpy as np
from sigmoid import sigmoid

def gradient(theta, X, y, lam_bda):
    m = len(y)
    
    grad_J = np.zeros(theta.shape)
    
    theta_ = theta.copy()
    theta_ = np.reshape(theta, (X.shape[1], 1))

    h = sigmoid(X.dot(theta_))
    h = h.reshape((m, 1))

    grad_J = (1 / m) * X.T.dot(h-y) + (lam_bda / m) * theta_

    grad_J[0] = grad_J[0] - (lam_bda / m) * theta_[0]
    
    grad_J = grad_J.flatten()
    
    return grad_J