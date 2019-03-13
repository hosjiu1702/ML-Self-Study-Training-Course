import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lam_bda):
    m = len(y)
    J = 0
    
    theta_ = theta.copy()
    theta_ = np.reshape(theta, (X.shape[1], 1))
    
    h = sigmoid(X.dot(theta_))
    h = h.reshape((m, 1))
    
    J = (1 / m) * ( (-1)*y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)) ) + \
        (lam_bda / (2*m)) * ( np.sum(theta_**2) - theta_[0]**2 )
    
    return J