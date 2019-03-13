import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, lam_bda):
    m = len(y)
    J = 0
    
    h = sigmoid(X.dot(theta))
    J = (1 / m) * ( (-1)*y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)) ) + \
        (lam_bda / (2*m)) * ( np.sum(theta**2) - theta[0]**2 )
    
    return J