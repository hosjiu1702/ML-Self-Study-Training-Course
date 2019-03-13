import numpy as np

def computeCost(X, y, theta):
    # init some useful values
    m = len(y) # number of training examples
    
    J = 0
    error = X.dot(theta) - y    
    sqr_error = error**2
    sum_sqr_error = np.sum(sqr_error)
    J = (1 / (2*m)) * sum_sqr_error
    
    return J