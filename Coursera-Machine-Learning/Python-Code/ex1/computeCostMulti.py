import numpy as np

def computeCostMulti(X, y, theta):
    m = len(y)
    J = 0
    err = X.dot(theta) - y
    sqr_err = err**2
    sum_sqr_err = np.sum(sqr_err)
    J = (1 / (2*m)) * sum_sqr_err
    
    return J