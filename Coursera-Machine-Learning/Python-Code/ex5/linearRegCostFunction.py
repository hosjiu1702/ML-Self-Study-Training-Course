import numpy as np

def linearRegCostFunction(theta, X, y, lambda_):
    m = len(y)
    J = 0
    
    theta = theta.reshape(theta.size, 1)
    grad = np.zeros(theta.shape)
    
    J = (1 / (2*m)) * np.sum((X.dot(theta) - y)**2) + (lambda_ / (2*m)) * np.sum(theta[1:, :]**2)
    grad = (1 / m) * ((X.dot(theta)-y).T.dot(X))
    grad = grad.T
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

    return J, grad.flatten()