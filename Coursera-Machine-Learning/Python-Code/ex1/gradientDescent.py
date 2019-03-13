import numpy as np

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))
    
    for iter_ in range(iterations):
        h = X.dot(theta)
        error_vector = h - y
        grad_cost = (1 / m) * (X.T.dot(error_vector))
        theta = theta - alpha * grad_cost
    
    return [theta, J_history]