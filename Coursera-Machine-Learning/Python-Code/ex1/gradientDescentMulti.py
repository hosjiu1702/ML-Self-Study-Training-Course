import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1)) # Using visualize gradient of cost function
    
    # Perform a single gradient step on the parameter vector theta
    for i in range(num_iters):
        # Compute gradient of cost function
        h = X.dot(theta)
        err_vec = h - y
        grad_J = (1 / m) * X.T.dot(err_vec)
        
        theta = theta - alpha * grad_J # update theta
        
        # Save the cost J in every iteration
        J_history[i] = computeCostMulti(X, y, theta)
        
    return [theta, J_history]