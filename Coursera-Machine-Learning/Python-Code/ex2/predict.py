import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    # Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    # This function computes the predictions for X using a
    # threshold at 0.5 (i.e, if sigmoid(theta.T.dot(x)) >= 0.5, predict 1)
    m = X.shape[0]
    p = np.zeros((m, 1))
    
    # Make predictions using your learned logistic function parameters.
    # You should set p to a vector of 0's and 1's
    
    # hypothesis on each of training examples
    h = sigmoid(X.dot(theta))
    
    for index in range(m):
        if h[index] >= 0.5:
            p[index] = 1
        else:
            p[index] = 0
            
    return p