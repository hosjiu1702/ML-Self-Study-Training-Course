import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    # This function computes cost of using theta as the params
    # for logistic regression and the gradient of the cost
    # w.r.t to the params.
    m = len(y)
    J = 0
    
    # Compute the cost of a particular choise of theta.
    # You should set J to the cost
    # Compute the partial derivatives and set grad to the
    # partial derivatives of the cost w.r.t each params in the theta
    
    # Compute hypothesis
    h = sigmoid(X.dot(theta))
    
    # Compute cost
    J = (1 / m) * ( (-1)*y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)) )
    
    return J