import numpy as np

def mapFeature(X1, X2):
    # Feature mapping function to polynomial features
    #
    # This function maps the two input features
    # to quadratic features used in the regularization exericise
    #
    # Return a new feature array with more features, comprising of
    # X1, X2, X1**2, X2**2, X1*X2, ...
    #
    # Inputs X1, X2 must be the same size
    degree = 6
    
    out = np.ones((X1.shape)) # X1.shape = (118, 1) => X1.shape[0] = 118
    
    for i in range(1, degree + 1):
        for j in range(i+1):
            temp = X1**(i-j) * X2**(j)
            # out = np.concatenate((out, temp), axis=1)
            out = np.hstack((out, temp))

    return out