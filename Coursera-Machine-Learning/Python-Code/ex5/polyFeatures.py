import numpy as np

def polyFeatures(X, degree):
    # Init pre-shape for X-poly
    X_poly = np.zeros((X.shape[0], degree))
    
    # Loop over each column of X_poly to mapping features
    # i-th is feature^i
    for col in range(X_poly.shape[1]):
        X_poly[:, col] = X.ravel()**(col+1)
    
    return X_poly