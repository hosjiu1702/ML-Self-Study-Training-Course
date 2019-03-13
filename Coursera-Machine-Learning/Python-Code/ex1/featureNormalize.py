import numpy as np

def featureNormalize(X):
    X_norm = X
    n = len(X_norm[0, :])
    m = len(X[:, 0])
    
    mu = np.zeros((n, 1))
    sigma = np.zeros((n, 1))
    
    for i in range(n):
        mu[i] = np.sum(X[:, i]) / m
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]
    
    return [X_norm, mu, sigma]