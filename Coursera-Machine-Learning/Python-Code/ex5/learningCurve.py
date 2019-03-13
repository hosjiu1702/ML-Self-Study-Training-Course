import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def learningCurve(X, y, Xval, yval, lambda_):
    m = len(y)
    error_train = np.zeros((X.shape[0], 1)).ravel()
    error_val = np.zeros((X.shape[0], 1)).ravel()
    theta = np.zeros((X.shape[1], 1))
    for i in range(m):
        theta = trainLinearReg(X[:i+1, :], y[:i+1], lambda_)
        error_train[i] = linearRegCostFunction(theta, X[:i+1, :], y[:i+1], 0)[0]
        error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)[0]

    return error_train, error_val 