import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape(10, 1)
    
    error_train = np.zeros((lambda_vec.shape[0], 1))
    error_val = np.zeros((lambda_vec.shape[0], 1))
    
    for i in range(lambda_vec.shape[0]):
        lambda_ = lambda_vec[i]
        
        # train with lambda[i] to get theta[i]
        theta = trainLinearReg(X, y, lambda_)

        # Get error training and error validation
        error_train[i] = linearRegCostFunction(theta, X, y, 0)[0]
        error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)[0]
    
    return lambda_vec, error_train, error_val