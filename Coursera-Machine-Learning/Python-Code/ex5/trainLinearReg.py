import numpy as np
import scipy.optimize as opt
from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, lambda_):
    initial_theta = np.zeros((X.shape[1]))
    
    theta = opt.fmin_cg(f = lambda x: linearRegCostFunction(x, X, y, lambda_)[0],
                        x0 = initial_theta,
                        fprime = lambda x: linearRegCostFunction(x, X, y, lambda_)[1],
                        maxiter = 200,
                        disp=0
                       )
    return theta