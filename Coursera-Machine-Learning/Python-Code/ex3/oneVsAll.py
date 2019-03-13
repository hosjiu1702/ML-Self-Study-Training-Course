import numpy as np
import scipy.optimize as opt

from lrCostFunction import lrCostFunction
from gradient import gradient

def oneVsAll(X, y, num_labels, lam_bda):
    m = X.shape[0]
    n = X.shape[1]
    
    all_theta = np.zeros((num_labels, n+1))
    
    X = np.hstack((np.ones((m, 1)), X)) # (5000, 401)
    
    options = {'maxiter': 50,
              'disp': False}
    
    for c in range(num_labels):
        init_theta = np.zeros(n+1) # (401, 1)
        
        result = opt.minimize(lrCostFunction, init_theta, (X, (y == c + 1), lam_bda), method='CG', \
                              jac=gradient, options=options)
        all_theta[c, :] = result.x
    
    return [all_theta]