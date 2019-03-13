import numpy as np
from bsxfun import bsxfun
from sub_op import sub_op
from div_op import div_op

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = bsxfun(sub_op, X, mu)
    
    sigma = np.std(X_norm, axis=0)
    X_norm = bsxfun(div_op, X_norm, sigma)
    
    return X_norm, mu, sigma