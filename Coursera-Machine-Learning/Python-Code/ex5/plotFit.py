import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import polyFeatures
from bsxfun import bsxfun
from sub_op import sub_op
from div_op import div_op


def plotFit(min_x, max_x, mu, sigma, theta, degree):
    x = np.arange(min_x - 60, max_x + 65, 0.05)
    x = x.reshape(x.size, 1)
    
    # Map X
    X_poly = polyFeatures(x, degree)
    X_poly = bsxfun(sub_op, X_poly, mu)
    X_poly = bsxfun(div_op, X_poly, sigma)
    
    # Add ones
    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))
    
    # plot
    plt.plot(x.flatten(), X_poly.dot(theta), '--')