import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData
from mapFeature import mapFeature

def plotDecisionBoundary(theta, X, y):
    # Plos the data points X and y into a new figure with
    # the decision boundary defined by theta
    # ------------------------------
    # This function plots the data point + for the 
    # positive examples and o the negative examples.
    # X is assumed to be a either
    # 1) Mx3 Matrix, where the first column is an all ones
    #    columns for the intercept
    # 2) MxN, N > 3 matrix, where the first column is all-ones\
    
    # plot data
    plotData(X[:, 1:], y)
    
    if X.shape[1] <=3:
        # Only need 2 points to define line, so choose two endpoints
        plot_x = np.array([[np.min(X[:, 1])-2], [np.max(X[:, 1])-2]])
        
        # Caculate the decision boundary line
        plot_y = -1./theta[2] * (theta[1] * plot_x + theta[0])
        
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.25, 50)
        v = np.linspace(-1, 1.25, 50)
        
        u = u.reshape((50, 1))
        v = v.reshape((50, 1))
        
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                f = mapFeature(u[i], v[j])
                f = f.reshape((len(f), 1))
                
                z[i][j] = f.T.dot(theta)
         
        uu, vv = np.meshgrid(u, v)
        
        plt.contour(uu, vv, z, 0)