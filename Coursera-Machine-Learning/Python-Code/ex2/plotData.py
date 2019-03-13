import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    # Plots the data points with + for the positive exams
    # and o the negative exams. X is assumed to be a Mx2 matrix
    m = len(y)
    
    fig = plt.figure()
    
    # plot the positive and negative examples on a
    # 2D plot, using option 'k+' for the positive
    # examples and 'ko' for the negative examples
    pos_point = np.zeros((1, 2))
    neg_point = np.zeros((1, 2))
    
    # Separating neg point and pos point into 2 separated vector
    for i in range(m):
        if y[i] == 0:
            neg_point = np.vstack((neg_point, np.array([[X[i, 0], X[i, 1]]])))
        else:
            pos_point = np.vstack((pos_point, np.array([[X[i, 0], X[i, 1]]])))
    
    # Remove first [0. 0.] from two those vector
    neg_point = neg_point[1:, :]
    pos_point = pos_point[1:, :]
    
    plt.plot(neg_point[:, 0], neg_point[:, 1], 'yo')
    plt.plot(pos_point[:, 0], pos_point[:, 1], 'k+')