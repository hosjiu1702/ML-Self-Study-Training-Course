import matplotlib.pyplot as plt

def plotData(X, y):
    plt.figure()
    plt.plot(X, y, 'ro')
    plt.xlabel('Populations')
    plt.ylabel('Profits')