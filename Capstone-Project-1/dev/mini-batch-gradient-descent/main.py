import numpy as np
import matplotlib.pyplot as plt

def mini_batch_gd(X, y):
	""" Training with Mini-Batch Gradient Descent
	
	Args:
		X (ndarray): Training set
		y (1-D ndarray): Labels

	Return
	"""

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

def main():
	X, y = twospirals(1000)

	plt.title('training set')
	plt.plot(X[y==0,0], X[y==0,1], '.', label='class 1')
	plt.plot(X[y==1,0], X[y==1,1], '.', label='class 2')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()