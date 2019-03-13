import numpy as np
import sys

sys.path.append('../')

from utils.load_datasets import load_mnist
from utils.visualize import displayData

# Load mnist dataset into X and y
# @X : training examples
# @y : labels
X, y = load_mnist()

# Number of training examples
m = X.shape[0]

# And visualze it on nice grid (randomly select 100 data points)
sel = np.random.permutation(m)
displayData(X[sel[:100], :])

print ('done')
