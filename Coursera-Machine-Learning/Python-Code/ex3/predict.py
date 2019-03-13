import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    # Pre-processing for computing in numpy.
    if X.ndim == 1:
        X = X.reshape(1, len(X)) # (1, len(X))   
    
    m = X.shape[0] # num of training examples
    num_labels = Theta2.shape[0]
    
    p = np.zeros((m, 1))
    
    print('X.shape  = {0}'.format(X.shape))
    # Add bias
    X = np.hstack((np.ones((m, 1)), X))
    
    for j in range(m):
        # Get training examples at j-th of X dataset
        x = X[j, :] # (401, )
        x = x.reshape((X.shape[1], 1)) # (401, 1)
        # Compute z2
        a1 = x # (401, 1)
        z2 = Theta1.dot(a1) # Theta1: (25, 401) --> (25, 1)
        # Compute a2
        a2 = sigmoid(z2) # --> (25, 1)
        
        # Compute z3
        a2 = np.vstack((1, a2)) # --> (26, 1)
        z3 = Theta2.dot(a2)
        
        # Compute a3
        a3 = sigmoid(z3)
        
        # Compute output
        h = a3
        
        # Get element and its index has highest prob in output vector above
        max_index = np.argmax(h) # 0 .. 9
        
        # Convert to output p
        p[j] = max_index + 1

        '''
        # Fix zero index for predicting
        if max_index > 0:
            p[j] = 
        else:
            p[j] = max_index
        '''
    return p