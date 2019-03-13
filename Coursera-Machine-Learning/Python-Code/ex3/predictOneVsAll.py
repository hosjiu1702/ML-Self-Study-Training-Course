import numpy as np

def predictOneVsAll(all_theta, X):
    # Predict the label for a trained one-vs-all classifier. The labels
    # are in range 1..K where K = all_theta.shape[0]
    # 
    # This function will return a vector of predictions
    # for each example in the matrix X. Note that X contains
    # the examples in rows. all_theta is a matrix where the i-th row
    # i a trained logistic regression theta vector for the i-th class.
    # You should set p to a vector of values from 1..K (e.g., p = [1; 3; 1; 2])
    # predicts classes 1, 3, 1, 2 for 4 examples.
    #
    # You should set p to a vector of preditions
    # (from 1 to num_labels)
    #
    # Hints: This code can be done all vectorized using the max function.
    # In particular, the max function can also return the index of the max
    # element.
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    
    p = np.zeros((m, 1))
    
    X = np.hstack((np.ones((m, 1)), X))
    
    index = np.zeros((m, 1))
    
    # Probabilities of every training examples
    z = X.dot(all_theta.T) # (5000, 10)
    p = np.argmax(z, axis=1) + 1
    
    return p