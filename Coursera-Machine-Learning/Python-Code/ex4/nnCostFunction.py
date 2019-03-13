import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    # This function implements the neural network cost function for two layer
    # neural network which performs classification
    #
    # This function compute the cost and gradient of the neural network.
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weigh matrices.
    #
    # The returned parameter grad should be "unrolled" vector of the 
    # partial derivatives of the neural network.
    #
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network.

    nn_params = np.reshape(nn_params, (len(nn_params), 1))

    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))
    
    m = X.shape[0] # number of training datas
    
    # Add bias for training examples
    X = np.hstack((np.ones((m, 1)), X))
    
    ## ========= VECTORIZATION VERSION ========
    # 1. Forward pass
    a1 = X
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2);
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(Theta2.T);
    a3 = sigmoid(z3)
    h = a3
    
    # 2. Compute Cost w Reg
    Y = np.zeros((m, num_labels))
    for ex in range(y.size):        
        if y[ex] != 10:
            gt_index = y[ex] # Ground truth index
            Y[ex, gt_index - 1] = 1
        else:
            gt_index = 9
            Y[ex, gt_index] = 1
    
    reg = (lambda_ / (2*m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))
    J = (1/m) * np.sum((-Y)*np.log(h) - (1-Y)*np.log(1-h), axis=1).sum() + reg
    
    # 3. Backward pass
    delta3 = (h-Y) # (5000, 10)
    delta2 = delta3.dot(Theta2[:, 1:]) * a2*(1-a2)
    Delta2 = a2.T.dot(delta3)
    Delta1 = a1.T.dot(delta2)

    theta1_grad = (1 / m) * Delta1
    theta2_grad = (1 / m) * Delta2
    
    theta1_grad = theta1_grad.T
    theta2_grad = theta2_grad.T

    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]    

    # Unroll gradient
    theta1_grad_r = theta1_grad.reshape(theta1_grad.size, 1)
    theta2_grad_r = theta2_grad.reshape(theta2_grad.size, 1)
    grad_J = np.vstack((theta1_grad_r, theta2_grad_r))
    ## ========= VECTORIZATION VERSION ========
    
    return J, grad_J.flatten()