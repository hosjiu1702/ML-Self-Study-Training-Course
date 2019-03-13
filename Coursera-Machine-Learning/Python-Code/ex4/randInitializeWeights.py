import numpy as np

def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, L_in + 1))
    
    epsilon_init = 0.12
    W = np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init
    
    return W