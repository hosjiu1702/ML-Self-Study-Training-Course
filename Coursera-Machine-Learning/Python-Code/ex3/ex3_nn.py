#!/usr/bin/env python
# coding: utf-8

# # Neural Network

# **Import third-party libraries and modules**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

from displayData import displayData
from predict import predict


# ## 1. Model representation
# Our neural network is shown in figure below. It has 3 layers - an input layer, a hidden layer and an output layer.
# 
# ![](fig/NN_2layer.png)

# Initializing:
# * Size of input layer
# * Size of hidden layer
# * Number of labels

# In[2]:


# Setup the parameters you will use for this exercise
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10


# We have been provied with a set of network parameters $\big(\Theta^{(1)}, \Theta^{(2)}\big)$ already trained by us. These are stored in **ex3weights.mat** and will be loaded by **loadmat** function of scipy library.

# In[3]:


# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta 1 and Theta 2
weights = io.loadmat('ex3weights.mat')

Theta1 = weights['Theta1']
Theta2 = weights['Theta2']


# Load dataset

# In[4]:


## ======================== Part 1: Loading and Visualizing data ===========================
# Load Training data
print('Loading and Visualizing Data ...\n')

data = io.loadmat('ex3data1.mat')

X = data['X']
y = data['y']

m = y.shape[0] # number of training examples


# Display randomly 100 examples in nice grid

# In[ ]:


# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]

displayData(sel)


# ## 2.2 Feedforward Propagation and Prediction
# Now, we will implement feedforward propagatoin for the neural network. We implement the feedforward computation  that computes $h_{\theta}\big(x^{(i)}\big)$ for every example i and returns the associated predictions. Similar to the one-vs-all classification strategy, the prediction from the neural network will be the label that has the largest output $\big(h_{\theta}(x)\big)_{k}$
# 
# We should see that the accuracy is about $97.5\%$

# In[ ]:


## ============================= Part 3: Implement Predict =============================
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use
# the neural network to predict the labels of the training set. This lets
# you compute the training set accuracy
pred = predict(Theta1, Theta2, X)

print('\nTraining Set Acurracy: %f\n' % (np.mean(np.double(pred == y) * 100 )))


# To give you an idea of the network's output, you can also run through the examples one at the a time to see what it is happening.

# In[ ]:


# Randomly peruate examples
rp = np.random.permutation(m)

for i in range(m):
    # Display
    print('\nDisplaying Example Image\n')
    fig = displayData(X[rp[i], :])
    
    pred = predict(Theta1, Theta2, X[rp[i], :])
    print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, np.mod(pred, 10)))
    
    # Pause with quit option
    s = input('Paused - press <Enter> to continue, q to exit:')
    if s == 'q':
        break


# In[ ]:




