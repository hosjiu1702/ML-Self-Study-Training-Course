import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def load_mnist():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    
    return X, y

def main():
    print('loading mnist ...')
    load_mnist()

if __name__ == ' __main__':
    print('running ...')
    main()
