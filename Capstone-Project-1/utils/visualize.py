import numpy as np
import matplotlib.pyplot as plt

def displayData(X, example_width=None):
    # This function displays 2D data
    # stored in X a nice grid. It returns figure handle h
    # and the displayd array if required
    
    # Example_width and example_height are width and height of single image, respectively
    
    # Set example_width automatically if you not passed in
    if example_width == None:
        try:
            example_width = np.round(np.sqrt(X.shape[1])).astype(np.int) # 20.0 with original data from Coursera
        except IndexError:
            X = X.reshape(1, len(X))
            example_width = np.round(np.sqrt(X.shape[1])).astype(np.int)
        
    # Compute rows, cols
    m, n = X.shape # With input 100 examples m = 100, n = 400
    example_height = (n / example_width).astype(np.int) # due to example_with x example_weight = n (20 x 20)

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(np.int) # 10
    display_cols = np.ceil(m / display_rows).astype(np.int) # 10

    # "Distance" between two adjacent images.
    pad = 1
    
    # Set up blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), 
                                pad + display_cols * (example_width + pad)))
    
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            else:
                # Copy the patch
                
                # Get the max value of the patch
                # max_val = np.max(np.abs(X[curr_ex, :]))
                
                # display_array[sub rows , sub cols]
                # Get sub range to slide over
                sub_range_of_height = pad + j * (example_height + pad)
                sub_range_of_width = pad + i * (example_width + pad)
                
                display_array[sub_range_of_height : sub_range_of_height + example_height,
                             sub_range_of_width : sub_range_of_width + example_width] = \
                                np.reshape(X[curr_ex, :], (example_height, example_width))
                curr_ex = curr_ex + 1
        if curr_ex > m:
            break
    
    # Display image
    plt.imshow(display_array)
    
    plt.show()
