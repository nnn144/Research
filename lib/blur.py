import math
import numpy as np
from scipy import signal
"""
PSF function: using simple 2D Gaussian function here
"""
# 2D gaussian function
def gaussian_func(sigma, x, y):
    left = 1 / (2 * math.pi * sigma**2)
    right = math.exp((-x**2 - y**2) / (2 * sigma**2))
    
    return left * right

# generate 2d gaussian filter
def gaussian_filter(size, sigma):
    gfilter = np.zeros((size, size))
    # Note center is (0, 0), suppose 5x5 filter, 
    # then the top left corner is (-2, -2)
    # offset = floor(5/2)
    offset = math.floor(size/2)
    for i in range(size):
        x = i - offset
        for j in range(size):
            y = j - offset
            gfilter[i, j] = gaussian_func(sigma, x, y)
    
    return gfilter
    
# gaussian blur for single image
def gaussian_blur(gaussian_filter, image):
    #print (image.shape, gaussian_filter.shape)
    return signal.convolve2d(image, gaussian_filter, boundary='symm', mode='same')

# gaussian blur for image set (numpy array)
def images_blur(image_set, filter_size, sigma):
    for i, img in enumerate(image_set):
        gf = gaussian_filter(filter_size, sigma)
        image_set[i] = gaussian_blur(gf, img)
    return image_set
