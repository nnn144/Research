"""
This module contains two functions to read the raw data, which is mainly medical imaging.
When using functions, need to make sure the dimention and the order of the axis are correct.
"""
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_raw(fname, rows, cols, slices):
    """
    Function to read the raw data from file
    
    Input:
        -fname: a string. the path of the file name
        -rows: int. Number of rows of the raw data
        -cols: int. NUmber of the columns of the raw data
        -slices: int. Number of the slices of the raw data
    Output:
        -Return an ndarray of the raw data.
    
    Notice: Normally rows, cols, and slices are the 1st, 2nd, 3rd dimensions of the raw data, correspondingly.
            But it might be different. For example, rows can be the 2nd dimension of the data. One way is to use
            matplotlib to plot the image and check if it is correct.
    """
    
    # initialize an array with all zeros. Numpy default data type is double (64bit-float). I used 32bit just for
    # less memory usage.
    data = np.zeros((rows*cols*slices)).astype("float32")
    
    # read the file
    with open(fname, "rb") as f:
        for k in range(rows*cols*slices):
            # struct.unpack returns a list. first parameter is the format, second is the buffer to be unpacked
            # some format examples: 'h' for short, 'i' for int, uppercase for unsigned. 'f' for float, 'd' for double 
            temp = struct.unpack('H', f.read(2))[0]
            data[k] = temp
    # return 1d-array. Need reshape when plotting the image
    return data

def write_raw(fname, data):
    """
    Function to write the 1D-array data to raw binary file.
    
    input:
        -fname: string. file name of the output raw binary data.
        -data: 1D array. an numpy 1D array
    """
    with open(fname, "wb") as f:
        file_byte_array = bytearray(data)
        f.write(file_byte_array)
    
    return

# a test. Change the file path and the dimensions to your case. Here is just an example.
# Also, change the data type in function "read_raw()"
if __name__ == "__main__":
    # read the data from file
    data = read_raw("./sinogram.raw", 128, 128, 120)
    
    # reshape the data, plot the image, and save it
    # in this example, every sinogram slice is 128x128, so the first sinogram is data[0:128x128]
    sino = data[:128*128].reshape((128, 128))
    
    # plot this image and save it
    plt.imshow(sino, cmap="gray")
    plt.savefig("test.png")
