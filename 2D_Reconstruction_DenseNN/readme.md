# 2D Reconstruction using fully connected neural network

Training image: U-shaped object
image size: 64x64
number of images:

input: sinogram
output: reconstruction

Structure:
input layer ==> 9 hidden layers ==> one dropout layer ==> output layer
All the layers are fully connected.
Each hidden layer has 10,000 nodes.

Loss function is: square of the Euclidean distance (avoid to use square root)
Optimizer function: Adam optimizer, learning rate 0.001
