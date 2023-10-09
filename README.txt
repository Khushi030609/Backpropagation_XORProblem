Neural Network for XOR Problem
This repository contains a Python implementation of a neural network designed to solve the XOR problem. The XOR problem is a classic problem in machine learning where the neural network needs to learn a non-linear decision boundary to correctly classify XOR inputs. The neural network architecture is implemented from scratch using Python and the NumPy library.

Overview
The XOR problem is a binary classification problem where the input consists of two binary values (0 or 1) and the output is also binary. The neural network architecture used in this project consists of an input layer, a hidden layer, and an output layer. The network is trained using a feedforward and backpropagation algorithm.

Implementation Details
Activation Function
The sigmoid activation function and its derivative are used in this neural network implementation.
Neural Network Architecture
The neural network consists of an input layer with 2 neurons, a hidden layer with 4 neurons, and an output layer with 1 neuron.
Training
The network is trained using backpropagation with a specified number of epochs. During training, the weights and biases of the network are updated to minimize the error between the predicted output and the actual target values.
Dataset
An example dataset for the XOR problem is provided, consisting of four input-output pairs.
Usage
To use this neural network for solving the XOR problem, follow these steps:

Clone this repository to your local machine.

Ensure you have Python and NumPy installed on your system.

Run the BPN.py script. This script creates and trains the neural network on the XOR dataset.

After training, the script will make predictions on the XOR dataset and display the results.
Results
The neural network is trained to correctly classify the XOR problem, and you will see the predictions after running the script.

Additionally, the script generates two plots:

Original Data Points: A scatter plot of the original XOR dataset, where each point is color-coded based on its class (0 or 1).

Decision Boundary: A contour plot that visualizes the decision boundary learned by the neural network. This plot demonstrates how the neural network separates the data points.

License
This project is licensed under the MIT License. Feel free to use and modify the code as needed for your own projects.