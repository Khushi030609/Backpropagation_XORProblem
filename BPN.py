#!/usr/bin/env python
# coding: utf-8

# In[15]:


# BPN XOR Problem

import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = 0.1  # Learning rate

        # Weights and biases for the input layer and hidden layer
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        # Weights and biases for the hidden layer and output layer
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def feedforward(self, X):
        # Calculate the output of the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Calculate the output of the output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

    def backpropagate(self, X, y):
        # Calculate the loss
        error = y - self.output

        # Compute the gradients for the output layer
        delta_output = error * sigmoid_derivative(self.output)
        d_weights_hidden_output = np.dot(self.hidden_output.T, delta_output)
        d_bias_output = np.sum(delta_output, axis=0, keepdims=True)

        # Compute the gradients for the hidden layer
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)
        d_weights_input_hidden = np.dot(X.T, delta_hidden)
        d_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)

        # Update the weights and biases
        self.weights_hidden_output += self.lr * d_weights_hidden_output
        self.bias_output += self.lr * d_bias_output
        self.weights_input_hidden += self.lr * d_weights_input_hidden
        self.bias_hidden += self.lr * d_bias_hidden

    def train(self, X, y, epochs):
        for _ in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y)

    def predict(self, X):
        self.feedforward(X)
        return self.output

if __name__ == "__main__":
    # Example dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the neural network
    input_size = 2
    hidden_size = 4
    output_size = 1

    neural_network = NeuralNetwork(input_size, hidden_size, output_size)
    neural_network.train(X, y, epochs=10000)

    # Make predictions
    predictions = neural_network.predict(X)

    print("Predictions:")
    print(predictions)
    
    
    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
    plt.title("Original Data Points")
    plt.show()

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = neural_network.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
    plt.title("Decision Boundary")
    plt.show()



