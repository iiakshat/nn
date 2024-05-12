# Neural Network Implementation using NumPy, Pandas, and Matplotlib on MNIST Dataset

This repository contains the implementation of a neural network from scratch using only NumPy, Pandas, and Matplotlib to classify handwritten digits from the MNIST dataset.

## Overview

The neural network follows a simple two-layer architecture:
- **Input Layer 𝑎[0]:** 784 units corresponding to the 784 pixels in each 28x28 input image.
- **Hidden Layer 𝑎[1]:** 10 units with ReLU activation.
- **Output Layer 𝑎[2]:** 10 units corresponding to the ten digit classes with softmax activation.

## Objectives

1. Preprocess the MNIST dataset.
2. Implement the neural network architecture.
3. Train the neural network using backpropagation and gradient descent.
4. Evaluate the performance of the neural network on the test dataset.

## Prerequisite Understanding

Before diving into the implementation of the neural network using NumPy, Pandas, and Matplotlib on the MNIST dataset, it's essential to have a good understanding of the following concepts:

### 1. Neural Networks
- Understanding the basics of neural networks, including architecture, layers, activation functions, and backpropagation.

### 2. NumPy
- Familiarity with NumPy, a powerful library for numerical computing in Python.
- Knowledge of array manipulation, indexing, broadcasting, and basic linear algebra operations using NumPy.

### 3. Pandas
- Understanding of Pandas, a popular library for data manipulation and analysis in Python.
- Knowledge of loading, exploring, and preprocessing data using Pandas DataFrame.

### 4. Matplotlib
- Basic knowledge of Matplotlib, a library for creating static, animated, and interactive visualizations in Python.
- Understanding of plotting graphs, histograms, and other types of visualizations using Matplotlib.

### 5. Machine Learning Basics
- Understanding of machine learning concepts, including supervised learning, classification, loss functions, and gradient descent optimization.

## Practice Exercises

- Implement small machine learning projects using NumPy and Pandas to gain hands-on experience with data manipulation and preprocessing.
- Explore different types of plots and visualizations using Matplotlib to understand how to present data effectively.
- Experiment with building simple neural networks from scratch using NumPy to reinforce your understanding of neural network architecture and backpropagation.

**Forward propagation**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$

**Backward propagation**

$$dZ^{[2]} = A^{[2]} - Y$$
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
$$dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$$
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$$
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
$$dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$$

**Parameter updates**

$$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
$$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
$$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
$$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$
**Vars and shapes**

Forward propagation

- $A^{[0]} = X$: 784 x m
- $Z^{[1]} \sim A^{[1]}$: 10 x m
- $W^{[1]}$: 10 x 784 (as $W^{[1]} A^{[0]} \sim Z^{[1]}$)
- $B^{[1]}$: 10 x 1
- $Z^{[2]} \sim A^{[2]}$: 10 x m
- $W^{[1]}$: 10 x 10 (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
- $B^{[2]}$: 10 x 1

Backpropagation

- $dZ^{[2]}$: 10 x m ($~A^{[2]}$)
- $dW^{[2]}$: 10 x 10
- $dB^{[2]}$: 10 x 1
- $dZ^{[1]}$: 10 x m ($~A^{[1]}$)
- $dW^{[1]}$: 10 x 10
- $dB^{[1]}$: 10 x 1

## Dataset

The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits (0-9). It consists of a training set of 60,000 examples and a test set of 10,000 examples.

## Approach

1. Load and preprocess the MNIST dataset.
2. Define the neural network architecture.
3. Implement forward propagation, activation functions (ReLU and Softmax), loss function (Cross-Entropy), and backpropagation.
4. Initialize weights and biases.
5. Train the neural network using gradient descent and backpropagation.
6. Evaluate the performance of the trained model on the test dataset
7. 
## Tools

- NumPy: For numerical computing and linear algebra operations.
- Pandas: For data manipulation and preprocessing.
- Matplotlib: For data visualization and plotting.

## Usage

1. Clone the repository:
`git clone https://github.com/iiakshat/nn.git`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run the Jupyter notebook `notebook.ipynb` to train and evaluate the neural network.

## References

- MNIST dataset used: https://www.kaggle.com/competitions/digit-recognizer
- NumPy documentation: https://numpy.org/doc/stable/
- Pandas documentation: https://pandas.pydata.org/pandas-docs/stable/
- Matplotlib documentation: https://matplotlib.org/stable/contents.html
- Understand the math behind: https://www.youtube.com/playlist?list=PLTDARY42LDV4Ic6ZPHIh_CdlPwkKDJmpk
