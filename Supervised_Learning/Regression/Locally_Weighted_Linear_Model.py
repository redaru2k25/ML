""""
This algorithm is used when the dataset is non linear and Linear Regression would be a poor fit for the data.
It relies on local linearity of the data.
It adds a weight term, so datapoints closer to the input point are given more importance or weight.

Hypothesis function ht(Xi) = T^tX
weight wi = exp((Xi -X)^2/2 Tau^2)
Tau decides the width of gaussian density
cost function J(T) = sum(wi[Yi - h(Xi)]^2)
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt

""""
M = no. of training examples
N = no. of parameters in the dataset
"""
dat_path = input("Enter the path of file containing the the Dataset:").strip("'\"")
target_path = input("Enter the path of the file containing the Target Attribute Vector:").strip("'\"")
dataset = np.loadtxt(dat_path) # loading the dataset
n_samples, n_param = dataset.shape
target = np.loadtxt(target_path) # loading the target attribute
tested_params = {} # Stores Tau and corresponding Parameter vector  

# Setting the hyperparameter

T = [0.20,0.25,0.3,0.35,0.4,0.45,0.50] # Bandwidth (Controls how fast the weight of datapoints drops off)

train_test_split = int(0.7*n_samples) # Splitting the dataset into Training and Testing sets

X_train = dataset[:train_test_split]
Y_train = target[:train_test_split]

errors = [] #Tracks the errors for different Tau values
inp = np.array(list(map(float, input("Enter feature values:").split()))) # Receiving user input

for k in range(len(T)):

    W = np.zeros(train_test_split) # Temporary vector to store the weights

    for i in range(train_test_split):
        Xi = X_train[i]
        temp = (((Xi - inp)**2) / (2*T[k]**2)).sum() # Calculating Gaussian Density
        W[i] = mt.exp(-temp)

    weights = np.diag(W) # Creating the diagonal weights matrix

    """
    I have used the Normal Equation to compute the optimal values of the parameters.
    The Normal Equation is:
    Parameter matrix, Theta = (Xt*W*X)^(-1) * (Xt*W*Y)
    It computes the values in a single iteration
    """

    XtWX_inv = np.linalg.pinv(X_train.T @ weights @ X_train)
    XtWY = X_train.T @ weights @ Y_train
    parameters = XtWX_inv @ XtWY # Computing the parameter matrix

    err2 = 0 # Computing the Sum of Squared Errors on the Training dataset

    for j in range(train_test_split, n_samples):
        err2 = err2 + (dataset[j] @ parameters - target[j])**2
    tested_params[T[k]] = parameters
    errors.append(err2)

plt.figure() # Plotting Tau vs SSE to find the optimal value of Tau
plt.scatter(T, errors)
plt.xlabel("Tau (T)")
plt.ylabel("SSE")
plt.title("Tau (T) vs SSE")
plt.grid()
plt.show()

Tau = T[errors.index(min(errors))] # Choosing optimal Tau
parameters = tested_params[Tau] # Retrieving the Parameter vector @ optimal Tau
# Receiving user input for prediction

pred = inp.T @ parameters
print("Predicted value:", pred)