"""
Softmax Regression for Multi-class Classification.
Uses Stochastic Gradient Descent (SGD) on Negative Log-Likelihood (NLL).
"""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

dat_path = input("Enter path to the file containing the dataset:")
targ_path = input("Enter path to the file containing the target feature:")
dataset = np.loadtxt(dat_path)
target = np.loadtxt(targ_path, dtype=int)
n_samples, n_parameters = dataset.shape
n_classes = target.shape[1]
parameters = np.zeros((n_classes,n_parameters)) # 1 Class -> 1 Parameter set


# Hyperparameters
n_iter = 1000
learn_rate = 0.01

# Training and testing split
train_test_split = int(0.7*n_samples)
dat_train = dataset[:train_test_split]
targ_train = target[:train_test_split]
dat_test = dataset[train_test_split:,:]
targ_test = target[train_test_split:,:]

# Softmax function 
def sftmax_vctr(i):
    tmp = (dat_train[i] @ parameters.T)
    tmp = tmp - np.max(tmp) # Just in case a value becomes too large
    smx = np.exp(tmp)/(np.sum(np.exp(tmp)))
    return smx # Returns entire softmax vector

# Starting Gradient Descent
for i in range(n_iter):
    sft_i = np.array(sftmax_vctr(i)) # Softmax array for i-th training example
    """
    Gradient of NLL:
    G = (Softmax(i) - Real value(i)) outer data[i]
    """
    err = sft_i - targ_train[i]
    grad = np.outer(err, dat_train[i]) # Have to find: why outer product?
    parameters = parameters - learn_rate * (grad) # SGD update step
    if i == train_test_split-1:
        break

# Testing the model | Metrics
pred = np.argmax(dat_test @ parameters.T, axis = 1) # axis 1 = hztl | 0 = vrtl
true = np.argmax(targ_test, axis = 1)
accuracy = accuracy_score(true, pred)
recall = recall_score(true, pred, average="macro")
precision = precision_score(true, pred, average="macro")
f1 = f1_score(true, pred, average="macro")
con_mtx = confusion_matrix(true, pred)

# Reporting the model's performance
print("\nAccuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("Confusion Matrix:",con_mtx)

# Taking user input
inp = np.array(list(map(float, input("Enter feature values:").split())))
pred_class = np.argmax(inp @ parameters.T)
print("Predicted Class:", pred_class)