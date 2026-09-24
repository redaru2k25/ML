"""
    Support Vector Machine for Binary Classification.
    Linear Kernel
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import minimize

# Get dataset
dat_path = input("Enter file name and path to dataset:")
targ_path = input("Enter file name and path to target set:")
datset = np.loadtxt(dat_path, delimiter=",") 
target = np.loadtxt(targ_path) 
n_samples, n_features = datset.shape

# Split
split = float(input("Enter split ratio: ")) 
split = int(split * n_samples)
dat_train = datset[:split]
dat_test = datset[split:,:]
tgt_train = target[:split]
tgt_test = target[split:]

# Kernel & Hyperparameters
C = 1.0 # Regularization constant

# Training the Model
Kernel = dat_train @ dat_train.T
YY_K = np.outer(tgt_train,tgt_train) * Kernel

# Objective function J(alpha) = 0.5 Yi*Yj alpha-i * alpha-j * Kernel
def calc_alpha (alpha): 
    return 0.5 * alpha @ YY_K @ alpha - np.sum(alpha)

# Gradient of objective
def CA_grad(alpha): 
    return YY_K @ alpha - np.ones(split)

# Constraints
cnstr = [{'type' : 'eq', 'fun' : lambda alpha : alpha @ tgt_train, 'jac' : lambda alpha : tgt_train}]
bounds = [(0,C)] * split
A = np.zeros(split) # Target variable

# Optimized Alpha
res = minimize(calc_alpha, A, jac = CA_grad, constraints=cnstr, bounds=bounds, method='SLSQP')
alpha = res.x

# Choose support vectors
SV = (alpha > 1e-5) & (alpha < C - 1e-5)
b = np.mean(tgt_train[SV] - (alpha * tgt_train) @ Kernel[:,SV]) # Compute bias term b
w = (alpha * tgt_train) @ dat_train # Hyperplane

# Testing the model
test_pred =  (dat_test @ w) + b 
test_pred = np.where(test_pred >= 0, 1, -1)
print("\nClassification Report:\n", classification_report(tgt_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(tgt_test, test_pred))

# Prediction on user input
vals = np.array(list(map(float, input("Enter feature values: ").split())))
pred = (vals @ w) + b 
pred = np.where(pred >= 0, 1, -1)
print("Predicted Class: ", pred)