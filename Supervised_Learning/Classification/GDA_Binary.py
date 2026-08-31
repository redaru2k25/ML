"""
    Gaussian Discriminant Analysis with Log Likelihood Minimization.
    Optimal parameter vlaues are computed in 1 step.
"""
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
dat_path = input("Enter file name and path to dataset:")
targ_path = input("Enter file name and path to target set:")
datset = np.loadtxt(dat_path) # Delimiter ' '
target = np.loadtxt(targ_path) # Delimiter ' '
n_samples,n_features  = datset.shape

# Training and Testing split
train_test_split = int(0.7 * n_samples)
dat_train = datset[:train_test_split]
targ_train = target[:train_test_split]
dat_test = datset[train_test_split:,:]
targ_test = target[train_test_split:,:]

# Parameters
mean = np.zeros((2, n_features))
phi = 0 # Probability of class 1
cov = np.zeros((n_features,n_features )) # Covariance matrix 
# Other entries C-ij contain covariance of i-th and j-th features

# Compute the model parameters
phi = np.sum(targ_train[:,1])
phi = phi/train_test_split # phi = proability of class 1 occuring
c1s = dat_train[targ_train[:, 1] == 1] # Entries of col 1(data) that equal 1
mean[1] = np.mean(c1s, axis=0) # Mean of entries(data) belonging to class 1
c0s = dat_train[targ_train[:, 0] == 1] # Entries of col 0(data) that equal 1 
mean[0] = np.mean(c0s, axis=0) # Mean of entries(data) belonging to class 0
cov = np.cov(dat_train, bias = True, rowvar = False) 

# MV Gaussian pdf-function
def mv_gaussian(X, n, cov,i): #n = no. of features & i = class (0/1)
    norm = pow(2*3.14159, n/2) * pow(np.linalg.det(cov),1/2) # Normalizing const.
    xmm = X - mean[i]
    c_inv = np.linalg.inv(cov)
    epw = np.einsum('ij,jk,ik->i',xmm,c_inv,xmm) # Mahalanobis distance ^ 2
    # Gotta fully undertand einsum
    val = np.exp(-(epw)/2)/norm # Core function
    return val

# Testing the model 
Px_y1 = mv_gaussian(dat_test, n_features, cov, 1) # P(X|Y=1)
Px_y0 = mv_gaussian(dat_test, n_features, cov, 0) # P(X|Y=0)
Px = Px_y1*phi + Px_y0*(1-phi) # P(X) = P(X|Y=1)*P(Y=1) + P(X|Y=0)*(1-P(Y=1))
Py_x = Px_y1 * phi/Px # P(y=1 | x)
test_pred = np.zeros_like(targ_test)
pred_c1 = (Py_x > 0.5).astype(int)
test_pred = np.column_stack([1 - pred_c1, pred_c1]) # Final matrix of model predictions on test set

# Computing Model Metrics
accuracy = accuracy_score(targ_test, test_pred)
recall = recall_score(targ_test, test_pred, average="macro")
f1 = f1_score(targ_test, test_pred, average="macro")
precision = precision_score(targ_test, test_pred, average="macro")
print("MODEL METRICS:")
print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nf1-Score: {f1}")

# Taking user input
inp = np.array(list(map(float, input("Enter the features:").split()))).reshape(1,-1)

# Computing P(Y=1 | X)
Pr_x_y1 = mv_gaussian(inp,n_features,cov,1)
Pr_x_y0 = mv_gaussian(inp,n_features,cov,0)
Pr_x = Pr_x_y1*phi + Pr_x_y0*(1-phi)
Pr_y_x = Pr_x_y1*phi/Pr_x
pred = int(Pr_y_x[0] > 0.5)
print("Predicted class: Class ",round(pred))
