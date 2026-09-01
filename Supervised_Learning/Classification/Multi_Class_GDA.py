"""
    Gaussian Discriminant Analysis for Multi Class Classification.
    Uses Negative Log Likelihood Minimization.
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
dat_path = input("Enter file name and path to dataset:")
targ_path = input("Enter file name and path to target set:")
datset = np.loadtxt(dat_path) # Delimiter ' '
target = np.loadtxt(targ_path) # Delimiter ' '
n_samples,n_features  = datset.shape
n_classes = target.shape[1]

# Training and Testing split
train_test_split = int(0.7 * n_samples)
dat_train = datset[:train_test_split]
targ_train = target[:train_test_split]
dat_test = datset[train_test_split:,:]
targ_test = target[train_test_split:,:]

# Parameter estimation
phi = np.sum(targ_train, axis = 0).reshape(1,-1)/train_test_split # phi = proability of class i occuring
mean = np.array([dat_train[targ_train[:,i] == 1].mean(axis=0) for i in range(n_classes)]) # Class Means for all classes 
cov = np.cov(dat_train, bias = True, rowvar = False) # Covariance matrix (diagonal entry C-ii has intra-class variance)
# Other entries C-ij contain covariance of i-th and j-th features

# MV Gaussian pdf-function -> P(X | Y = i) follows MV Gaussian
def mv_gaussian(X, n, cov,i): # n = no. of features & i = class
    norm = pow(2*3.14159, n/2) * pow(np.linalg.det(cov),1/2) # Normalizing const.
    xmm = X - mean[i]
    c_inv = np.linalg.inv(cov)
    epw = np.einsum('ij,jk,ik->i',xmm,c_inv,xmm) # Mahalanobis distance ^ 2
    # Gotta fully undertand einsum
    val = np.exp(-(epw)/2)/norm # Core function
    return val

# Testing the model 
Px_yi = np.array([mv_gaussian(dat_test, n_features, cov,i) for i in range(0,n_classes)]) # P(x | y = i) for all classes, shape (n_classes, n_test)
Px = np.sum(phi.T * Px_yi, axis=0) # P(X) = sum over i of P(X|Y=i)*P(Y=i), one value per test sample -> shape (n_test,)
Pyi_x = (phi.T * Px_yi) / Px # P(y=i | x) = P(x | y=i)*P(y=i)/P(X), shape (n_classes, n_test)
test_pred = np.argmax(Pyi_x, axis=0) # Prediction matrix -> class index per test sample

# Computing Model Metrics
targ_test = targ_test.argmax(axis=1)
accuracy = accuracy_score(targ_test, test_pred)
con_mtx = confusion_matrix(targ_test, test_pred)
print("MODEL METRICS:")
print(f"Accuracy: {accuracy}\n")
print("\nConfusion Matrix:\n",con_mtx)
print("\n\nClassification Report:\n", classification_report(targ_test, test_pred))

# Taking user input
inp = np.array(list(map(float, input("Enter the features:").split()))).reshape(1,-1)

# Computing P(Y=i | X)
Pr_x_yi = np.array([mv_gaussian(inp, n_features, cov,i) for i in range(0,n_classes)]).flatten() # P(x | y = i) for all classes, shape (n_classes,)
phi_1d = phi.flatten() # flatten so * works instead of broadcasting into an outer product
Pr_x = np.sum(Pr_x_yi*phi_1d) # P(X)
Pr_yi_x = Pr_x_yi*phi_1d/Pr_x
print("Pr_x_yi:", Pr_x_yi)
print("phi_1d:", phi_1d)
print("Pr_yi_x:", Pr_yi_x)
pred = np.argmax(Pr_yi_x) # Class with highest posterior probability
print("Predicted class: Class ",pred)