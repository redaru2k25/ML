import numpy as np
"""
Notation:
    m = Number of data points in the Training dataset
    n = Number of attributes in the Training dataset
    X = Design matrix (m x n)
    Y = Target attribute vector (m x 1)
    Q = Parameter vector (n x 1)
    ^ = Inverse operator
"""
"""
    The Normal Equation is used to compute the optimal value of the parameter vector Q for a given dataset. 
    Standard Linear Regression uses the iterative Gradient Descent method to compute Q.
    But it is slow and expensive for extremely large datasets.
    The Normal Equation is a closed-form solution to compute Q in a single step.  
"""
# Reads X from text file
X = np.loadtxt("X.txt")
# Reads Y from text file
Y = np.loadtxt("Y.txt")
m,n = X.shape
XtX = X.T @ X
"""
We assume that the matrix Xt * X is invertible.
If it is not, the pseudo-inverse is used, which returns a least squares solution.
"""
XtXi = np.linalg.pinv(XtX)
# Normal Equation: Q = (Xt * X)^ * Xt * Y
Q = XtXi @ X.T @ Y
print("\nParameter vector Q:")
print(Q)
Inp = list(map(float, input("Enter input feature values: ").split()))
Iarr = np.array([1] + Inp).reshape(1, n)  
op = Iarr @ Q  # Computing the predicted value using the parameter vector Q
print("\nPredicted output:", op[0, 0])
