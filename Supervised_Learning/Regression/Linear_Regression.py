"""  PSEUDOCODE

dataset matrix structure:
M x N matrix
M = Number of attributes + 1 target row (Y)
N = no. of samples
learn_rate = 0.01
max_pass = 1000
for z: 0 -> max_pass-1
	for i: 0 -> N-1:
		for: j: 0 -> M-2:
			predction[i] = predction[i] + parameter[j] * dataset[j][i]  
		err[i] = prediction[i] - dataset[M-1][i]
	
	for r: 0 -> len(parameter)-1:
		grad = 0
		for k: 0 -> N-1:
			grad = grad + err[k] * dataset[r][k]
		parameter[r] = parameter[r] - grad * learn_rate / N

"""

import numpy as np

n_par = int(input("Enter no. of parameters: "))
n_samples = int(input("Enter no. of samples in dataset: "))

# Dataset contains X1...Xn (Input attributes) and Y (Target Attribute @ Final row)
dataset = np.zeros((n_par + 1, n_samples), dtype=np.float32)
# Parameters matrix; parameters decide importance of an attribute
parameters = np.zeros(n_par, dtype=np.float32)

print("Enter the dataset row-wise (last row should be Y):")
for i in range(n_par + 1):
    dataset[i, :] = [float(x) for x in input().split()]

# Setting Batch Gradient Descent parameters.
learn_rate = 0.01 # Decides size of the leap taken by the algorithm
epochs = 1000 # It represents the total number of steps the algorithm takes before termination. 

pred = np.zeros(n_samples)
err = np.zeros(n_samples)

for epoch in range(epochs):

    # Prediction and error
    for i in range(n_samples):
        prediction = 0
        for j in range(n_par):
            prediction += dataset[j][i] * parameters[j] # Predicted value = sum(Attribute * Parameter)

        pred[i] = prediction
        err[i] = pred[i] - dataset[n_par][i] # Represents how far off our estimate is from the true value

    # Updating the parameter values
    for j in range(n_par):
        gradient = 0
        for i in range(n_samples):
            gradient += err[i] * dataset[j][i]

        parameters[j] -= learn_rate * gradient / n_samples 

print("\nFinal Parameters:")
print(parameters)

print("\nPredicted Values:")
print(pred)

req = np.array(input("Enter the values of the attributes: ").split(), dtype=np.float32)
if len(req) != n_par:
    print("Error: Enter exactly", n_par, "values.")
else:
    pval = np.dot(req, parameters)
    print("Predicted value:", pval)
