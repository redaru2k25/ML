"""
Logistic Regression for Binary Classification
Batch Gradient Ascent was used

Formulae used:
Sigmoid function S = 1 / (1 + exp(-Z)), Z = (Xt * parameters)
error = actual target value - predicted value
Iterations:
    new parameters = old parameters + (learning rate) * X.T * (error)

"""
import numpy as np
dat_path = input("Enter path to the file containing the dataset:").replace("'","")
targ_path = input("Enter path to the file containing the target feature:").replace("'","")
dataset = np.loadtxt(dat_path)
target = np.loadtxt(targ_path)
n_samples, n_parameters = dataset.shape
parameters = np.zeros(n_parameters)

# Hyperparameters
learn_rate = 0.01
n_iter = 1000

# Setting class margin
margin = 0.5

# Training and testing split 
train_test_split = int(0.7*n_samples)
dat_train = dataset[:train_test_split]
targ_train = target[:train_test_split]
dat_test = dataset[train_test_split:,:]
targ_test = target[train_test_split:]

# Model's Classsification metrics
accuracy = 0
recall = 0
precision = 0

# Starting Batch Gradient Ascent
for i in range(n_iter):
    pred = 1/(1 + np.exp(-(dat_train @ parameters)))
    err = targ_train - pred # Calculating the error
    parameters += learn_rate * err @ dat_train 

# Testing the model
prob = 1 / (1 + np.exp(-(dat_test @ parameters)))
guess = (prob > margin).astype(int)

n_TP = 0
n_FP = 0
n_TN = 0
n_FN = 0

for i in range(len(targ_test)):
    predicted = guess[i]
    actual = targ_test[i]
    match actual:
        case 1:
            match predicted:
                case 1:
                    n_TP += 1
                case 0:
                    n_FN += 1
                case _:
                    raise Exception("Error in model prediction")
        case 0:
            match predicted:
                case 1:
                    n_FP += 1
                case 0:
                    n_TN += 1
                case _:
                    raise Exception("Error in model prediction")

# Calculating the metrics 
accuracy = (n_TP + n_TN)/(n_TN + n_FN + n_FP + n_TP)
recall = n_TP/(n_TP + n_FN)
precision = n_TP/(n_TP + n_FP)
print("\n" * 2)
print("Classification Report:\n")
print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}")

# Prediction for user input
values = np.array(list(map(float,input("Enter feature values:").split())))
prediction = 1/(1 + np.exp(-parameters @ values))
print("Predicted class:", int(prediction > margin))