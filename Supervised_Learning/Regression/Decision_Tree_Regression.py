"""
    Decision Tree for Regression with binary splits.
    Based on the C.A.R.T. Algorithm.
    Notation Used:
            X = Input Attributes
            Y = Target Attribute (selectable)
    Available Split Criteria:
        1. Sum of Squared Errors
        2. Mean Absolute Error
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Get Datasets
dat_path = input("Enter file name and path to dataset:")
datset = np.loadtxt(dat_path) # Doesn't take .csv files, only matrices with values seperated by whitespaces
tgt = int(input("Enter index of target attribute: ")) 
n_samples, n_features = datset.shape
Y = datset[:, tgt]
X = np.delete(datset, tgt, axis = 1)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
n_features = X_train.shape[1]

# Utilities
def SSE(arr): # SSE = (yi - y_median)**2 
    avg = np.mean(arr)
    return np.sum((arr - avg)**2)
    
def MAE(arr): # MAE = sum(|yi - median y|) of left and right splits
    med = np.median(arr)
    dev = np.absolute(arr - med) / arr.shape[0]
    return np.sum(dev)

# Create Node Class
class Node:
    def __init__(self, data, left = None, right = None, feature = None, threshold = None, depth = None):
        self.data = data # Indices of samples belonging to current node
        self.left = left # Left child
        self.right = right # Right child
        self.feature = feature # Decision feature
        self.threshold = threshold # Decision threshold
        self.depth = depth # Depth of Node

    def get_data(self): # Returns data belonging to current node
        return X_train[self.data]
    
    def eval_split(self, attr, val, impurity_type): # Evaluate Candidate Split
        parent = Y_train[self.data]
        feat = X_train[self.data, attr]
        if parent.shape[0] < model.min_split:
            return
        left = parent[feat < val]
        right = parent[feat >= val]
        if left.shape[0] < model.min_leaf_size or right.shape[0] < model.min_leaf_size:
            return
        match impurity_type.lower():
            case "sse":
                LS = SSE(left)
                RS = SSE(right)
                PS = SSE(parent)
                gain = PS - (LS + RS)
                return gain
            case "mae":
                LE = MAE(left)
                RE = MAE(right)
                PE = MAE(parent)
                gain = PE - (LE * left.shape[0] / parent.shape[0] + RE * right.shape[0] / parent.shape[0])
                return gain
            case _:
                raise Exception("Undefined string passed to Node.eval_split().")

    def create_candidates(self, impurity):
        best_feature = None
        best_threshold = None
        best_gain = - np.inf
        parent = self.get_data()
        for i in range(n_features):
            dat = parent[:, i]
            dat = np.sort(dat)
            # Don't create a candidate between equal values
            dat = dat[np.concatenate(([True], dat[1:] != dat[:-1]))]
            if dat.shape[0] < 2:
                continue
            dat = (dat[1:] + dat[:-1])/2 # Create candidate thresholds
            for j in range(dat.shape[0]):
                gain = self.eval_split(i,dat[j], impurity_type=impurity.lower())
                if gain is None:
                    continue
                vs_vest = gain > best_gain
                match vs_vest:
                    case 1: # Current feature is best.
                        best_feature = i
                        best_threshold = dat[j]
                        best_gain = gain
                    case 0: # Unfavorable
                        continue
                    case _:
                        raise Exception("Invalid gain value.")
        return [best_feature, best_threshold]

    def split(self):
        if self.data.shape[0] < model.min_split: # Smallest possible node
            return
        if self.depth >= model.max_depth: # Maximum depth reached.
            return
        match model.criteria.lower(): # Node data variance is very low
            case "sse":
                if SSE(Y_train[self.data]) <= model.stop_split:
                    return
            case "mae":
                if MAE(Y_train[self.data]) <= model.stop_split:
                    return
            case _:
                raise Exception("Invalid Model Criteria.")
        if np.ptp(Y_train[self.data]) == 0: # All values are same
            return   
        best = self.create_candidates(impurity=model.criteria) # Picking best candidate split
        # Do not change name of initialized DTreeR object
        if best[0] is None:
            return
        parent = self.get_data()
        # Setting data for Left and Right children.
        left_mask = parent[:, best[0]] < best[1]
        right_mask = parent[:, best[0]] >= best[1]
        left_data = self.data[left_mask]
        right_data = self.data[right_mask]
        if left_data.shape[0] < model.min_leaf_size or right_data.shape[0] < model.min_leaf_size: # New nodes don't meet Node/Model criteria.
            return # Node stays a leaf (feature/threshold remain None)
        self.feature = best[0] # Assigned only after the child-size check passes
        self.threshold = best[1]
        self.left = Node(data=left_data, depth=self.depth + 1) # Create left child
        self.right = Node(data=right_data, depth=self.depth + 1) # Create right child
        self.left.split() # Split left node
        self.right.split() # Split right node
        

# Create Tree Class
class DTreeR:
    def __init__(self, max_depth = 8, min_split = 4, stop_split = 0, min_leaf_size = 2, criteria = "SSE"):
        self.max_depth = max_depth # Maximum depth of tree
        self.min_split = min_split # Minimum data in a node to split it
        self.stop_split = stop_split # Minimum value of model criterion to stop splitting
        self.min_leaf_size = min_leaf_size # Minimum number of elements in a leaf node
        self.criteria = criteria

    def make_tree(self): # Create tree
            root = Node(data=np.arange(X_train.shape[0]), depth=0)
            root.split()
            self.root = root
    
    def predict(self, given): # Predict a value based on user input
        current = self.root
        while current.left is not None:  # a leaf has no children
            if given[current.feature] < current.threshold:
                current = current.left
            else:
                current = current.right
        match self.criteria.lower():
            case "mae":
                return np.median(Y_train[current.data])
            case "sse":
                return np.mean(Y_train[current.data])

    def test(self): # Test the model
        predictions = np.zeros_like(Y_test)
        for i in range(Y_test.shape[0]): # Prediction loop
            predictions[i] = self.predict(X_test[i])
        # Display Accuracy Metrics
        print("\n\nRegression Report:")
        print("Root Mean Squared Error (RMSE) = ", root_mean_squared_error(Y_test, predictions))
        print("Mean Absolute Error = ", mean_absolute_error(Y_test, predictions))
        print("R2 Score = ", r2_score(Y_test, predictions))
    

# Create and test model
model = DTreeR()
model.make_tree()
model.test()

# Receive user input and make a prediction on it
user_input = np.array(list(map(float, input("Enter feature values:").split())))
prediction = model.predict(user_input)
print("Predicted Value: ", prediction)