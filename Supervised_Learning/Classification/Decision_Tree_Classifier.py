"""
    Multi-class Decision Tree Classifier with binary splits.
    Based on the C.A.R.T. Algorithm, but can only take continuous features.
    Notation Used:
        X = Input Attributes
        Y = Target Attribute (selectable)
    
    Available impurity measures:
        A. Gini
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Get Datasets
dat_path = input("Enter file name and path to dataset:")
datset = np.loadtxt(dat_path, delimiter=",")
tgt = int(input("Enter index of target attribute: ")) 
n_samples, n_features = datset.shape
Y = datset[:, tgt]
X = np.delete(datset, tgt, axis = 1)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
n_features = X_train.shape[1]

# Utilities
def gini(arr): # Gini, G = 1 - sum(Pk^2) | Calculate Gini for candidate splits
    if arr.shape[0] == 0:
        return 0
    class_counts = np.bincount(arr.astype(int)) # Number of samples belonging to each class.
    class_probs = class_counts / arr.shape[0] # Probability of each class appearing.
    p_sqrd = np.square(class_probs)
    g = 1 - np.sum(p_sqrd)
    return g

def info_gain():
    pass

# Create Node Class
class Node:
    def __init__(self, data, left = None, right = None, feature = None, threshold = None, depth = 0):
        self.data = data # Indices of samples belonging to current node
        self.left = left # Left child
        self.right = right # Right child
        self.feature = feature # Decision feature
        self.threshold = threshold # Decision threshold
        self.depth = depth # Depth of current node

    def get_data(self): # Returns data belonging to current node
        return X_train[self.data]

    def eval_split(self, attr, val, impurity_type): # Evaluate a candidate split
        parent = self.get_data()
        if parent.shape[0] < 3: # Doesn't meet minimum node size
            return
        left = parent[parent[:, attr] < val]
        right = parent[parent[:, attr] >= val]
        if left.shape[0] == 0 or right.shape[0] == 0:
            return
        match impurity_type.lower():
            case "gini":
                PG = gini(Y_train[self.data]) # Parent node impurity
                LG = gini(Y_train[self.data][parent[:, attr] < val]) # Left node impurity
                RG = gini(Y_train[self.data][parent[:, attr] >= val]) # Right node impurity
                NP = parent.shape[0]
                NL = left.shape[0]
                NR = right.shape[0]
                weighted_gini = (NL * LG)/NP + (NR * RG)/NP # Weighted Gini of the 2 child nodes
                return weighted_gini
            case "info gain":
                pass
            case _:
                raise Exception("Undefined string passed to Node.eval_split().")

    def create_candidates(self): # Create, evaluate candidates & return best
        best_feature = None
        best_threshold = None
        best_gini = np.inf
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
                gini_value = self.eval_split(i, dat[j], impurity_type="gini")
                if gini_value is None:
                    continue
                vs_best = gini_value < best_gini # Check if new split is better than the current best.
                match vs_best: 
                    case 1: # New is better
                        best_gini = gini_value
                        best_threshold = dat[j]
                        best_feature = i
                    case 0: # Unfavourable
                        continue
                    case _:
                        raise Exception("Invalid symbol.")
        return [best_feature, best_threshold]

    def split(self): # Split the parent node
        if self.data.shape[0] < 2: # Smallest possible node
            return
        if self.depth >= model.max_depth: # Maximum depth reached.
            return
        if gini(Y_train[self.data]) == 0: # Pure node
            return
        best = self.create_candidates() # Getting best splitting feature and threshold.
        if best[0] is None:
            return
        parent = self.get_data()
        # Setting data for Left and Right children.
        left_mask = parent[:, best[0]] < best[1]
        right_mask = parent[:, best[0]] >= best[1]
        left_data = self.data[left_mask]
        right_data = self.data[right_mask]
        if left_data.shape[0] < model.min_split or right_data.shape[0] < model.min_split: # New nodes don't meet Node criteria.
            return # Node stays a leaf (feature/threshold remain None)
        self.feature = best[0] # Assigned only after the child-size check passes
        self.threshold = best[1]
        self.left = Node(data=left_data, depth=self.depth + 1) # Create left child
        self.right = Node(data=right_data, depth=self.depth + 1) # Create right child
        self.left.split() # Split left node
        self.right.split() # Split right node

# Create Tree Class
class DTree:
    def __init__(self, max_depth = 8, min_split = 2, stop_split = None, criteria = None):
        self.max_depth = max_depth # Maximum height of tree.
        self.min_split = min_split # Minimum no. of samples in a node to split it.
        self.stop_split = stop_split # Value of the impurity measure at which node won't be split further.
        self.criteria = criteria # Impurity Measure

    def make_tree(self):
        root = Node(data=np.arange(X_train.shape[0]), depth=0)
        root.split()
        self.root = root

    def predict(self, given):
        current = self.root
        while current.left is not None:  # a leaf has no children
            if given[current.feature] < current.threshold:
                current = current.left
            else:
                current = current.right
        class_counts = np.bincount(Y_train[current.data].astype(int))
        return np.argmax(class_counts)
    
    def test(self):
        predictions = np.zeros_like(Y_test)
        for i in range(Y_test.shape[0]):
            predictions[i] = self.predict(X_test[i])
        print("\n\nClassification Report:\n", classification_report(y_true=Y_test, y_pred=predictions))
        print("\n\nConfusion Matrix:\n", confusion_matrix(y_true=Y_test, y_pred=predictions))

# Initialize tree
model = DTree(criteria="gini")
model.make_tree()
model.test()
user_input = np.array(list(map(float, input("Enter feature values:").split())))
prediction = model.predict(user_input)
print("\nPredicted Class: ", prediction)
