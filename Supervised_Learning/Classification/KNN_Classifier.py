"""
k Nearest Neighbours Classifier
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Utils
def euclidean(x, y):
    """
        Euclidean distance of vector X from all points in the matrix Y.
        Returns a matrix of distance of each point from x.
    """
    return np.sqrt(np.sum((y - x)**2, axis=1))

def manhattan(x, y):
    """
        Manhattan distance of vector X from all points in the matrix Y.
        Returns a matrix of distance of each point from x.
    """
    return np.sum(np.abs(y - x), axis=1)

def minkowski(x, y, p):
    """
        Minkowski distance of vector X from all points in the matrix Y.
        Returns a matrix of distance of each point from x.
        p = power.
    """
    return np.sum(np.abs(y - x)**p, axis=1)**(1/p)

def weigh_points(arr, metric):
    """
    Returns the weights of the points based on values in the distance array (arr).
    Supports inverse & squared inverse weighting.
    """
    match metric:
        case 'uniform':
            return np.ones_like(arr)
        case 'inverse':
            return 1 / (arr + 1e-10)
        case 'inv squared':
            return 1 / (np.square(arr) + 1e-10)
        case _:
            raise ValueError("Invalid weight metric.")
            
# Define kNN class
class KNN_Classifier:
    def __init__(self, k = 5, distance='euclidean', p=0, weight = 'uniform'):
        if k < 1:
            raise ValueError("K must be greater than zero.")
        self.k = k
        self.distance = distance.lower() # Distance metric: L1, L2, Minkowski
        if distance.lower() == 'minkowski' and p is None:
            raise ValueError("p must be greater than or equal to 1, if Minkowski distance is selected.")
        
        self.p = p # Value of exponent in Minkwski distance.
        self.weight = weight.lower() 
        """ 
            Available weights: Inverse (1/d), Inv Squared (1/d^2), Uniform (1)
            Determines contribution of points based on their distance from the target point
        """

    def fit(self, X, Y): 
        """
        Stores the dataset with the model.
        Use predict() to perform a predction or test the model.
        """
        self.X_Train = X
        self.Y_Train = Y

    def predict(self, data):
        """
        Makes a prediction on given datapoint by assigning the most common (modal) class of the k Nearest Neighbours to it.
        """
        match self.distance: # Get k nearest points
            case 'euclidean':
                distances = euclidean(data, self.X_Train)
            case 'manhattan':
                distances = manhattan(data, self.X_Train)
            case 'minkowski':
                distances = minkowski(data, self.X_Train, self.p)
            case _:
                raise ValueError("Invalid Distance measure.")

        indices = np.argsort(distances)[:self.k] # Contains indices of neighbour points.
        knn_distances = distances[indices] # Contains distance from each neighbour point to input point.
        knn_y = self.Y_Train[indices] # Class labels of neighbours.
        knn_weights = weigh_points(knn_distances, self.weight) # Weights of neighbour points.
        classes = np.unique(self.Y_Train) # Class Labels
        class_weights = np.array([knn_weights[knn_y == c].sum() for c in classes]) # Number of weughted neighbours in each class.
        return classes[np.argmax(class_weights)]
    
    def test(self, X, Y): 
        """
        Tests the model & returns performance data.
        """
        preds = np.array([self.predict(x) for x in X])
        print("Accuracy: ", accuracy_score(Y, preds))
        print("\nClassification Report:\n", classification_report(Y, preds))
        print("\nConfusion Matrix:\n", confusion_matrix(Y, preds))

# Local code on user data
if __name__ == "__main__":  
    
    # Get Datasets
    dat_path = input("Enter file name and path to dataset:")
    targ = input("Enter file name and path to target:")
    datset = np.loadtxt(dat_path, delimiter=",") # Doesn't take .csv files, only matrices with values seperated by whitespaces
    target = np.loadtxt(targ).astype(int)
    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(datset, target, train_size=0.7, random_state=42)

    # Initialize model
    model = KNN_Classifier()

    # Test model
    model.test(X_test, Y_test)

    # Run model on user data
    model.fit(X_train, Y_train)
    data = np.array(list(map(float, input("Enter features: ").split())))
    print("Predicted Class: ", model.predict(data))