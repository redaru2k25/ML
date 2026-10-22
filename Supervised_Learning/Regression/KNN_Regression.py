"""
kNN Regression model with distance based weighting.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# Utils
def euclidean(x, y):
    """
        Euclidean distance of vector X from all points in the matrix Y.\n
        Returns a vector of distance of each point from x.
    """
    return np.sqrt(np.sum((y - x)**2, axis=1))

def manhattan(x, y):
    """
        Manhattan distance of vector X from all points in the matrix Y.\n
        Returns a vector of distance of each point from x.
    """
    return np.sum(np.abs(y - x), axis=1)

def minkowski(x, y, p):
    """
        Minkowski distance of vector X from all points in the matrix Y.\n
        Returns a vector of distance of each point from x.
        p = power.
    """
    return np.sum(np.abs(y - x)**p, axis=1)**(1/p)

def weigh_points(arr, metric):
    """
    Returns a vector of the weights of the points based on values in the distance array (arr).\n
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
class KNN_Regressor:
    def __init__(self, k = 5, distance='euclidean', p=None, weight = 'uniform'):
        """ 
            k = No. of neighbours\n
            Distance metrics available: L1, L2, Minkowski\n
            Available weights: Inverse (1/d), Inv Squared (1/d^2), Uniform (1)\n
            Determines contribution of points based on their distance from the target point.\n
            p = Value of exponent in Minkwski distance.
        """
        if k < 1:
            raise ValueError("K must be greater than zero.")
        self.k = k
        self.distance = distance.lower() 
        if distance.lower() == 'minkowski' and (p is None or p < 1):
            raise ValueError("p must be greater than or equal to 1, if Minkowski distance is selected.")
        
        self.p = p 
        self.weight = weight.lower() 

    def fit(self, X, Y): 
        """
        Stores the dataset with the model.\n
        Use predict() to perform a predction or test the model.\n
        Doesn't partition the dataset. Needs preprocessing.
        """
        self.X_Train = np.asarray(X)
        self.Y_Train = np.asarray(Y).flatten()

    def predict(self, data):
        """
        Makes a prediction on a given datapoint by assigning the weighted average of the k Nearest Neighbours to it.
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
        pred_value = np.sum(knn_weights * knn_y) / (np.sum(knn_weights) + 1e-10) # Predicted value = SUM(Weight(i) * Y(i)) / SUM(weight(i))
        return pred_value

    def test(self, X, Y, print_data=True, metric='MAE'):
        """
        X: set of input features\n
        Y: Set of true output values\n
        print_data: Prints the metrics is set to True; else returns specified metric.\n
        metric: The metric to be returned, to be used only if print_data is set to False.\n
        Tests the model on a testing set & displays the accuracy  metrics or returns them.\n
        Available metrics: RMSE, R2 Score, MAE
        """
        preds = np.array([self.predict(x) for x in X])
        match print_data:
            case True:
                print("\nREGRESSION REPORT:\n")
                print("Root Mean Squared Error(RMSE): ", root_mean_squared_error(Y, preds))
                print("Mean Absolute Error(MAE): ", mean_absolute_error(Y, preds))
                print("R2 Score: ", r2_score(Y, preds))
            case False: # Metric must be returned
                if metric is None:
                    raise ValueError("Parameter metric was not assigned any value.")
                match metric.upper():      
                    case 'MAE':
                        return float(mean_absolute_error(Y, preds))
                    case 'RMSE':
                        return float(root_mean_squared_error(Y, preds))
                    case 'R2 SCORE' | 'R2':
                        return float(r2_score(Y, preds))
                    case _:
                        raise ValueError("Accepted Accuracy Metrics: RMSE, MAE, R2 Score")
        

    def best_k(self, X, Y, error_metric='MAE'):
        """
            X = input validation set\n
            Y = validation set of true values\n
            error_metric = Metric for comparision\n
        """
        k_vals = [2, 5, 10, 25, 50, 75, 100]
        k_vals = [k for k in (2, 5, 10, 25, 50, 75, 100) if k <= len(self.X_Train)]
        if not k_vals:
            k_vals = [len(self.X_Train)]
        errors = []
        for k in k_vals:
            self.k = k
            errors.append(self.test(X, Y, print_data=False, metric=error_metric))
        pick = np.argmax(errors) if error_metric.upper() == 'R2 SCORE' else np.argmin(errors)
        self.k = k_vals[pick]


# Initializing & running the model.
if __name__ == "__main__":
    # Get Datasets
    dat_path = input("Enter file name and path to dataset:")
    targ = input("Enter file name and path to target:")
    datset = np.loadtxt(dat_path) # Doesn't take .csv files, only matrices with values seperated by whitespaces
    target = np.loadtxt(targ)
    # Train-Test-Validation Split
    X_tr, X_te, Y_tr, Y_te = train_test_split(datset, target, train_size=0.7, random_state=42)
    X_fit, X_val, Y_fit, Y_val = train_test_split(X_tr, Y_tr, train_size=0.8, random_state=42)
    # Create Model
    model = KNN_Regressor(weight='inverse')
    model.fit(X_fit, Y_fit)
    model.best_k(X_val, Y_val) # Find best k from validation set     
    model.fit(X_tr, Y_tr) # refit on full train
    model.test(X_te, Y_te) # final test
    # Make prediction
    data = np.array(list(map(float, input("Enter features: ").split())))
    print("Predicted Value: ", model.predict(data))