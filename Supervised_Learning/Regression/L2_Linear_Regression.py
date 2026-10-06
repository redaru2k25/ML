"""
    Linear Regression with L2 Norm.
    Closed form Equation based implementation
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class L2_Linear:
    def __init__(self, penalty, save_params = False):
        self.penalty = penalty # L2 Norm penalty
        self.save_params = save_params # Option to save model parameters after training

    def _get_data(self): # Gets Dataset
        dat_path = input("Enter file name and path to dataset:")
        datset = np.loadtxt(dat_path) # Doesn't take .csv files, only matrices with values seperated by whitespaces
        tgt = int(input("Enter index of target attribute: ")) 
        Y = datset[:, tgt]
        X = np.delete(datset, tgt, axis = 1)

        # Train-Test Split
        X_tr, self.X_test, Y_tr, self.Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
        # Train-Validation Split (validation set is used only for penalty tuning)
        self.X, self.X_val, self.Y, self.Y_val = train_test_split(X_tr, Y_tr, train_size=0.8, random_state=42)

    def fit(self): # Trains model on Training set
        I = np.eye(self.X.shape[1], self.X.shape[1]) # dimensions: (n_features, n_features)
        I[0, 0] = 0
        self.parameters = np.linalg.solve(self.X.T @ self.X + self.penalty * I, self.X.T @ self.Y)
        match self.save_params:
            case True: # Saves parameters of model to a file
                np.save(r"C:\ML\Models\L2_Linear.npy", self.parameters) 
            case False:
                return

    def save_model(self): # Saves current parameter set
        np.save(r"C:\ML\Models\L2_Linear(1).npy", self.parameters)

    def use_saved_params(self): # Loads saved parameters file
        path = input("Enter filename and path to parameter file: ")
        self.parameters = np.load(path)

    def test(self, display_metrics):
        test_pred = self.X_test @ self.parameters
        match display_metrics:
            case True:
                print("\nRegression Metrics:\n")
                print("R^2 Score: ", r2_score(self.Y_test, test_pred))
                print("Mean Absolute Error: ", mean_absolute_error(self.Y_test, test_pred))
                print("Root Mean Squared Error: ", root_mean_squared_error(self.Y_test, test_pred))
            case False:
                return
    def test_penalties(self, vals, metric = "mae"): # Tests input penalty values and returns best one & parameters according to selected metric
        vals = np.array(vals, dtype=float).reshape(-1)
        I = np.eye(self.X.shape[1], self.X.shape[1])
        I[0, 0] = 0
        performance = np.zeros_like(vals, dtype=float)
        match metric:
            case "mae":
                for i in range(vals.shape[0]):
                    P = np.linalg.solve(self.X.T @ self.X + vals[i] * I, self.X.T @ self.Y) # Uses model's own training set
                    val_pred = self.X_val @ P
                    performance[i] = mean_absolute_error(self.Y_val, val_pred)
            case "rmse":
                for i in range(vals.shape[0]):
                    P = np.linalg.solve(self.X.T @ self.X + vals[i] * I, self.X.T @ self.Y) # Uses model's own training set
                    val_pred = self.X_val @ P
                    performance[i] = root_mean_squared_error(self.Y_val, val_pred)
            case "r2":
                for i in range(vals.shape[0]):
                    P = np.linalg.solve(self.X.T @ self.X + vals[i] * I, self.X.T @ self.Y) # Uses model's own training set
                    val_pred = self.X_val @ P
                    performance[i] = r2_score(self.Y_val, val_pred)
            case _:
                raise ValueError("Invalid metric name passed to test_penalties()")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(vals, performance, marker='o', color='tab:blue', linestyle='-', linewidth=2, markersize=8, zorder=3)
        ax.set_xscale('log')
        ax.grid(True, which="both", linestyle=':', alpha=0.4, color='gray')
        for x in vals:
            ax.axvline(x=x, color='tab:red', linestyle='--', alpha=0.6, linewidth=1.2, zorder=1)
        ax.set_xlabel('Penalty Values', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title('Penalty vs Error', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

        # Select best value
        if metric != "r2":
            return vals[np.argmin(performance)]
        else:
            return vals[np.argmax(performance)]

    def predict(self, data):
        pred = np.dot(data, self.parameters)
        print("Predicted value: ", pred)
        return pred

# Initialize model
model = L2_Linear(penalty=None,save_params=True)
model._get_data()
a = model.test_penalties([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1])
model.penalty = a
model.fit()
model.test(True)
