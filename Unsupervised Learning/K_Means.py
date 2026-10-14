"""
    Hard Clustering with K Means
"""
import numpy as np
from sklearn.model_selection import train_test_split

# Utilities
def euclidean_distance(a, b):
    sd = np.sum((a - b) * (a - b), axis = 1)
    return sd ** 0.5

def manhattan_distance(a,b):
    rd = np.sum(np.abs(a-b), axis = 1)
    return rd

# Create K Means Class
class KMeans:
    def __init__(self, n_centroids = 5, distance = 'euclidean', max_iter = 1000):
        self.n_centroids = n_centroids # Number of centroids.
        self.distance = distance # Distance metric (L1/L2)
        self.max_iter = max_iter # Maximum number of iterations allowed.

    def fit(self, X, rand_state): 
        """
            Fits a K Means model to data.
            X is training dataset.
            Rand_state is the seed value for the R.N.G function.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        gen = np.random.default_rng(seed=rand_state) # Randomly pick 'k' points from data.
        self.centroids = X[gen.choice(len(X), self.n_centroids, replace=False)]
        old = np.zeros((self.n_centroids, n_features))
        clusters = np.zeros((X.shape[0]), dtype=int) # Tracks cluster that point i belongs to.
        curr = 0 # Current iteration
        while np.allclose(old, self.centroids) == False and curr < self.max_iter:

            for i in range(n_samples): # Find points belonging to a cluster
                match self.distance.lower():
                    case 'euclidean':
                        dist = euclidean_distance(self.centroids, X[i])
                        clusters[i] = np.argmin(dist)
                    case 'manhattan':
                        dist = manhattan_distance(self.centroids, X[i])
                        clusters[i] = np.argmin(dist)
            old = np.copy(self.centroids)

            for j in range(self.n_centroids):
                members = X[clusters == j]
                if len(members) == 0:      # empty cluster: keep the old centroid
                    continue
                match self.distance.lower():
                    case 'euclidean':
                        self.centroids[j] = np.mean(members, axis=0)
                    case 'manhattan':
                        self.centroids[j] = np.median(members, axis=0)
            curr = curr + 1
        self.labels = clusters # Stores index of clusters that points belong to. 

    def predict(self, data): # Make a predicition on a single input datapoint.
        match self.distance.lower():
            case 'euclidean':
                dist = euclidean_distance(self.centroids, data)
                pred = np.argmin(dist)
            case 'manhattan':
                dist = manhattan_distance(self.centroids, data)
                pred = np.argmin(dist)
        return pred

if __name__ == "__main__":  
    # Get Datasets
    dat_path = input("Enter file name and path to dataset:")
    datset = np.loadtxt(dat_path) # Doesn't take .csv files, only matrices with values seperated by whitespaces

    # Train-Test Split
    X_train, X_test = train_test_split(datset, train_size=0.7, random_state=42)

    # Initialize model
    model = KMeans()
    model.fit(X_train, rand_state=42)

    # Make a prediction on user input
    dat = np.array(list(map(float, input("Enter datapoint: ").split())))
    print("Predicted Cluster: ", model.predict(dat))