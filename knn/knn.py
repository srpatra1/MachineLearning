import numpy as np
from utils import euclidean_distance
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        distances = [euclidean_distance(x, x1) for x1 in self.X_train]
        k_i = np.argsort(distances)[0:self.k]
        k_nearest_labels = [y_train[i] for i in k_i]
        y_most_common = Counter(k_nearest_labels).most_common(1)
        return y_most_common[0][0]




