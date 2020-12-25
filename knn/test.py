import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=.2, random_state=1234)

