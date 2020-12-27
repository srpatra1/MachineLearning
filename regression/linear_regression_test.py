import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import mse

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

from linear_regression import LinearRegression

regressor = LinearRegression(lr=.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print(mse(y_test, predicted))

m1 = plt.scatter(X_train, y_train)
m2 = plt.scatter(X_test, y_test)
plt.plot(X_test,predicted)
plt.show()
