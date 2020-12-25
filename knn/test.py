import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN


iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

test_knn = KNN(k=5)
test_knn.fit(X_train, y_train)
y_predicted = test_knn.predict(X_test)
accuracy = np.sum(y_predicted == y_test)/len(y_test)
print(accuracy)

