import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_iris()
X,y =data.data,data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf=SVC(decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
