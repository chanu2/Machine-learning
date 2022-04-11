X=[[0],[1],[2],[3]]
y=[0,0,1,1]


from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# clf.fit(X,y)
# y_pred=clf.predict(X,y)
# print(y,y_pred)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
data = load_iris()
X,y=data.data,data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)
