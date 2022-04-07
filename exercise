
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data)




data.data[0]





#오늘의 실습
#1.breast cancor data 셋 load
# train/test 데이터 나누기
# lr/knn/dt/svm 네개의 모델로 만들고
#test 데이터로 검증해서 성능보기




from sklearn.datasets import load_breast_cancer  # data load
data = load_breast_cancer()
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score # 
from sklearn.model_selection import train_test_split   # 데이터 나누기 
X,y =data.data,data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)





#LinearRegression
reg = LinearRegression() #모델 만들기
reg.fit(X_train,y_train) # 학습하기
y_pred= np.round(reg.predict(X_test))  # 1차식이기때문에 
accuracy_score(y_test,y_pred)




#KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train) 
y_pred=neigh.predict(X_test)  
accuracy_score(y_test,y_pred)





#DT
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)  
accuracy_score(y_test,y_pred)





#SVM
clf2=SVC()
clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)  
accuracy_score(y_test,y_pred)




# 더 바른 코드 
# mdl=[LinearRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),SVC()]
# res=[]
# clf=[]
# for model in mdl:
#     model.fit((X_train,y_train)
#     res.append(accuracy_score(np.round(model,predict(X_test)),y_test))  
#     clf.append(model)
# print(res)              



