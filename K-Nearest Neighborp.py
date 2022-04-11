from sklearn.datasets import load_iris
import numpy as np

data = load_iris()






from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
X,y=data.data,data.target
neigh.fit(X,y)
y_pred=neigh.predict(X)
# print(y_pred,y)
# print(a)
# print(neigh.predict_proba([X[0]]))
# print(neigh.kneighbors([X[0]]))
error=abs(y_pred-y)
error=np.array(error)



print(error.shape)
print(np.nonzero(error))


np.array(np.nonzero(error)).shape[1]

acc= 1-np.array(np.nonzero(error)).shape[1]/error.shape[0]


print(acc)


def acc_measure(y,y_pred):
  error=np.array(abs(y-y_pred))
  return 1-np.array(np.nonzero(error)).shape[1]/error.shape[0]
  
  
  
  acc_measure(y,y_pred)
