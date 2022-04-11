from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LinearRegression
data = load_iris()



X,y=data.data,data.target
reg = LinearRegression()
reg.fit(X, y)
y_pred=reg.predict(X)  
print(y,np.round(y_pred))    # 일차식으로 분류하므로 반올림 해야함
