from sklearn.datasets import load_iris    
import numpy as np
from sklearn.linear_model import LinearRegression     #입력과 결과사이의 관계를 1차식의 형태로 만들어 표현하는 방법
data = load_iris()



X,y=data.data,data.target
reg = LinearRegression()
reg.fit(X, y)
y_pred=reg.predict(X)  
print(y,np.round(y_pred))    # 일차식으로 분류하므로 반올림 해야함
