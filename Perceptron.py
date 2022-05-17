# adaline using keras
from tensorflow.keras import models, layers
import numpy as np

# INPUT / OUTPUT DATA
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0,0,0,1])

model = models.Sequential()
model.add(layers.Dense(units = 1,  activation='linear',input_shape=(2,)))

# training    #loss를 최소화시키는 것이 목표
model.compile(optimizer='sgd',   #loss를 최소화시키기 위해 optimaize
             loss="mse",   #w,b 가 얼마나 차이가 나는지 
             metrics = ['acc'])

model.fit(X,T,epochs = 10000, batch_size = 1, verbose = 1)

# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)











from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras import models,layers
import numpy as np
X,y=load_iris().data,load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,)

model=models.Sequential()
model.add(layers.Dense(units=1,activation='linear',input_shape=(4,)))
model.compile(optimizer='sgd',loss='mse',metrics=['acc'])
model.fit(X_train,y_train,epochs=100,batch_size=1, verbose=1)



#테스트
Xtest_loss,Xtest_acc=model.evaluate(X_test,y_test)
print(Xtest_acc)
ytest=model.predict(X)
print(ytest)
