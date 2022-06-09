
################################# # # and 게이트  ###############################################
# # and 게이트 
# from tensorflow.keras import models, layers
# import numpy as np

# X=np.array([[0,0],[0,1],[1,0],[1,1]])
# T=np.array([0,0,0,1])

# model=models.Sequential()   # 모델을 겹겹이 쌓을 수 있다(layer들을 겹겹히 쌓을 수 있다)
# model.add(layers.Dense(units =1  ,activation = 'linear', input_shape=(2,)))   #units: 노드가 1개  actuvation:비선형 전달 함수  input_shape: 노드가 받아들일 수 있는 입력의 크기
#                                                # 'sigmoid'  ->을 사용하면[0,0]이 들어 갔을 때 결과가 1일 확률이[[0.9999]]  시그모이드는 loss function을 크로스 엔트로피를 사용하는게 좋다 출력이 0아니면 1  (binary_crossentropy) 
#                                                # relu  좋은 편


# #learning                                 

# model.compile(optimizer='sgd',loss="mse",metrics=['acc'])   # loss: y값과 target값의 차이가 얼마나 나고 있는지 나타내는 척도  클수록 w 와 b 가 안좋다  optimizer: w 와 b를 줄여주는 역할로 경사하강법(sgd)
#               # adam을 쓰는것이 가장 좋다   local 미디엄을 탈출 하기 위해서 산골짜기 습골 미분가능한 곳이 많음               
# model.fit(X,T,epochs =1000, batch_size = 1, verbose =1) #batch_size n개의 데이터를 한번에 넣어서 평균 하나만 업데이트 한다




# #test
# X_test_loss,Xtest_acc = model.evaluate(X,T)
# print(Xtest_acc)
# Ttest =model.predict(X)   # [0,0,0,1]이 나와야 한다
# print(Ttest)



################################# # #iris 데이터로  ###############################################



# #iris 데이터로 
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from keras import models,layers
# import numpy as np
# X,y=load_iris().data,load_iris().target
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,)

# model=models.Sequential()
# model.add(layers.Dense(units=1,activation='linear',input_shape=(4,)))
# model.compile(optimizer='sgd',loss='mse',metrics=['acc'])
# model.fit(X_train,y_train,epochs=100,batch_size=1, verbose=1)



# #테스트
# Xtest_loss,Xtest_acc=model.evaluate(X_test,y_test)
# print(Xtest_acc)
# ytest=model.predict(X_test)
# print(ytest)





################################# 넘파이사용 퍼셉트론 ###############################################

# # adaline using numpy
# import numpy as np

# alpha = 0.1     #0.001정도 사0용
# training_endnum = 10000
# W = np.random.random((1,2))   #w 값 랜덤
# dataset_num = 4     

# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]])

# T = np.array([0, 0, 0, 1])

# for epochs in range(0,training_endnum) :
#     print("----- %d th epoch -----" % (epochs+1))
#     for j in range(0,dataset_num) :
        
#         ## implement delta rule : w <- w + alpha * e * x 
#         y = np.dot(W,np.transpose(X[j,:]))     #  loss로 mse를 사용하고 optimazer로 sgd를 쓴 선형전달함수를 사용한 퍼셉트론을 학습시키는 루틴     
#         e = T[j]-y
#         dW = alpha * e*X[j,:] 
#         W = W+dW


# for i in range(0,dataset_num) :
#     y = np.dot(W,np.transpose(X[i,:]))
#     print(y)








################################# 다중 퍼셉트론###############################################





# #xor 게이트 
# from tensorflow.keras import models, layers
# import numpy as np

# X=np.array([[0,0],[0,1],[1,0],[1,1]])
# T=np.array([0,1,1,0])

# model=models.Sequential()   
# model.add(layers.Dense(units =5 , activation = 'sigmoid', input_shape=(2,)))
# model.add(layers.Dense(units =5 , activation = 'sigmoid'))
# model.add(layers.Dense(units =1 , activation = 'sigmoid'))
                                             
# model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['acc'])  
                

# model.fit(X,T,epochs =1000, batch_size = 4, verbose =1)

# X_test_loss,Xtest_acc = model.evaluate(X,T)
# print(Xtest_acc)
# Ttest =model.predict(X)   # [0,0,0,1]이 나와야 한다
# print(Ttest)



################################# softmax ###############################################





# from sklearn.datasets import load_iris
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import models, layers


# X, y = load_iris().data,load_iris().target 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# # import tensorflow as tf
# # y_train=tf.keras.utils.to_categorical(y_train,num_classes=3)   # 원햇 인코딩으로 바꾸는 것    굳이 이거 할 필요없다


# model=models.Sequential()
# model.add(layers.Dense(units =3  ,activation = 'softmax', input_shape=(4,))) # 출력결과 3개

# # 아웃풋 0,1,2 -->[1,0,0],[0,0,1][0,1,0] 이런식으로 바꾸어서


# model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=['acc'])    #sparse_categorical_crossentropy 알아서 원햇 인코딩으로 바꿔서 컴파일한다
                        
# model.fit(X_train,y_train,epochs =1000, batch_size = 1, verbose =1) 

# # 멀티에서 relu를 사용하고 마지막 출력단에는 'linear'에 mse
# # 이진으로 출력이 나올때 'sigmoid' binary_crossentropy를 사용
# # 멑리이고 [0,0,1] 식으로 나올려면  원햇 인코딩으로 바꾸고 softmax  ,"categorical_crossentropy"


# #test
# X_test_loss,Xtest_acc = model.evaluate(X_test,y_test)
# print(Xtest_acc)
# model.predict(X_test) #[9.546456, 7,123123 ,0,123123] 이런식으로 나오고

#import numpy as np
#np.argmax(model.predict(X_test),axis=1) # 하면 원하는 [0,2,1]














################################# 데이터 사용하는 법 ###############################################


# #iris 데이터로 
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris

# from sklearn.datasets import load_breast_cancer  # 입력 30개    0,1 출력      ---> sigmioid


# from sklearn.datasets import load_wine    # 입력 13개   풀력 0,1,2

# import numpy as np
# X,y=load_iris().data,load_iris().target #load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,)   # 입력 4개 출력은 개 중에 하나










################################# pca를 사용한 ###############################################

# from sklearn.datasets import load_iris
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA 



# X, y = load_iris().data, load_iris().target

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,)

# pca = PCA(n_components=2)
# pca.fit(X_train)

# pca_X_train = pca.transform(X_train)
# pca_X_test = pca.transform(X_test)
# print(pca_X_train.shape,pca_X_test.shape)

# """activation 

# Linear => mse

# sigmoid => binary_crossentropy

# softmax => sparse_categorical_crossentropy
# """

# from tensorflow.keras import models, layers
# import numpy as np


# # model = models.Sequential()
# # model.add(layers.Dense(units=1, activation='softmax', input_shape=(2,)))
 
# # sigmoid => binary_crossentropy => [ 손실 : -60.4836540222168, 정확도 : 0.6666666865348816] mse => [0.3333333432674408, 0.6666666865348816]
# # relu => binary_crossentropy => [손실 : -5 , 정확도 : 0.666666] [-5.0830793380737305, 0.6666666865348816] mse => [0.0439104363322258, 0.6666666865348816]

# # softmax => sparse_categorical_crossentropy = [0.07778985798358917, 0.9777777791023254]
# model=models.Sequential()
# model.add(layers.Dense(units=7,activation='softmax',input_shape=(2,))) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=3,activation='softmax')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)


# ###
# # model.compile(optimizer = 'adam',  # 최적화 알고리즘 적용(찾아보면 엄청 많으니까 "딥러닝 최적화 알고리즘"이라는 키워드로 찾아보는 것도 좋다)
# #               loss = 'categorical_crossentropy', # loss도 우리가 정의해줄수 있는데 우리가 풀려고 하는 문제(회귀,분류 등)에 따라 달라지니 궁금하면 찾아보자 
# #               metrics = ['accuracy']) # 우리가 볼 지표, 이것도 회귀문제냐 분류문제냐에 따라 다르다 여기에서는 분류 문제이기때문에 Accuracy를 사용한다
# ###
# # 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서(즉, 뉴런의 출력을 0으로 만들어) 과대 적합을 만듣 규제 기법



# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics= ['acc'])

# model.fit(pca_X_train, y_train, epochs=1000, batch_size=1, verbose=1)

# model.evaluate(pca_X_test, y_test, verbose = 2) # 모델 평가

# model.predict(pca_X_test)
