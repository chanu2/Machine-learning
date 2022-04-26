# x가 현재 축에서 안보일떄 축을 변환시켜 잘보이도록한다.(행렬을 곱해서)

#현재축을 다른축으로 바꾸는 이유 데이터를 가장 잘보이는 축으로 바꿔주는 것이 목적이다
#데이터가 잘 보인다 데이터가 가장 넓게 흩어져 있어야 함 공분산으로 부터



# #공분산으로

# import numpy as np
# a=np.array([[1,2],[3,4]])
# np.linalg.eig(a)


#import numpy as np
#from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#import matplotlib.pyplot as plt
#plt.scatter(X[:,0],X[:,1])



# pca = PCA(n_components=2)   #n com-> 몇차원으로 볼것인가
# pca.fit(X)  # 비지도학습으로 y없음   
# X_pca=pca.transform(X)  #--->축 변환
# plt.scatter(X_pca[:,0],X_pca[:,1])

#print(X_pca)
# pca_component_
# pca_explained_val
