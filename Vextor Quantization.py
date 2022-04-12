#백터양자화
#비지도학습 입력 데이터만 있을때
#데이터 특징들을 뽑아냅
#비슷한것 끼리 묶기 군집화
#데이터를 많이보면 모델 형성


#클러스터링 -> 거리가 작을 수 록 비슷한 데이터다.
# k-means



# from sklearn.cluster import KMeans
# import numpy as np
# X = np.array([[1, 2], [1, 4], [1, 0],  [10, 2], [10, 4], [10, 0]])


# import matplotlib.pyplot as plt
# plt.scatter(X[:,0],X[:,1])



# kmeans = KMeans(n_clusters=2) #2개의 클러스터 찾기
# kmeans.fit(X) 
# kmeans.predict(X)
# #kmeans.labels_   # y가 필요 없다
# kmeans.predict([[0, 0], [12, 3]])



#3개의 클러스터로 클러스티를 하고
#predict해서 싷제 타겟과 비교


from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris
X = load_iris().data
kmeans = KMeans(n_clusters=3) # 모델 만들기 (3개의 클러스터 )
kmeans.fit(X) 
k_pred=kmeans.predict(X)

print(k_pred,"---",load_iris().target)  # 수가 중요한것이 아니다 비슷한 것 끼리 모인것
print(abs(k_pred-load_iris().target))


kmeans.cluster_centers_   # 클러스터 중심 찾는 코드
