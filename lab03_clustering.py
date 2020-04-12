import codecs
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import OrderedDict

def read_file(path):
    data = []
    file = open(path, 'r', encoding='utf-8')
    lines = file.readlines()
    del lines[0]
    for line in lines:
        line = line.replace('\n', '')
        line = line.split('\t')
        data.append(line[5:7])
    
    return data 

def draw_graph(data, labels):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.show()

def get_euclidean(data1, data2):
    if len(data1) != len(data2) :
        return -1

    res = 0
    for i in range(0, len(data1)):
        res += (float(data1[i]) - float(data2[i]))**2
    return res**0.5

def dbscan(data):
    X = np.array(data)
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(X)
    return clustering.labels_

def agglomerative_clustering(data):
    X = np.array(data)
    clustering = AgglomerativeClustering(n_clusters=8, affinity='Euclidean', linkage='complete').fit(X)
    return clustering.labels_


class KMeans:
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.cluster =  OrderedDict()
    
    def init_center(self):
        index = random.randint(0, self.n)
        index_list = []
        for i in range(self.n):
            while index in index_list:
                index = random.randint(0, self.n)
            index_list.append(index)
            self.cluster[i] = {'center': self.data[index], 'data': []}

    def clustering(self, cluster):
        #데이터 하나씩 꺼냄
            #중심점 하나씩 꺼냄
                #해당 데이터 & 해당 중심점의 거리를 유클리디안에서 꺼냄
                #꺼낸 값을 비교한다. 더 작다면(if min 사용)
                    #min_d, min_index ->update
            #cluster에 삽입
        
        new_cluster = cluster

        for v in new_cluster.values(): # new_cluster data 초기화
            v['data'] = []

        for p in self.data:
            min_d = 1000
            min_k = -1
            for key, value in cluster.items():
                center = value['center']
                dis = get_euclidean(p, center)
                if dis == -1 :
                    print("distance error!")
                    return False
                
                if min_d > dis:
                    min_d = dis
                    min_k = key

            new_cluster[min_k]['data'].append(p)
        
        return new_cluster

    def update_center(self):
        o_center = [] # update 전 center
        n_center = [] # update 후 center

        for v in self.cluster.values():
            o_center.append(v['center'])

            sum = [0,0]
            l = len(v['data'])
            for d in v['data']:
                sum += d
            avg = sum / l

            v['center'] = avg
            n_center.append(avg)

        if np.array_equal(o_center, n_center) :
            return False
        else :
            return True


    def update(self):
        while self.update_center() :
            self.cluster = self.clustering(self.cluster)

    def fit(self):
        self.init_center()
        self.cluster = self.clustering(self.cluster)
        self.update()

        result, labels = self.get_result(self.cluster)
        draw_graph(result, labels)

    def get_result(self, cluster):
        result = []
        labels = []
        for key, value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)

        return np.array(result), labels


if __name__ == '__main__':
    covidData = []
    covidData = read_file('/Users/user/Desktop/보윤/충남대/3-1/알고리즘응용/covid-19.txt')

    #정규화
    scaler = MinMaxScaler()
    covidData_norm = scaler.fit_transform(covidData)

    #DBSCAN
    dbs_l = dbscan(covidData_norm)
    draw_graph(covidData_norm, dbs_l)

    #AgglomerativeClustering
    agg_l = agglomerative_clustering(covidData_norm)
    draw_graph(covidData_norm, agg_l)

    #KMeans
    km = KMeans(covidData_norm, 8)
    km.fit()
